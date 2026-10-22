"""Validate on inorganic materials (DFT, base GRACE's home domain): show that adapting base
GRACE to organics did not worsen it on materials, and that GRACE-OFF cannot replace it there.

Structures are split by the adapted (GRACE-OFF) elements, by default from the teacher model:
  A  no adapted element          control: an adapter changes nothing here, by construction
  B  adapted + other elements    the real forgetting test (oxides, nitrides, halides, ...)
  C  only adapted elements       the only materials GRACE-OFF can run on at all
Every model is run on the structures whose elements it supports (coverage is reported):
force MAE and R², energy MAE per atom (raw, and after per-element offset alignment, since the
models use different energy references), stress MAE, and trained-vs-base paired statistics.

Data: MPtrj parquet shard (e.g. Hugging Face nimashoghi/mptrj, val split) or extended XYZ.

Usage:
    python scripts/validate_materials.py --data mptrj_val.parquet \\
        --models base=~/.cache/grace/GRACE-2L-OMAT-medium-ft-AM "smoke test"=runs/.../saved_model \\
                 GRACE-OFF=models/grace-off/.../saved_model --out validation_materials
"""

import argparse
import json
import time
from pathlib import Path

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import pandas as pd
import yaml
from ase import Atoms
from ase.io import read
from tensorpotential.calculator import TPCalculator
from tqdm import tqdm

import config

EV_A3_TO_GPA = 160.21766
GROUP_TEXT = {"A": "A: no adapted elements (control)",
              "B": "B: adapted + other elements (forgetting test)",
              "C": "C: only GRACE-OFF elements"}


def model_elements(model_dir: Path):
    meta = model_dir / "metadata.yaml"
    return set(yaml.safe_load(meta.read_text())["chemical_symbols"]) if meta.exists() else None


def load_structures(path: Path, max_atoms: int):
    """List of dicts: atoms, energy, forces, stress (eV/Å³, ASE sign) or None."""
    out = []
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df[df.num_atoms <= max_atoms]
        for r in df.itertuples():
            atoms = Atoms(numbers=r.numbers, positions=np.stack(r.positions),
                          cell=np.stack(r.cell), pbc=True)
            out.append({"atoms": atoms, "energy": float(r.energy), "forces": np.stack(r.forces),
                        "stress": np.stack(r.stress).ravel(), "id": r.mp_id})
    else:
        for i, atoms in enumerate(read(path, index=":")):
            if len(atoms) > max_atoms:
                continue
            stress = atoms.get_stress(voigt=False).ravel() if "stress" in atoms.calc.results else None
            out.append({"atoms": atoms, "energy": atoms.get_potential_energy(),
                        "forces": atoms.get_forces(), "stress": stress, "id": str(i)})
            atoms.calc = None
    return out


def sample_groups(structures, adapted, sizes, seed):
    rng = np.random.default_rng(seed)
    groups = {"A": [], "B": [], "C": []}
    for s in structures:
        els = set(s["atoms"].get_chemical_symbols())
        key = "A" if not els & adapted else ("C" if els <= adapted else "B")
        groups[key].append(s)
    for key, n in sizes.items():
        if n and len(groups[key]) > n:
            groups[key] = [groups[key][i] for i in rng.permutation(len(groups[key]))[:n]]
    return groups


def r2(pred, ref):
    return float(1 - np.sum((pred - ref) ** 2) / np.sum((ref - ref.mean()) ** 2))


def aligned_energy_mae(e_pred, e_ref, structures, n_cal, align_elements):
    """Per-atom energy MAE on the evaluation split after fitting per-element offsets for the
    align_elements only (the energy references an adapter changes), on the first n_cal
    structures. Elements outside align_elements keep the model's own reference."""
    natoms = np.array([len(s["atoms"]) for s in structures], dtype=float)
    elements = sorted({e for s in structures for e in s["atoms"].get_chemical_symbols()}
                      & set(align_elements))
    if not elements:
        return 1000 * float(np.abs((e_pred - e_ref) / natoms)[n_cal:].mean())
    comp = np.array([[s["atoms"].get_chemical_symbols().count(e) for e in elements]
                     for s in structures], dtype=float)
    offsets, *_ = np.linalg.lstsq(comp[:n_cal], (e_ref - e_pred)[:n_cal], rcond=None)
    return 1000 * float(np.abs((e_pred + comp @ offsets - e_ref) / natoms)[n_cal:].mean())


def bootstrap_ci(values, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(calc, structures, desc):
    e, f, st = [], [], []
    for s in tqdm(structures, desc=desc, leave=False):
        a = s["atoms"].copy()
        a.calc = calc
        e.append(a.get_potential_energy())
        f.append(a.get_forces())
        st.append(a.get_stress(voigt=False).ravel())
    return np.array(e), f, np.array(st)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(config.materials_shard("val")),
                   help="MPtrj .parquet shard or extended XYZ with DFT labels (default: the val shard)")
    p.add_argument("--models", nargs="+", required=True, help="name=SavedModel; the one named 'base' "
                   "(optional) is the untrained reference for the paired statistics")
    p.add_argument("--groups-by", default=config.TEACHER_MODEL_DIR,
                   help="model whose elements define the adapted set (default: the teacher)")
    p.add_argument("--n-a", type=int, default=200)
    p.add_argument("--n-b", type=int, default=500)
    p.add_argument("--n-c", type=int, default=0, help="0 = all")
    p.add_argument("--max-atoms", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(Path(config.VALIDATION_DIR) / "materials"))
    args = p.parse_args()
    config.setup_device()

    models = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        models[name] = Path(path).expanduser()
    if "base" not in models:
        print("Note: no model named 'base': the paired trained-vs-base statistics are skipped")
    adapted = model_elements(config.require(args.groups_by, "TEACHER_MODEL_DIR"))
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    structures = load_structures(Path(args.data).expanduser(), args.max_atoms)
    all_groups = sample_groups(structures, adapted, {}, args.seed)
    counts = {k: len(v) for k, v in all_groups.items()}
    groups = sample_groups(structures, adapted, {"A": args.n_a, "B": args.n_b, "C": args.n_c}, args.seed)
    print(f"Adapted elements: {' '.join(sorted(adapted))}")
    print(f"Structures (<= {args.max_atoms} atoms): {len(structures)}; groups available "
          f"A {counts['A']}, B {counts['B']}, C {counts['C']}; evaluated "
          + ", ".join(f"{k} {len(v)}" for k, v in groups.items()))

    results, preds = {}, {}
    for name, path in models.items():
        supported = model_elements(path)
        calc = TPCalculator(model=str(path))
        cover = sum(1 for s in structures
                    if supported is None or set(s["atoms"].get_chemical_symbols()) <= supported)
        results[name] = {"coverage_all": cover / len(structures), "groups": {}}
        for key, group in groups.items():
            ok = [s for s in group if supported is None
                  or set(s["atoms"].get_chemical_symbols()) <= supported]
            if not ok:
                results[name]["groups"][key] = {"n": 0, "n_group": len(group)}
                continue
            t0 = time.perf_counter()
            e, f, st = evaluate(calc, ok, f"{name} {key}")
            dft_e = np.array([s["energy"] for s in ok])
            dft_f = np.concatenate([s["forces"] for s in ok])
            nat = np.array([len(s["atoms"]) for s in ok])
            fp = np.concatenate(f)
            g = {"n": len(ok), "n_group": len(group),
                 "force_mae_meV_A": 1000 * float(np.abs(fp - dft_f).mean()),
                 "force_r2": r2(fp.ravel(), dft_f.ravel()),
                 "energy_mae_raw_meV_atom": 1000 * float(np.abs((e - dft_e) / nat).mean()),
                 "energy_mae_aligned_meV_atom": aligned_energy_mae(e, dft_e, ok, max(len(ok) // 5, 1),
                                                                   adapted),
                 "force_mae_median_structure_meV_A": float(np.median(
                     [1000 * np.abs(fi - s["forces"]).mean() for fi, s in zip(f, ok)])),
                 "ms_per_structure": 1000 * (time.perf_counter() - t0) / len(ok)}
            if all(s["stress"] is not None for s in ok):
                dft_s = np.stack([s["stress"] for s in ok])
                g["stress_mae_GPa"] = EV_A3_TO_GPA * float(np.abs(st - dft_s).mean())
            results[name]["groups"][key] = g
            preds[(name, key)] = {"ids": [s["id"] for s in ok], "f": f, "dft_f": dft_f,
                                  "fp": fp,
                                  "f_struct": np.array([1000 * np.abs(fi - s["forces"]).mean()
                                                        for fi, s in zip(f, ok)])}

    # paired trained-vs-base per group (same structures)
    paired = {}
    for name in models:
        if name == "base":
            continue
        for key in groups:
            if (name, key) in preds and ("base", key) in preds and \
                    preds[(name, key)]["ids"] == preds[("base", key)]["ids"]:
                diff = preds[(name, key)]["f_struct"] - preds[("base", key)]["f_struct"]
                lo, hi = bootstrap_ci(diff)
                paired[f"{name}|{key}"] = {"mean_change_meV_A": float(diff.mean()),
                                           "ci95_meV_A": [lo, hi],
                                           "max_abs_change_meV_A": float(np.abs(diff).max())}

    # ---- report ---------------------------------------------------------------------------
    names = list(models)
    for key in groups:
        print(f"\n== {GROUP_TEXT[key]}  ({len(groups[key])} structures)")
        print(f"  {'model':14s} {'coverage':>9s} {'force MAE':>10s} {'median':>8s} {'force R²':>9s} "
              f"{'E raw':>8s} {'E aligned':>10s} {'stress':>8s}   (meV/Å, meV/atom, GPa)")
        for n in names:
            g = results[n]["groups"][key]
            if not g["n"]:
                print(f"  {n:14s} {'0%':>9s}   cannot run: elements not supported")
                continue
            print(f"  {n:14s} {100 * g['n'] / g['n_group']:8.0f}% {g['force_mae_meV_A']:10.1f} "
                  f"{g['force_mae_median_structure_meV_A']:8.1f} "
                  f"{g['force_r2']:9.4f} {g['energy_mae_raw_meV_atom']:8.1f} "
                  f"{g['energy_mae_aligned_meV_atom']:10.1f} {g.get('stress_mae_GPa', np.nan):8.3f}")
    print("\n== Trained vs base on the same materials (force MAE change, + = worse)")
    for k, d in paired.items():
        name, key = k.split("|")
        print(f"  {name:14s} group {key}: {d['mean_change_meV_A']:+7.2f} meV/Å "
              f"(95% CI [{d['ci95_meV_A'][0]:+.2f}, {d['ci95_meV_A'][1]:+.2f}], "
              f"max |change| {d['max_abs_change_meV_A']:.1f})")
    print("\n== Coverage of the whole materials set (structures whose elements the model knows)")
    for n in names:
        print(f"  {n:14s} {100 * results[n]['coverage_all']:6.1f}%")

    (out / "materials_validation.json").write_text(json.dumps(
        {"data": args.data, "adapted_elements": sorted(adapted), "group_sizes_available": counts,
         "results": results, "paired_vs_base": paired, "time": time.strftime("%Y-%m-%d %H:%M")},
        indent=2))
    plot(out, names, groups, results, preds)
    parity(out, names, groups, results, preds)
    np.savez_compressed(out / "predictions.npz", **{
        f"{n}__{k}__{field}": v[field] for (n, k), v in preds.items() for field in ("fp", "dft_f")})
    print(f"\nSaved in {out}: materials_validation.json, materials_summary.png, "
          "materials_parity.png, predictions.npz")


def parity(out, names, groups, results, preds):
    """Force parity vs DFT: one row per group, one column per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(groups)
    fig, axes = plt.subplots(len(keys), len(names), figsize=(3.7 * len(names), 3.7 * len(keys)),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for row, key in enumerate(keys):
        ref_any = next((preds[(n, key)]["dft_f"] for n in names if (n, key) in preds), None)
        lim = 1.1 * np.percentile(np.abs(ref_any), 99.5) if ref_any is not None else 1.0
        for col, n in enumerate(names):
            ax = axes[row][col]
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
            if (n, key) not in preds:
                ax.text(0, 0, "cannot run\n(elements not supported)", ha="center", va="center",
                        fontsize=10, color="tab:red")
                ax.set_title(f"{n} | group {key}", fontsize=9)
            else:
                ref, pred = preds[(n, key)]["dft_f"].ravel(), preds[(n, key)]["fp"].ravel()
                sel = rng.choice(len(ref), size=min(len(ref), 20000), replace=False)
                outside = int((np.abs(pred) > lim).sum())
                ax.scatter(ref[sel], np.clip(pred[sel], -lim, lim), s=2, alpha=0.25, rasterized=True)
                g = results[n]["groups"][key]
                ax.set_title(f"{n} | group {key}\nMAE {g['force_mae_meV_A']:.0f} (median "
                             f"{g['force_mae_median_structure_meV_A']:.0f}) meV/Å, R² "
                             f"{g['force_r2']:.3f}" + (f"\n{outside} pts beyond axes (clipped)"
                                                       if outside else ""), fontsize=8.5)
            if row == len(keys) - 1:
                ax.set_xlabel("DFT force component (eV/Å)", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"{GROUP_TEXT[key].split(' (')[0]}\nmodel force (eV/Å)", fontsize=8.5)
    fig.suptitle("Inorganic materials (MPtrj DFT): force parity per group and model", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "materials_parity.png", dpi=130)
    plt.close(fig)


def plot(out, names, groups, results, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    keys = list(groups)
    width = 0.8 / len(names)
    for ax, (metric, label) in zip(axes, [("force_mae_meV_A", "force MAE vs DFT (meV/Å)"),
                                          ("energy_mae_aligned_meV_atom",
                                           "energy MAE vs DFT, offset-aligned (meV/atom)"),
                                          ("stress_mae_GPa", "stress MAE vs DFT (GPa)")]):
        for i, n in enumerate(names):
            vals = [results[n]["groups"][k].get(metric, np.nan) if results[n]["groups"][k]["n"]
                    else np.nan for k in keys]
            xs = np.arange(len(keys)) + (i - (len(names) - 1) / 2) * width
            bars = ax.bar(xs, vals, width, color=colors[i % len(colors)], label=n)
            for x, v, k in zip(xs, vals, keys):
                ax.text(x, (0 if np.isnan(v) else v), "n/a" if np.isnan(v) else f"{v:.0f}" if v >= 10
                        else f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([GROUP_TEXT[k].split(" (")[0].replace(": ", ":\n") for k in keys], fontsize=8.5)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Inorganic materials (MPtrj DFT): lower is better; n/a = model cannot run "
                 "(elements outside its training)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "materials_summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
