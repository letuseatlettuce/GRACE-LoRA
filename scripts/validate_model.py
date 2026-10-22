"""Validate LoRA-trained GRACE models against the GRACE-OFF teacher, and show they beat the
untrained base GRACE model.

On the same held-out structures it predicts energies and forces with
    teacher   (GRACE-OFF)           -> the reference
    base      (untrained GRACE)     -> what we started from
    trained   (one or more models)  -> what training produced
and reports, for base and every trained model against the teacher:
    force MAE / RMSE, force-direction cosine, per-element force MAE,
    energy MAE per atom after per-element offset alignment (fit on a 20% calibration split),
    per-structure win rate vs base and a bootstrap 95% CI of the improvement.
If the structures carry DFT labels (e.g. GRACE_test_set.xyz), errors vs DFT are reported too.

Use structures the trained models never saw: GRACE_test_set.xyz or make_dataset's test.pkl.gz.

Usage:
    python scripts/validate_model.py --structures data/GRACE_test_set.xyz --n 500
    python scripts/validate_model.py --trained medium=runs/pilot/lora_r8/saved_model \\
        C10=checkpoint_10/saved_model C11=checkpoint_11/saved_model --structures data/pilot/test.pkl.gz
    (a raw training checkpoint must first be exported: scripts/export_checkpoint.py)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import pandas as pd
import yaml
from ase.io import read
from tensorpotential.calculator import TPCalculator
from tqdm import tqdm

import config

CALIBRATION_FRACTION = 0.2
N_BOOTSTRAP = 2000


def find_best_run(runs_dir: Path) -> Path:
    """Trained SavedModel of the run with the lowest validation force MAE under runs_dir."""
    candidates = []
    for summary in runs_dir.expanduser().rglob("summary.json"):
        model = summary.parent / "saved_model"
        if model.exists():
            best = json.loads(summary.read_text()).get("val_best_f_mae_meV_A", np.inf)
            candidates.append((best, model))
    if not candidates:
        raise SystemExit(f"No finished training run (summary.json + saved_model) under {runs_dir}; "
                         "pass --trained <run>/saved_model")
    best, model = min(candidates)
    print(f"Best trained run: {model} (val force MAE {best:.2f} meV/A)")
    return model


def parse_trained(specs):
    """['name=path', 'path', ...] -> {name: Path}; a raw checkpoint prefix is rejected."""
    models = {}
    for spec in specs:
        name, _, path = spec.rpartition("=") if "=" in spec else ("", "", spec)
        path = Path(path).expanduser()
        if not (path / "saved_model.pb").exists():
            hint = (f"\n  looks like a raw checkpoint: export it first with\n"
                    f"  python scripts/export_checkpoint.py {path}" if Path(str(path) + ".index").exists()
                    or (path / "best_checkpoint.index").exists() else "")
            raise SystemExit(f"No SavedModel at {path}{hint}")
        name = name or (path.parent.name if path.name == "saved_model" else path.name)
        models[name] = path
    return models


def model_elements(model_dir: Path):
    meta = model_dir / "metadata.yaml"
    if meta.exists():
        return set(yaml.safe_load(meta.read_text()).get("chemical_symbols", []))
    return None


def load_structures(path: Path, n: int, seed: int):
    """Return (atoms list without calculators, DFT energies or None, DFT forces or None)."""
    if path.name.endswith((".pkl.gz", ".pkl")):
        df = config.load_frame(path)
        atoms = list(df.ase_atoms)
        dft_e = df.dft_energy.values if "dft_energy" in df else None
        dft_f = list(df.dft_forces) if "dft_forces" in df else None
    else:
        atoms = read(path, index=":")
        labelled = all(a.calc is not None and "forces" in a.calc.results for a in atoms)
        dft_e = np.array([a.get_potential_energy() for a in atoms]) if labelled else None
        dft_f = [a.get_forces() for a in atoms] if labelled else None
    if dft_f is not None and any(f is None for f in dft_f):
        dft_e = dft_f = None
    idx = np.random.default_rng(seed).permutation(len(atoms))[:n] if n and n < len(atoms) \
        else np.arange(len(atoms))
    atoms = [atoms[i].copy() for i in idx]
    for a in atoms:
        a.calc = None
    pick = (lambda xs: [xs[i] for i in idx]) if dft_f is not None else (lambda xs: None)
    return atoms, (np.asarray(dft_e)[idx] if dft_e is not None else None), pick(dft_f)


def predict(model_dir: Path, structures, label):
    calc = TPCalculator(model=str(model_dir))
    energies, forces = [], []
    t0 = time.perf_counter()
    for atoms in tqdm(structures, desc=label, leave=False):
        a = atoms.copy()
        a.calc = calc
        energies.append(a.get_potential_energy())
        forces.append(a.get_forces())
    return np.array(energies), forces, 1000 * (time.perf_counter() - t0) / len(structures)


def composition(structures, elements):
    return np.array([[a.get_chemical_symbols().count(e) for e in elements] for a in structures],
                    dtype=float)


def aligned_energy_errors(e_pred, e_ref, comp, natoms, n_cal):
    """Per-atom |error| on the evaluation split after fitting per-element offsets on n_cal."""
    offsets, *_ = np.linalg.lstsq(comp[:n_cal], (e_ref - e_pred)[:n_cal], rcond=None)
    return np.abs((e_pred + comp @ offsets - e_ref) / natoms)[n_cal:]


def force_metrics(f_pred, f_ref, symbols):
    diff = np.concatenate([p - r for p, r in zip(f_pred, f_ref)])
    pred, ref = np.concatenate(f_pred), np.concatenate(f_ref)
    norm_p, norm_r = np.linalg.norm(pred, axis=1), np.linalg.norm(ref, axis=1)
    mask = (norm_r > 0.1) & (norm_p > 1e-8)  # direction is meaningless for ~zero forces
    cosine = np.sum(pred[mask] * ref[mask], axis=1) / (norm_p[mask] * norm_r[mask])
    all_symbols = np.concatenate(symbols)
    per_element = {el: 1000 * float(np.abs(diff[all_symbols == el]).mean())
                   for el in sorted(set(all_symbols))}
    return {
        "force_mae_meV_A": 1000 * float(np.abs(diff).mean()),
        "force_rmse_meV_A": 1000 * float(np.sqrt((diff ** 2).mean())),
        "force_r2": r2_score(pred.ravel(), ref.ravel()),
        "force_cosine_mean": float(cosine.mean()),
        "per_element_force_mae_meV_A": per_element,
    }


def r2_score(pred, ref):
    """Coefficient of determination of pred against ref (force components pooled)."""
    return float(1 - np.sum((pred - ref) ** 2) / np.sum((ref - ref.mean()) ** 2))


def per_structure_force_mae(f_pred, f_ref):
    return np.array([1000 * np.abs(p - r).mean() for p, r in zip(f_pred, f_ref)])


def bootstrap_ci(values, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values)).mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def parity_plot(path, f_reference, panels, reference_label="GRACE-OFF"):
    """panels: list of (title, forces); one parity panel per model vs the reference forces."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.concatenate(f_reference).ravel()
    sel = np.random.default_rng(0).choice(len(ref), size=min(len(ref), 30000), replace=False)
    lim = np.percentile(np.abs(ref), 99.5) * 1.1
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.8),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (title, pred) in zip(axes[0], panels):
        p = np.concatenate(pred).ravel()
        ax.scatter(ref[sel], p[sel], s=2, alpha=0.25, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{reference_label} force component (eV/Å)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("model force component (eV/Å)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default=config.TEACHER_MODEL_DIR, help="GRACE-OFF saved_model")
    p.add_argument("--base", default=config.STUDENT_MODEL_DIR, help="untrained GRACE SavedModel")
    p.add_argument("--trained", nargs="*", default=None,
                   help="trained SavedModels as name=path or path "
                        "(default: best run found under --runs-dir)")
    p.add_argument("--runs-dir", default=config.RUNS_DIR)
    p.add_argument("--reference-only", action="store_true",
                   help="evaluate only base GRACE and the teacher (reference rows; needs --out)")
    p.add_argument("--structures", default=config.TEST_XYZ,
                   help="held-out structures: .xyz (e.g. GRACE_test_set.xyz) or test.pkl.gz")
    p.add_argument("--n", type=int, default=500, help="number of structures (0 = all)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None,
                   help="output folder (default: <run>/validation for one model, "
                        "else <project>/validation)")
    args = p.parse_args()
    config.setup_device()

    teacher = config.require(args.teacher, "TEACHER_MODEL_DIR")
    base = config.require(args.base, "STUDENT_MODEL_DIR")
    if "casted_model" in str(teacher):
        raise SystemExit("Use the teacher's fp64 saved_model folder, not casted_model")
    if args.reference_only:
        trained = {}
    elif args.trained:
        trained = parse_trained(args.trained)
    else:
        trained = {"trained": find_best_run(config.require(args.runs_dir, "RUNS_DIR"))}
    structures_path = config.require(args.structures, "TEST_XYZ")
    if args.out:
        out = Path(args.out).expanduser()
    elif len(trained) == 1:
        out = next(iter(trained.values())).parent / "validation"
    else:
        out = Path(config.VALIDATION_DIR).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    structures, dft_e, dft_f = load_structures(structures_path, args.n, args.seed)
    allowed = [s for s in (model_elements(m) for m in [teacher, base, *trained.values()])
               if s is not None]
    if allowed:
        common = set.intersection(*allowed)
        keep = [i for i, a in enumerate(structures) if set(a.get_chemical_symbols()) <= common]
        if len(keep) < len(structures):
            print(f"Skipping {len(structures) - len(keep)} structures with elements outside the models")
            structures = [structures[i] for i in keep]
            dft_e = dft_e[keep] if dft_e is not None else None
            dft_f = [dft_f[i] for i in keep] if dft_f is not None else None
    print(f"Structures: {len(structures)} from {structures_path}"
          f"{' (with DFT labels)' if dft_f is not None else ''}")

    models = {"teacher": teacher, "base": base, **trained}
    preds = {}
    for name, path in models.items():
        print(f"Predicting with {name}: {path}")
        preds[name] = predict(path, structures, name)

    symbols = [np.array(a.get_chemical_symbols()) for a in structures]
    elements = sorted(set(np.concatenate(symbols)))
    comp = composition(structures, elements)
    natoms = np.array([len(a) for a in structures])
    n_cal = int(CALIBRATION_FRACTION * len(structures))
    e_teacher, f_teacher, _ = preds["teacher"]
    students = ["base", *trained]

    metrics = {}
    for name in students:
        e, f, ms = preds[name]
        m = {"vs_teacher": force_metrics(f, f_teacher, symbols), "ms_per_structure": ms}
        m["vs_teacher"]["energy_mae_meV_atom"] = 1000 * float(
            aligned_energy_errors(e, e_teacher, comp, natoms, n_cal).mean())
        if dft_f is not None:
            m["vs_dft"] = force_metrics(f, dft_f, symbols)
            m["vs_dft"]["energy_mae_meV_atom"] = 1000 * float(
                aligned_energy_errors(e, dft_e, comp, natoms, n_cal).mean())
        metrics[name] = m
    if dft_f is not None:
        teacher_dft = force_metrics(f_teacher, dft_f, symbols)
        teacher_dft["energy_mae_meV_atom"] = 1000 * float(
            aligned_energy_errors(e_teacher, dft_e, comp, natoms, n_cal).mean())
        metrics["teacher"] = {"vs_dft": teacher_dft, "ms_per_structure": preds["teacher"][2]}

    # paired, per-structure comparison of every trained model against base
    per_struct = {name: per_structure_force_mae(preds[name][1], f_teacher) for name in students}
    paired = {}
    for name in trained:
        improvement = per_struct["base"] - per_struct[name]
        lo, hi = bootstrap_ci(improvement)
        scale = per_struct["base"].mean()
        paired[name] = {
            "mean_improvement_meV_A": float(improvement.mean()), "ci95_meV_A": [lo, hi],
            "relative_improvement_pct": float(100 * improvement.mean() / scale),
            "ci95_pct": [100 * lo / scale, 100 * hi / scale],
            "better_than_base_on": int((improvement > 0).sum()), "n_structures": len(structures),
            "passed": lo > 0,  # whole 95% CI of the improvement above zero
        }
        if dft_f is not None:
            b, t = metrics["base"]["vs_dft"]["force_mae_meV_A"], metrics["teacher"]["vs_dft"]["force_mae_meV_A"]
            paired[name]["gap_to_teacher_closed_vs_dft_pct"] = \
                100 * (b - metrics[name]["vs_dft"]["force_mae_meV_A"]) / (b - t) if b != t else float("nan")

    # ---- report ---------------------------------------------------------------------------
    w = max(12, *(len(n) + 2 for n in students))
    header = "".join(f"{n:>{w}s}" for n in students)
    print(f"\n== Agreement with GRACE-OFF ({len(structures)} held-out structures)")
    print(f"  {'metric':40s}{header}")
    for key, label in [("force_mae_meV_A", "force MAE (meV/A)"),
                       ("force_rmse_meV_A", "force RMSE (meV/A)"),
                       ("force_r2", "force R^2 (parity, 1 = perfect)"),
                       ("force_cosine_mean", "force direction cosine (1 = same)"),
                       ("energy_mae_meV_atom", "energy MAE, offset-aligned (meV/atom)")]:
        print(f"  {label:40s}" + "".join(f"{metrics[n]['vs_teacher'][key]:>{w}.3f}" for n in students))
    print("\n  per-element force MAE vs GRACE-OFF (meV/A):")
    for el in elements:
        print(f"    {el:36s}" + "".join(
            f"{metrics[n]['vs_teacher']['per_element_force_mae_meV_A'][el]:>{w}.1f}" for n in students))
    if dft_f is not None:
        rows = [*students, "teacher"]
        print(f"\n== Agreement with DFT (same structures)")
        print(f"  {'model':16s} {'force MAE (meV/A)':>18s} {'force R^2':>10s} {'energy MAE (meV/atom)':>22s} {'gap closed':>11s}")
        for name in rows:
            d = metrics[name]["vs_dft"]
            closed = paired.get(name, {}).get("gap_to_teacher_closed_vs_dft_pct")
            print(f"  {name:16s} {d['force_mae_meV_A']:18.2f} {d['force_r2']:10.4f} {d['energy_mae_meV_atom']:22.2f} "
                  f"{(f'{closed:.0f}%' if closed is not None else '-'):>11s}")
    print(f"\n== Trained vs base (paired, per structure, force MAE vs GRACE-OFF)")
    for name, d in paired.items():
        print(f"  {name:16s} improved by {d['mean_improvement_meV_A']:6.1f} meV/A "
              f"({d['relative_improvement_pct']:5.1f}%), 95% CI [{d['ci95_meV_A'][0]:.1f}, "
              f"{d['ci95_meV_A'][1]:.1f}]; better on {d['better_than_base_on']}/{d['n_structures']} "
              f"structures -> {'PASS' if d['passed'] else 'FAIL'}")
    ranking = sorted(trained, key=lambda n: metrics[n]["vs_teacher"]["force_mae_meV_A"])
    all_pass = bool(paired) and all(d["passed"] for d in paired.values())
    if trained:
        print(f"\nRanking (closest to GRACE-OFF first): "
              + " < ".join(f"{n} ({metrics[n]['vs_teacher']['force_mae_meV_A']:.1f})" for n in ranking)
              + f" < base ({metrics['base']['vs_teacher']['force_mae_meV_A']:.1f}) meV/A")
        print(f"VERDICT: {'PASS' if all_pass else 'FAIL'}: "
              f"{'every' if all_pass else 'not every'} trained model is significantly closer to "
              "GRACE-OFF than base GRACE")
    else:
        print("\nReference only (base GRACE and the teacher; no trained model).")

    result = {"teacher": str(teacher), "base": str(base),
              "trained": {k: str(v) for k, v in trained.items()},
              "structures": str(structures_path), "n_structures": len(structures),
              "metrics": metrics, "paired_vs_base": paired, "ranking": ranking, "passed": all_pass,
              "time": time.strftime("%Y-%m-%d %H:%M")}
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    with open(out / "per_structure.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["index", "formula", "n_atoms",
                     *[f"{n}_force_mae_vs_teacher" for n in students]])
        for i, a in enumerate(structures):
            wr.writerow([i, a.get_chemical_formula(), len(a),
                         *[f"{per_struct[n][i]:.3f}" for n in students]])
    panels = [(f"{n}: MAE {metrics[n]['vs_teacher']['force_mae_meV_A']:.0f} meV/Å, "
                f"R² {metrics[n]['vs_teacher']['force_r2']:.3f}", preds[n][1]) for n in students]
    parity_plot(out / "parity_forces.png", f_teacher, panels)
    saved = ["validation.json", "per_structure.csv", "parity_forces.png", "predictions.npz"]
    if dft_f is not None:
        dft_panels = [(f"{n}: MAE {metrics[n]['vs_dft']['force_mae_meV_A']:.0f} meV/Å, "
                       f"R² {metrics[n]['vs_dft']['force_r2']:.3f}", preds[n][1])
                      for n in [*students, "teacher"]]
        dft_panels[-1] = (dft_panels[-1][0].replace("teacher", "GRACE-OFF"), dft_panels[-1][1])
        parity_plot(out / "parity_forces_vs_dft.png", dft_f, dft_panels, reference_label="DFT")
        saved.append("parity_forces_vs_dft.png")
    # raw predictions (forces concatenated over atoms) so plots can be redone without the models
    np.savez_compressed(
        out / "predictions.npz",
        n_atoms=natoms, symbols=np.concatenate(symbols),
        **{f"{n}_energy": preds[n][0] for n in models},
        **{f"{n}_forces": np.concatenate(preds[n][1]) for n in models},
        **({"dft_energy": dft_e, "dft_forces": np.concatenate(dft_f)} if dft_f is not None else {}))
    print(f"\nSaved in {out}: {', '.join(saved)}")


if __name__ == "__main__":
    main()
