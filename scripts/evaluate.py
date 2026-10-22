"""Evaluate base student, LoRA students and the teacher on the held-out test set.

Metrics (test.pkl.gz from make_dataset.py):
  - force MAE vs DFT and vs teacher (meV/A)
  - energy MAE vs DFT (meV/atom), after fitting per-element offsets on a calibration
    split, so models with different energy references are compared fairly
Materials drift (forgetting check): how far each fine-tuned model moves away from the base
student on rattled inorganic crystals (no DFT needed: base model = reference). Energy drift
uses only crystals without organic elements: those elements get new per-element energy
offsets by design, which shifts energies without changing any physics.

Usage:
    python scripts/evaluate.py --runs lora_r8 full_baseline
"""

import argparse
import json
import time
from pathlib import Path

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import pandas as pd
from ase.build import bulk
from tensorpotential.calculator import TPCalculator
from tqdm import tqdm

import config

CALIBRATION_FRACTION = 0.2


def predict(calc, structures, desc):
    energies, forces = [], []
    for atoms in tqdm(structures, desc=desc, leave=False):
        a = atoms.copy()
        a.calc = calc
        energies.append(a.get_potential_energy())
        forces.append(a.get_forces())
    return np.array(energies), forces


def composition(structures, elements):
    return np.array([[a.get_chemical_symbols().count(e) for e in elements] for a in structures],
                    dtype=float)


def aligned_energy_mae(e_pred, e_ref, comp, natoms, n_cal):
    """Fit per-element offsets on the first n_cal structures, report MAE/atom on the rest."""
    offsets, *_ = np.linalg.lstsq(comp[:n_cal], (e_ref - e_pred)[:n_cal], rcond=None)
    err = (e_pred + comp @ offsets - e_ref)[n_cal:] / natoms[n_cal:]
    return 1000 * np.abs(err).mean()


def force_mae(f_pred, f_ref):
    return 1000 * np.mean(np.concatenate([np.abs(p - r).ravel() for p, r in zip(f_pred, f_ref)]))


def materials_set(seed=0):
    """Small inorganic sanity set: rattled supercells of common crystals."""
    crystals = [
        bulk("Si", "diamond", a=5.43), bulk("Cu", "fcc", a=3.61), bulk("Fe", "bcc", a=2.87),
        bulk("Al", "fcc", a=4.05), bulk("NaCl", "rocksalt", a=5.64),
        bulk("MgO", "rocksalt", a=4.21), bulk("GaAs", "zincblende", a=5.65),
        bulk("TiO", "rocksalt", a=4.18),
    ]
    out = []
    for i, c in enumerate(crystals):
        s = c.repeat((2, 2, 2))
        s.rattle(stdev=0.05, seed=seed + i)
        out.append(s)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--runs-dir", default=config.RUNS_DIR)
    p.add_argument("--runs", nargs="*", default=[], help="run names inside runs-dir")
    p.add_argument("--student-model", default=config.STUDENT_MODEL_DIR,
                   help="base student SavedModel")
    p.add_argument("--teacher", default=config.TEACHER_MODEL_DIR)
    p.add_argument("--n-test", type=int, default=None, help="limit test structures")
    p.add_argument("--out", default=None, help="results JSON (default: runs-dir/eval.json)")
    args = p.parse_args()
    config.setup_device()

    data_dir = config.require(args.data_dir, "DATA_DIR")
    runs_dir = config.require(args.runs_dir, "RUNS_DIR")
    models = {
        "student_base": config.require(args.student_model, "STUDENT_MODEL_DIR"),
        **{f"student_{r}": runs_dir / r / "saved_model" for r in args.runs},
        "teacher": config.require(args.teacher, "TEACHER_MODEL_DIR"),
    }

    test = config.load_frame(data_dir / "test.pkl.gz")
    if args.n_test:
        test = test.iloc[:args.n_test]
    structures = list(test.ase_atoms)
    elements = sorted({e for a in structures for e in a.get_chemical_symbols()})
    comp = composition(structures, elements)
    natoms = np.array([len(a) for a in structures])
    n_cal = int(CALIBRATION_FRACTION * len(structures))
    has_dft = bool(test.dft_forces.notna().all())  # custom structures may carry no DFT labels
    dft_e, dft_f = test.dft_energy.values, list(test.dft_forces)
    teacher_f = list(test.forces)
    materials = materials_set()

    results, base_materials = {}, None
    for name, path in models.items():
        print(f"\n== {name}: {path}")
        calc = TPCalculator(model=str(path))
        t0 = time.perf_counter()
        e, f = predict(calc, structures, name)
        dt_ms = 1000 * (time.perf_counter() - t0) / len(structures)
        res = {}
        if has_dft:
            res["force_mae_vs_dft_meV_A"] = force_mae(f, dft_f)
        res["force_mae_vs_teacher_meV_A"] = force_mae(f, teacher_f)
        if has_dft:
            res["energy_mae_vs_dft_meV_atom"] = aligned_energy_mae(e, dft_e, comp, natoms, n_cal)
        res["ms_per_structure"] = dt_ms
        if name != "teacher":  # the teacher has no inorganic elements
            me, mf = predict(calc, materials, f"{name} materials")
            if base_materials is None:
                base_materials = (me, mf)
            else:
                nat_m = np.array([len(a) for a in materials])
                inorganic = np.array([not set(a.get_chemical_symbols()) & set(elements)
                                      for a in materials])
                res["materials_drift_force_meV_A"] = force_mae(mf, base_materials[1])
                res["materials_drift_energy_meV_atom"] = float(1000 * np.mean(
                    (np.abs(me - base_materials[0]) / nat_m)[inorganic]))
        results[name] = res
        print("   " + "  ".join(f"{k}={v:.2f}" for k, v in res.items()))

    out = Path(args.out) if args.out else runs_dir / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    table = pd.DataFrame(results).T
    print("\n" + table.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nTest structures: {len(structures)} ({n_cal} used only to align energy offsets)")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
