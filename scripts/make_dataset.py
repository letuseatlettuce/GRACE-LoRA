"""Build the distillation dataset: sample SPICE structures and label them with the
GRACE-OFF teacher.

Outputs (in DATA_DIR):
    train.pkl.gz, val.pkl.gz : teacher labels (energy, forces) for LoRA training
    test.pkl.gz              : held-out structures with both DFT and teacher labels

Columns: ase_atoms, energy, energy_corrected, forces (teacher), dft_energy, dft_forces.
The format is a gracemaker-compatible pandas DataFrame.

Usage:
    python scripts/make_dataset.py --n-train 2000 --n-val 200 --n-test 1000
"""

import argparse
import hashlib
import time
from pathlib import Path

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import pandas as pd
from ase.io import read
from tensorpotential.calculator import TPCalculator
from tqdm import tqdm

import config


def structure_hash(atoms):
    key = np.round(atoms.positions, 5).tobytes() + atoms.numbers.tobytes()
    return hashlib.md5(key).hexdigest()


def label(structures, calc, desc):
    """Return a DataFrame with teacher labels and the original DFT labels (NaN/None if the
    input structures carry no energies/forces)."""
    rows = []
    for atoms in tqdm(structures, desc=desc):
        if atoms.calc is not None:
            dft_energy = atoms.get_potential_energy()
            dft_forces = atoms.get_forces()
        else:
            dft_energy, dft_forces = np.nan, None
        clean = atoms.copy()  # copy() drops the attached calculator
        clean.calc = calc
        energy = clean.get_potential_energy()
        forces = clean.get_forces()
        clean.calc = None
        rows.append({
            "ase_atoms": clean,
            "energy": energy,
            "energy_corrected": energy,  # reference_energy = 0: train on energies as-is
            "forces": forces,
            "dft_energy": dft_energy,
            "dft_forces": dft_forces,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default=config.TRAIN_POOL_XYZ, help="training pool .xyz")
    parser.add_argument("--test", default=config.TEST_XYZ,
                        help="held-out test .xyz ('' = sample the test set from the pool)")
    parser.add_argument("--teacher", default=config.TEACHER_MODEL_DIR, help="GRACE-OFF SavedModel")
    parser.add_argument("--out", default=config.ORGANIC_DIR, help="output folder")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--max-atoms", type=int, default=None,
                        help="skip structures larger than this (speeds up training)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config.setup_device()

    pool_path = config.require(args.pool, "TRAIN_POOL_XYZ")
    test_path = config.require(args.test, "TEST_XYZ") if args.test else None
    teacher_path = config.require(args.teacher, "TEACHER_MODEL_DIR")
    out = config.require(args.out, "DATA_DIR")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print(f"Reading {pool_path}" + (f" and {test_path}" if test_path else ""))
    pool = read(pool_path, index=":")
    if test_path:
        test = read(test_path, index=":")
    else:
        # no separate test file: hold out a random part of the pool
        idx = rng.permutation(len(pool))
        test = [pool[i] for i in idx[:args.n_test]]
        pool = [pool[i] for i in idx[args.n_test:]]
        print(f"No test file: held out {len(test)} pool structures as the test set")

    # The two Zenodo sets overlap slightly; keep the test set strictly unseen.
    test_hashes = {structure_hash(a) for a in test}
    n_before = len(pool)
    pool = [a for a in pool if structure_hash(a) not in test_hashes]
    print(f"Removed {n_before - len(pool)} pool structures that are also in the test set")

    if args.max_atoms:
        pool = [a for a in pool if len(a) <= args.max_atoms]
        test = [a for a in test if len(a) <= args.max_atoms]

    n_needed = args.n_train + args.n_val
    if n_needed > len(pool):
        raise SystemExit(f"Requested {n_needed} train+val structures, pool has {len(pool)}")
    idx = rng.permutation(len(pool))
    train = [pool[i] for i in idx[:args.n_train]]
    val = [pool[i] for i in idx[args.n_train:n_needed]]
    test = [test[i] for i in rng.permutation(len(test))[:args.n_test]]

    teacher = TPCalculator(model=str(teacher_path))
    t0 = time.perf_counter()
    for name, structures in [("train", train), ("val", val), ("test", test)]:
        df = label(structures, teacher, f"labelling {name}")
        config.save_frame(df, out / f"{name}.pkl.gz")
        msg = f"  {name}: {len(df)} structures, {sum(map(len, df.ase_atoms))} atoms"
        if df.dft_forces.notna().all():
            f_mae = np.mean([np.abs(f - g).mean() for f, g in zip(df.forces, df.dft_forces)])
            msg += f", teacher force MAE vs DFT = {f_mae * 1000:.1f} meV/A"
        print(msg)
    print(f"Done in {time.perf_counter() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
