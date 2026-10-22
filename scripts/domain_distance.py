"""How far is a candidate domain from what base GRACE learned? (before training an adapter)

Uses the base model's built-in extrapolation grade gamma (per atom; GMM in the model's
latent space): gamma <= 1 interpolation, 1-2 mild extrapolation, > 2 strong extrapolation.
A LoRA adapter can only re-weight features the frozen base computes, so domains where many
atoms have gamma > 2 are poor adapter targets: their local environments are not represented.

Reports, per dataset and per element: median / 95th percentile gamma and the fraction of
atoms with gamma > 1 and > 2.

Usage:
    python scripts/domain_distance.py name=data/GRACE_test_set.xyz other=my_domain.xyz --n 200
    python scripts/domain_distance.py --builtin crystals water        # reference sets
"""

import argparse
import json
from pathlib import Path

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import read
from tensorpotential.calculator import TPCalculator
from tqdm import tqdm

import config


def builtin(name: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    if name == "crystals":  # the base model's home domain (inorganic, periodic)
        specs = [("Si", "diamond", 5.43), ("Cu", "fcc", 3.61), ("Fe", "bcc", 2.87),
                 ("Al", "fcc", 4.05), ("NaCl", "rocksalt", 5.64), ("MgO", "rocksalt", 4.21),
                 ("GaAs", "zincblende", 5.65), ("TiO", "rocksalt", 4.18)]
        out = []
        for i, (f, s, a) in enumerate(specs):
            c = bulk(f, s, a=a).repeat((2, 2, 2))
            c.rattle(stdev=0.05, seed=seed + i)
            out.append(c)
        return out
    if name == "water":  # liquid-like water boxes (32 molecules, ~1 g/cm3), crude packing
        boxes = []
        for k in range(6):
            cell, pos, sym = 9.85, [], []
            while len(sym) < 96:
                centre = rng.uniform(0, cell, 3)
                if all(np.linalg.norm((centre - p + cell / 2) % cell - cell / 2) > 2.6
                       for p in pos[::3]):
                    w = molecule("H2O")
                    w.rotate(rng.uniform(0, 360), rng.normal(size=3))
                    pos += list(w.positions - w.positions[0] + centre)
                    sym += list(w.get_chemical_symbols())
            boxes.append(Atoms(sym, positions=pos, cell=[cell] * 3, pbc=True))
        return boxes
    raise SystemExit(f"unknown builtin set '{name}' (choose: crystals, water)")


def load(path: Path, n: int, seed: int):
    if path.name.endswith((".pkl.gz", ".pkl")):
        atoms = list(config.load_frame(path).ase_atoms)
    else:
        atoms = read(path, index=":")
    if n and n < len(atoms):
        atoms = [atoms[i] for i in np.random.default_rng(seed).permutation(len(atoms))[:n]]
    return atoms


def summarise(gammas, symbols):
    g = np.asarray(gammas)
    row = {"atoms": int(g.size), "median": float(np.median(g)), "p95": float(np.percentile(g, 95)),
           "frac_gt1": float((g > 1).mean()), "frac_gt2": float((g > 2).mean())}
    row["per_element"] = {
        el: {"atoms": int((symbols == el).sum()), "median": float(np.median(g[symbols == el])),
             "frac_gt2": float((g[symbols == el] > 2).mean())}
        for el in sorted(set(symbols))}
    return row


def verdict(frac_gt2):
    if frac_gt2 < 0.05:
        return "close: good adapter target"
    if frac_gt2 < 0.20:
        return "moderate: adapter should work, expect a larger remaining gap"
    return "far: base features under-represent this domain; adapter likely insufficient"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", help="name=path (.xyz or .pkl.gz)")
    p.add_argument("--builtin", nargs="*", default=[], help="crystals and/or water reference sets")
    p.add_argument("--base", default=config.STUDENT_MODEL_DIR, help="base GRACE SavedModel")
    p.add_argument("--n", type=int, default=200, help="structures per dataset (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON output")
    args = p.parse_args()
    config.setup_device()

    sets = {name: builtin(name, args.seed) for name in args.builtin}
    for spec in args.datasets:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        sets[name] = load(Path(path).expanduser(), args.n, args.seed)

    calc = TPCalculator(model=str(config.require(args.base, "STUDENT_MODEL_DIR")),
                        enable_uq_if_available=True)
    calc.enable_uq("gamma_only")
    results = {}
    for name, structures in sets.items():
        gammas, symbols = [], []
        for atoms in tqdm(structures, desc=name, leave=False):
            a = atoms.copy()
            a.calc = calc
            a.get_potential_energy()
            gammas.append(np.asarray(calc.results["gamma"]).ravel()[:len(a)])
            symbols.append(a.get_chemical_symbols())
        results[name] = summarise(np.concatenate(gammas), np.concatenate(symbols))

    print(f"\n== Distance from base GRACE (extrapolation grade gamma; <=1 interpolation)")
    print(f"  {'dataset':18s} {'atoms':>7s} {'median':>7s} {'p95':>6s} {'>1':>6s} {'>2':>6s}  verdict")
    for name, r in results.items():
        print(f"  {name:18s} {r['atoms']:7d} {r['median']:7.2f} {r['p95']:6.2f} "
              f"{100 * r['frac_gt1']:5.0f}% {100 * r['frac_gt2']:5.0f}%  {verdict(r['frac_gt2'])}")
    print("\n  per element: median gamma (share of atoms with gamma > 2)")
    for name, r in results.items():
        cells = "  ".join(f"{el} {d['median']:.2f} ({100 * d['frac_gt2']:.0f}%)"
                          for el, d in r["per_element"].items())
        print(f"  {name:18s} {cells}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
