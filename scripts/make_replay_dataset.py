"""Mixed ("replay") training set: teacher-labelled organic molecules + DFT-labelled materials
that contain the adapted elements, so a LoRA adapter learns organics without drifting away
from base GRACE on materials.

  organics : train/val.pkl.gz from make_dataset.py (teacher labels). Their energies are moved
             onto base GRACE's energy reference with per-element offsets fitted against base
             predictions (exact for energies; forces are unchanged).
  materials: MPtrj parquet shard(s) (DFT, PBE/PBE+U = base GRACE's reference). Only structures
             with at least one adapted element; structures with H or F are over-sampled (most
             damaged elements); materials listed in --exclude shards are never used; one frame
             per material. --n-only-adapted sets a quota of crystals made ONLY of adapted
             elements (molecular crystals, C/P/S solids), with H/F/O over-sampled.

Modes:
  default          writes train.pkl.gz, val.pkl.gz, energy_reference.json into --out
                   (and legacy monitor_*.pkl.gz files unless --no-monitors)
  --monitors-only  writes the fixed held-out monitor sets organic.pkl.gz, materials.pkl.gz,
                   only_adapted.pkl.gz into --out, plus energy_reference.json: per-element
                   offsets (teacher - base) fitted on the organic train pool against base
                   GRACE, which are applied to the organic monitor energies (forces unchanged)
                   and can be reused by the default mode via --energy-reference.

Usage:
    python scripts/make_replay_dataset.py --organic-dir <workdir>/data/organic \\
        --materials test.parquet train0.parquet --exclude val.parquet \\
        --n-organic 1000 --n-materials 600 --n-materials-val 100 --n-only-adapted 170 \\
        --out <workdir>/experiments/exp_001/data --no-monitors
    python scripts/make_replay_dataset.py --monitors-only --organic-dir <workdir>/data/organic \\
        --exclude val.parquet --out <workdir>/data/monitors
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

LABEL_COLUMNS = ["ase_atoms", "energy", "energy_corrected", "forces"]


# --------------------------------------------------------------------------------------
# pure helpers (no TensorFlow; unit-tested in tests/test_core.py)
# --------------------------------------------------------------------------------------
def element_sets(numbers: pd.Series) -> pd.Series:
    """Series of atomic-number lists -> Series of element-symbol sets."""
    from ase.data import chemical_symbols as SYM

    return numbers.map(lambda z: {SYM[int(i)] for i in z})


def eligible_materials(mats: pd.DataFrame, adapted: set, max_atoms: int, excluded_ids: set,
                       seed: int):
    """Structures with >= 1 adapted element, <= max_atoms, mp_id not excluded; one random frame
    per material (more distinct materials; no material in both splits).
    Returns (materials, element_sets), both with a fresh 0..n-1 index."""
    els = element_sets(mats.numbers)
    keep = ((mats.num_atoms <= max_atoms) & els.map(lambda s: bool(s & adapted))
            & ~mats.mp_id.isin(excluded_ids))
    mats, els = mats[keep].reset_index(drop=True), els[keep].reset_index(drop=True)
    first = mats.sample(frac=1, random_state=seed).drop_duplicates("mp_id").index
    return mats.loc[first].reset_index(drop=True), els.loc[first].reset_index(drop=True)


def draw(els: pd.Series, mask: np.ndarray, n_train: int, n_val: int, emphasis: set,
         oversample: float, rng: np.random.Generator):
    """Weighted sample without replacement among rows where mask is True; rows containing any
    `emphasis` element get weight `oversample` (others 1). Returns (train_idx, val_idx)."""
    idx = np.flatnonzero(mask)
    n = min(max(n_train, 0) + max(n_val, 0), len(idx))
    if n == 0:
        return [], []
    w = 1 + (oversample - 1) * els.iloc[idx].map(lambda s: bool(s & emphasis)).values
    pick = idx[rng.choice(len(idx), size=n, replace=False, p=w / w.sum())]
    n_tr = min(n_train, n)
    return [int(i) for i in pick[:n_tr]], [int(i) for i in pick[n_tr:]]


def select_materials(els: pd.Series, adapted: set, n_materials: int, n_materials_val: int,
                     n_only_adapted: int, oversample: float, rng: np.random.Generator) -> dict:
    """Split eligible materials into train/val index lists with an only-adapted (C-type) quota.
    C-type rows over-sample H/F/O; the rest over-sample H/F."""
    only = els.map(lambda s: s <= adapted).values
    frac_val = n_materials_val / max(n_materials, 1)
    n_c = min(n_only_adapted, n_materials)
    c_tr, c_va = (draw(els, only, n_c, round(n_c * frac_val), {"H", "F", "O"}, oversample, rng)
                  if n_c else ([], []))
    b_tr, b_va = draw(els, ~only, n_materials - len(c_tr), n_materials_val - len(c_va),
                      {"H", "F"}, oversample, rng)
    return {"train": c_tr + b_tr, "val": c_va + b_va, "only_adapted_train": len(c_tr),
            "only_adapted_val": len(c_va), "only_adapted_eligible": int(only.sum())}


def composition_matrix(atoms_list, elements) -> np.ndarray:
    return np.array([[a.get_chemical_symbols().count(e) for e in elements] for a in atoms_list],
                    dtype=float)


def fit_energy_offsets(atoms_list, teacher_energies, base_energies):
    """Per-element offsets (teacher - base) by least squares. Returns (elements, offsets)."""
    elements = sorted({e for a in atoms_list for e in a.get_chemical_symbols()})
    comp = composition_matrix(atoms_list, elements)
    offsets, *_ = np.linalg.lstsq(comp, np.asarray(teacher_energies) - np.asarray(base_energies),
                                  rcond=None)
    return elements, offsets


def apply_energy_offsets(df: pd.DataFrame, elements, offsets) -> pd.DataFrame:
    """Subtract sum_el n_el * offset_el from energy and energy_corrected (forces unchanged)."""
    shift = composition_matrix(df.ase_atoms, elements) @ np.asarray(offsets)
    return df.assign(energy=df.energy - shift, energy_corrected=df.energy_corrected - shift)


def offsets_payload(elements, offsets, n_fit: int) -> dict:
    return {"note": "organic teacher energies converted to base GRACE reference: "
                    "E - sum_el n_el*offset_el",
            "offsets_eV": dict(zip(elements, map(float, offsets))), "n_fit": int(n_fit)}


def offsets_from_payload(payload: dict, needed_elements) -> tuple:
    """(elements, offsets) from an energy_reference.json payload; SystemExit if an element of
    the data has no offset."""
    table = payload["offsets_eV"]
    missing = sorted(set(needed_elements) - set(table))
    if missing:
        raise SystemExit(f"energy reference has no offset for {missing}; rebuild it "
                         "(pipeline.py prepare --force)")
    elements = sorted(table)
    return elements, np.array([table[e] for e in elements])


def organic_subset(train: pd.DataFrame, n_organic) -> pd.DataFrame:
    """The first n_organic molecules of the organic training pool (all if None)."""
    if n_organic is None:
        return train
    if n_organic > len(train):
        raise SystemExit(f"--n-organic {n_organic} > organic pool ({len(train)}); "
                         "rerun `pipeline.py prepare` with a larger --n-organic-pool")
    return train.iloc[:n_organic].reset_index(drop=True)


def materials_frame(df: pd.DataFrame) -> pd.DataFrame:
    """MPtrj parquet rows -> training DataFrame (ase_atoms, energy, forces, DFT columns)."""
    from ase import Atoms

    rows = []
    for r in df.itertuples():
        atoms = Atoms(numbers=r.numbers, positions=np.stack(r.positions), cell=np.stack(r.cell),
                      pbc=True)
        rows.append({"ase_atoms": atoms, "energy": float(r.energy),
                     "energy_corrected": float(r.energy), "forces": np.stack(r.forces),
                     "dft_energy": float(r.energy), "dft_forces": np.stack(r.forces),
                     "source": "materials", "mp_id": r.mp_id})
    return pd.DataFrame(rows, columns=LABEL_COLUMNS + ["dft_energy", "dft_forces", "source", "mp_id"])


def monitor_frames(organic_test: pd.DataFrame, monitor_shard: pd.DataFrame, adapted: set,
                   max_atoms: int, n_monitor: int, seed: int) -> dict:
    """The three fixed held-out monitor sets (see docs/CONTRACTS.md):
    organic = first n_monitor test molecules (teacher labels); materials = n_monitor shard
    structures with >= 1 adapted element; only_adapted = up to n_monitor shard structures made
    only of adapted elements; both <= max_atoms atoms (DFT labels)."""
    frames = {"organic": organic_test.iloc[:n_monitor][LABEL_COLUMNS].reset_index(drop=True)}
    els = element_sets(monitor_shard.numbers)
    small = (monitor_shard.num_atoms <= max_atoms).values
    for name, mask in [("materials", small & els.map(lambda s: bool(s & adapted)).values),
                       ("only_adapted", small & els.map(lambda s: s <= adapted).values)]:
        pool = monitor_shard[mask]
        pick = pool.sample(n=min(n_monitor, len(pool)), random_state=seed)
        frames[name] = materials_frame(pick)[LABEL_COLUMNS]
    return frames


def write_monitors(frames: dict, out: Path, prefix: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, df in frames.items():
        if len(df):
            path = out / f"{prefix}{name}.pkl.gz"
            config.save_frame(df, path)
            written[name] = (path, len(df))
    return written


def read_ids(paths) -> set:
    ids = set()
    for path in paths:
        ids |= set(pd.read_parquet(Path(path).expanduser(), columns=["mp_id"]).mp_id)
    return ids


# --------------------------------------------------------------------------------------
def base_energies(atoms_list, base_model: Path):
    """Base GRACE energies (loads TensorFlow; only used in the default mode)."""
    import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
    from tensorpotential.calculator import TPCalculator
    from tqdm import tqdm

    calc = TPCalculator(model=str(base_model))
    out = []
    for a in tqdm(atoms_list, desc="base energies (organics)"):
        b = a.copy()
        b.calc = calc
        out.append(b.get_potential_energy())
    return np.array(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--organic-dir", default=config.ORGANIC_DIR,
                   help="folder with train/val/test.pkl.gz (teacher labels)")
    p.add_argument("--materials", nargs="*", default=[],
                   help="MPtrj parquet shard(s) used for training (not needed with --n-materials 0)")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="parquet shards whose materials must never be trained on (the first one "
                        "also provides the monitor materials)")
    p.add_argument("--base", default=config.STUDENT_MODEL_DIR, help="base GRACE SavedModel")
    p.add_argument("--adapted-from", default=config.TEACHER_MODEL_DIR,
                   help="model whose elements are the adapted ones (default: the teacher)")
    p.add_argument("--n-organic", type=int, default=None,
                   help="use the first N molecules of organic train.pkl.gz (default: all)")
    p.add_argument("--n-materials", type=int, default=500, help="0 = organic-only training data")
    p.add_argument("--n-materials-val", type=int, default=100)
    p.add_argument("--n-only-adapted", type=int, default=0,
                   help="how many of --n-materials (and proportionally of the val set) must contain "
                        "ONLY adapted elements (molecular crystals, C/P/S solids...); 0 = no quota")
    p.add_argument("--n-monitor", type=int, default=150, help="structures per monitor set")
    p.add_argument("--max-atoms", type=int, default=30, help="largest material (memory, speed)")
    p.add_argument("--oversample", type=float, default=3.0, help="weight for structures with H or F")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--energy-reference", default=None,
                   help="reuse per-element offsets from this energy_reference.json (e.g. "
                        "<workdir>/data/monitors/energy_reference.json) instead of fitting them "
                        "against base GRACE again")
    p.add_argument("--n-offset-fit", type=int, default=500,
                   help="--monitors-only: organic train molecules used to fit the offsets")
    p.add_argument("--no-monitors", action="store_true",
                   help="default mode: do not write the legacy monitor_*.pkl.gz files")
    p.add_argument("--monitors-only", action="store_true",
                   help="only write organic/materials/only_adapted.pkl.gz monitor sets into --out")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    out = Path(args.out).expanduser()
    organic_dir = Path(args.organic_dir).expanduser()
    adapted_dir = config.require(args.adapted_from, "TEACHER_MODEL_DIR")
    adapted = set(config.model_elements(adapted_dir))
    if not adapted:
        raise SystemExit(f"No chemical_symbols in {adapted_dir}/metadata.yaml")

    if args.monitors_only:
        if not args.exclude:
            raise SystemExit("--monitors-only needs --exclude <val.parquet> (monitor materials)")
        frames = monitor_frames(config.load_frame(organic_dir / "test.pkl.gz"),
                                pd.read_parquet(Path(args.exclude[0]).expanduser()), adapted,
                                args.max_atoms, args.n_monitor, args.seed)
        # organic monitor energies -> base GRACE reference (same offsets as the training data)
        config.setup_device()
        pool = config.load_frame(organic_dir / "train.pkl.gz")
        fit = organic_subset(pool, min(args.n_offset_fit, len(pool)))
        base_e = base_energies(list(fit.ase_atoms), config.require(args.base, "STUDENT_MODEL_DIR"))
        elements, offsets = fit_energy_offsets(list(fit.ase_atoms), fit.energy.values, base_e)
        mon_elements = {e for a in frames["organic"].ase_atoms for e in a.get_chemical_symbols()}
        elements, offsets = offsets_from_payload(offsets_payload(elements, offsets, len(fit)),
                                                 mon_elements)
        frames["organic"] = apply_energy_offsets(frames["organic"], elements, offsets)
        out.mkdir(parents=True, exist_ok=True)
        (out / "energy_reference.json").write_text(
            json.dumps(offsets_payload(elements, offsets, len(fit)), indent=2))
        written = write_monitors(frames, out)
        print("Monitors: " + ", ".join(f"{k} {n} -> {p_}" for k, (p_, n) in written.items())
              + f"; energy_reference.json (offsets fitted on {len(fit)} molecules)")
        return

    config.setup_device()
    rng = np.random.default_rng(args.seed)
    out.mkdir(parents=True, exist_ok=True)

    # ---- organics: move teacher energies onto base GRACE's energy reference ----------------
    org = {"train": organic_subset(config.load_frame(organic_dir / "train.pkl.gz"), args.n_organic),
           "val": config.load_frame(organic_dir / "val.pkl.gz")}
    needed = {e for s in ("train", "val") for a in org[s].ase_atoms for e in a.get_chemical_symbols()}
    if args.energy_reference and Path(args.energy_reference).expanduser().exists():
        payload = json.loads(Path(args.energy_reference).expanduser().read_text())
        elements, offsets = offsets_from_payload(payload, needed)
        n_fit = payload.get("n_fit", 0)
        print(f"Energy offsets reused from {args.energy_reference}")
    else:
        base_e = base_energies(list(org["train"].ase_atoms),
                               config.require(args.base, "STUDENT_MODEL_DIR"))
        elements, offsets = fit_energy_offsets(list(org["train"].ase_atoms),
                                               org["train"].energy.values, base_e)
        n_fit = len(org["train"])
    for s in ("train", "val"):
        org[s] = apply_energy_offsets(org[s], elements, offsets).assign(source="organic")
    payload = offsets_payload(elements, offsets, n_fit)
    payload["n_organic"] = len(org["train"])
    (out / "energy_reference.json").write_text(json.dumps(payload, indent=2))
    print(f"Organics: {len(org['train'])} train, {len(org['val'])} val. Energy offsets "
          "(teacher - base, eV/atom): " + ", ".join(f"{e} {o:+.2f}" for e, o in zip(elements, offsets)))

    # ---- materials: adapted-element structures, H/F over-sampled, excluded shards removed -----
    m_train = m_val = pd.DataFrame(columns=LABEL_COLUMNS + ["source"])
    if args.n_materials > 0:
        if not args.materials:
            raise SystemExit("--n-materials > 0 needs --materials <parquet shards>")
        mats = pd.concat([pd.read_parquet(Path(m).expanduser()) for m in args.materials],
                         ignore_index=True)
        excluded = read_ids(args.exclude)
        mats, els = eligible_materials(mats, adapted, args.max_atoms, excluded, args.seed)
        sel = select_materials(els, adapted, args.n_materials, args.n_materials_val,
                               args.n_only_adapted, args.oversample, rng)
        m_train = materials_frame(mats.iloc[sel["train"]])
        m_val = materials_frame(mats.iloc[sel["val"]])
        print(f"Materials only-adapted (C-type): {sel['only_adapted_eligible']} eligible; "
              f"train {sel['only_adapted_train']}, val {sel['only_adapted_val']}")
        print(f"Materials: {len(mats)} eligible (adapted elements, <= {args.max_atoms} atoms, "
              f"{len(excluded)} excluded ids); train {len(m_train)}, val {len(m_val)}")
        for name, df in [("train", m_train), ("val", m_val)]:
            if len(df):
                e = df.ase_atoms.map(lambda a: set(a.get_chemical_symbols()))
                print(f"  {name}: " + ", ".join(f"with {x} {e.map(lambda s, x=x: x in s).mean():.0%}"
                                                for x in ("O", "H", "F", "N")))
    else:
        print("Materials: none (--n-materials 0, organic-only training data)")

    cols = LABEL_COLUMNS + ["source"]
    train = pd.concat([org["train"][cols], m_train[cols]]).sample(frac=1, random_state=args.seed)
    val = pd.concat([org["val"][cols], m_val[cols]]).sample(frac=1, random_state=args.seed)
    config.save_frame(train.reset_index(drop=True), out / "train.pkl.gz")
    config.save_frame(val.reset_index(drop=True), out / "val.pkl.gz")
    saved = [f"train.pkl.gz ({len(train)})", f"val.pkl.gz ({len(val)})", "energy_reference.json"]

    # ---- legacy monitor files next to the training data (held out from training) ----------
    if not args.no_monitors and args.exclude:
        frames = monitor_frames(config.load_frame(organic_dir / "test.pkl.gz"),
                                pd.read_parquet(Path(args.exclude[0]).expanduser()), adapted,
                                args.max_atoms, args.n_monitor, args.seed)
        frames["organic"] = apply_energy_offsets(frames["organic"], elements, offsets)
        written = write_monitors(frames, out, prefix="monitor_")
        saved += [f"{p_.name} ({n})" for p_, n in written.values()]
    print(f"Saved in {out}: " + ", ".join(saved))


if __name__ == "__main__":
    main()
