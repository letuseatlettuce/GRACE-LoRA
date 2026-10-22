"""Machine-readable facts for pipeline.py (runs inside the project venv; prints one
line `JSON:{...}`).

    python scripts/inspect_setup.py versions
    python scripts/inspect_setup.py catalog
    python scripts/inspect_setup.py elements --student-model DIR --teacher DIR [--xyz FILE ...]
"""

import argparse
import importlib.metadata as md
import json
from pathlib import Path

PINNED = {"tensorpotential": "0.6.2.1", "tensorflow": "2.19.1", "tf-keras": "2.19.0"}
OTHERS = ["tensorflow-metal", "ase", "numpy", "pandas", "scipy", "rdkit"]


def versions():
    out = {}
    for pkg in list(PINNED) + OTHERS:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return {"versions": out, "pinned": PINNED,
            "ok": all(out[p] == v for p, v in PINNED.items())}


def catalog():
    from tensorpotential.calculator.foundation_models import (
        CHECKPOINT_URL_KEY, DESCRIPTION_KEY, MODELS_METADATA,
    )

    students = [
        {"name": name, "description": meta.get(DESCRIPTION_KEY, "")}
        for name, meta in MODELS_METADATA.items()
        if meta.get(CHECKPOINT_URL_KEY) and not name.endswith("-fp64")
    ]
    return {"students": students}


def model_elements(model_dir):
    import yaml

    meta = Path(model_dir).expanduser() / "metadata.yaml"
    if not meta.exists():
        return None
    return sorted(yaml.safe_load(meta.read_text()).get("chemical_symbols", []))


def xyz_summary(path):
    from ase.io import iread

    elements, n, n_labelled, n_atoms = set(), 0, 0, 0
    for atoms in iread(str(Path(path).expanduser()), index=":"):
        n += 1
        n_atoms += len(atoms)
        elements.update(atoms.get_chemical_symbols())
        if atoms.calc is not None and "forces" in atoms.calc.results:
            n_labelled += 1
    return {"structures": n, "with_dft_forces": n_labelled,
            "mean_atoms": n_atoms / max(n, 1), "elements": sorted(elements)}


def elements(args):
    out = {"student": model_elements(args.student_model),
           "teacher": model_elements(args.teacher), "data": {}}
    for path in args.xyz or []:
        out["data"][path] = xyz_summary(path)
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("versions")
    sub.add_parser("catalog")
    e = sub.add_parser("elements")
    e.add_argument("--student-model", required=True)
    e.add_argument("--teacher", required=True)
    e.add_argument("--xyz", nargs="*")
    args = p.parse_args()
    result = {"versions": versions, "catalog": catalog}.get(args.cmd)
    # prefixed: tensorpotential prints info lines to stdout on import
    print("JSON:" + json.dumps(result() if result else elements(args)))


if __name__ == "__main__":
    main()
