"""Find, check and recommend TEACHER models for the LoRA distillation loop.

Only GRACE / tensorpotential SavedModels are supported: a folder with saved_model.pb,
variables/ and metadata.yaml. MACE / PyTorch models (*.model, *.pt) are rejected.
Stdlib only (metadata.yaml is read as text; no model is ever loaded).

    python scripts/teacher_tools.py --workdir W resolve <spec>
        spec: GRACE-OFF-{1L,2L}-{small,medium,large} | grace:<foundation-model> |
              <local path> | <git URL (github/gitlab/huggingface)> | <DOI / paper URL>
    python scripts/teacher_tools.py check <saved_model_dir>
    python scripts/teacher_tools.py --workdir W recommend --elements C H O N [--domain "text"]

Every command ends with one line `RESULT:{json}`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

GRACE_OFF_REPO = "https://github.com/heid-lab/grace-off.git"
GRACE_OFF = {f"GRACE-OFF-{layers.upper()}-{size}": f"models/{layers}/b_off_{size}/seed/1/saved_model"
             for layers in ("1l", "2l") for size in ("small", "medium", "large")}
GRACE_OFF_ELEMENTS = ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"]
# elements of the GRACE OMAT foundation models (student and grace:<name> teachers)
FOUNDATION_ELEMENTS = (
    "Ac Ag Al Ar As Au B Ba Be Bi Br C Ca Cd Ce Cl Co Cr Cs Cu Dy Er Eu F Fe Ga Gd Ge H He Hf "
    "Hg Ho I In Ir K Kr La Li Lu Mg Mn Mo N Na Nb Nd Ne Ni Np O Os P Pa Pb Pd Pm Pr Pt Pu Rb "
    "Re Rh Ru S Sb Sc Se Si Sm Sn Sr Ta Tb Tc Te Th Ti Tl Tm U V W Xe Y Yb Zn Zr").split()
PERIODIC = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr "
            "Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra "
            "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts "
            "Og").split()

CATALOG = (
    [{"spec": name, "name": name, "elements": GRACE_OFF_ELEMENTS,
      "domain": "organic molecules and liquids (drug-like molecules, peptides, solvents)",
      "keywords": ["organic", "molecule", "molecular", "liquid", "drug", "peptide", "protein",
                   "solvent", "polymer", "biomolecule", "spice"],
      "level_of_theory": "SPICE v2, wB97M-D3(BJ)/def2-TZVPPD",
      "source": "heid-lab/grace-off (GRACE-OFF, JCTC 2026)",
      "size": name.rsplit("-", 1)[-1], "default": name == "GRACE-OFF-2L-medium"}
     for name in GRACE_OFF]
    + [{"spec": f"grace:{name}", "name": name, "elements": FOUNDATION_ELEMENTS,
        "domain": "inorganic materials (no specialist knowledge)",
        "keywords": ["material", "materials", "crystal", "alloy", "oxide", "metal", "inorganic",
                     "solid", "bulk", "surface"],
        "level_of_theory": "PBE/PBE+U (OMat24, MPtrj, sAlex)",
        "source": "GRACE foundation models (grace_models)",
        "size": "large", "default": False,
        "why_extra": "use only to distill a larger GRACE into the student, not for a new domain"}
       for name in ("GRACE-2L-OMAT-large-ft-AM", "GRACE-3L-OMAT-large-ft-AM")]
)
NOT_GRACE_PATTERNS = ("*.model", "*.pt", "*.pth", "*.ckpt", "*.torchscript")
PUBLICATION = re.compile(r"(^10\.\d{4,9}/\S+$)|doi\.org/|arxiv\.org/|chemrxiv|pubs\.acs\.org|"
                         r"nature\.com/|sciencedirect\.com|wiley\.com|iopscience|aip\.org/|"
                         r"rsc\.org/|springer\.com/|\.pdf$", re.I)
GIT_HOSTS = ("github.com", "gitlab.", "huggingface.co", "bitbucket.org", "codeberg.org")


# --------------------------------------------------------------------------------------
# metadata / model inspection
# --------------------------------------------------------------------------------------
def read_metadata(model_dir) -> dict:
    """{'elements': [...]|None, 'cutoff': float|None} from metadata.yaml (text parsing)."""
    meta = Path(model_dir) / "metadata.yaml"
    out = {"elements": None, "cutoff": None}
    if not meta.exists():
        return out
    symbols, inside = [], False
    for line in meta.read_text(errors="replace").splitlines():
        if line.startswith("chemical_symbols:"):
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("["):
                symbols = [s.strip().strip("'\"") for s in inline.strip("[]").split(",") if s.strip()]
            else:
                inside = True
            continue
        if inside:
            m = re.match(r"\s*-\s*(\S+)", line)
            if m:
                symbols.append(m.group(1).strip("'\""))
                continue
            inside = False
        m = re.match(r"cutoff:\s*([0-9.eE+-]+)", line)
        if m:
            out["cutoff"] = float(m.group(1))
    out["elements"] = symbols or None
    return out


def _variables_size(model_dir: Path) -> int:
    var = model_dir / "variables"
    return sum(f.stat().st_size for f in var.glob("variables.data-*")) if var.is_dir() else 0


def precision(model_dir) -> tuple[str, bool]:
    """(fp64|fp32|unknown, is_casted). casted_model folders are fp32; a saved_model next to a
    casted_model is fp64 when its variables are ~2x larger; '-fp64' in a name means fp64."""
    model_dir = Path(model_dir)
    if model_dir.name == "casted_model":
        return "fp32", True
    if "fp64" in str(model_dir).lower():
        return "fp64", False
    sibling = model_dir.parent / "casted_model"
    if sibling.is_dir():
        mine, casted = _variables_size(model_dir), _variables_size(sibling)
        if mine and casted:
            return ("fp64" if mine / casted > 1.6 else "fp32"), False
    return "unknown", False


def _lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def non_grace_files(folder: Path) -> list:
    return sorted(str(p.name) for pattern in NOT_GRACE_PATTERNS for p in folder.glob(pattern))


def check_model(model_dir) -> dict:
    """Validate one SavedModel folder without loading it."""
    model_dir = Path(model_dir).expanduser()
    warnings, ok = [], True
    if not model_dir.is_dir():
        return {"model_dir": str(model_dir), "ok": False, "elements": None, "cutoff": None,
                "precision": "unknown", "warnings": [f"{model_dir} is not a folder"]}
    foreign = non_grace_files(model_dir)
    pb = model_dir / "saved_model.pb"
    if not pb.exists():
        ok = False
        if foreign:
            warnings.append(f"not a GRACE model: found {', '.join(foreign)} (MACE/PyTorch). Only "
                            "GRACE/tensorpotential SavedModels are supported as teachers.")
        else:
            warnings.append("saved_model.pb missing: not a TensorFlow SavedModel")
    elif _lfs_pointer(pb):
        ok = False
        warnings.append("saved_model.pb is a git-lfs pointer, not the real file: install git-lfs "
                        "and run `git lfs pull` in the cloned repository")
    if not (model_dir / "variables").is_dir():
        ok = False
        warnings.append("variables/ folder missing")
    meta = read_metadata(model_dir)
    if meta["elements"] is None:
        ok = False
        warnings.append("metadata.yaml missing or without chemical_symbols: cannot tell which "
                        "elements the model knows")
    if meta["cutoff"] is None:
        warnings.append("cutoff not found in metadata.yaml")
    prec, casted = precision(model_dir)
    if casted:
        warnings.append("casted_model is fp32 and crashes with the fp64 inputs of the calculator: "
                        "use the sibling saved_model folder instead")
        ok = False
    return {"model_dir": str(model_dir), "ok": ok, "elements": meta["elements"],
            "cutoff": meta["cutoff"], "precision": prec, "warnings": warnings}


def find_candidates(root: Path) -> list:
    """Every folder under root containing saved_model.pb (skipping checkpoints and .git),
    saved_model folders first, casted_model last."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "checkpoints", "__pycache__")]
        if "saved_model.pb" in filenames:
            path = Path(dirpath)
            meta = read_metadata(path)
            prec, casted = precision(path)
            found.append({"path": str(path), "elements": meta["elements"], "cutoff": meta["cutoff"],
                          "precision": prec, "is_casted": casted})
    found.sort(key=lambda c: (c["is_casted"], Path(c["path"]).name != "saved_model", c["path"]))
    return found


# --------------------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------------------
def grace_cache() -> Path:
    return Path(os.environ.get("GRACE_CACHE") or "~/.cache/grace").expanduser()


def classify(spec: str) -> str:
    if spec in GRACE_OFF:
        return "builtin"
    if spec.startswith("grace:"):
        return "grace_models"
    if spec.startswith("git@") or spec.endswith(".git") or (
            spec.startswith(("http://", "https://")) and any(h in spec for h in GIT_HOSTS)
            and not PUBLICATION.search(spec)):
        return "git"
    if PUBLICATION.search(spec) or spec.lower().startswith("doi:"):
        return "publication"
    return "local"


def repo_name(url: str) -> str:
    name = url.rstrip("/").split("/")[-1].split(":")[-1]
    return re.sub(r"\.git$", "", name) or "repo"


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def clone_repo(url: str, dest: Path) -> list:
    """Shallow, blobless clone that checks out only the folders holding saved_model.pb
    (a model repository can be hundreds of MB). Returns notes. Existing clones are reused."""
    notes = []
    if (dest / ".git").exists():
        notes.append(f"reusing existing clone {dest}")
        return notes
    if shutil.which("git") is None:
        raise RuntimeError("git is not installed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = _git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", url, str(dest))
    if r.returncode != 0:
        raise RuntimeError(f"git clone failed: {r.stderr.strip()[-400:]}")
    files = _git("ls-tree", "-r", "--name-only", "HEAD", cwd=dest).stdout.splitlines()
    model_dirs = sorted({str(Path(f).parent) for f in files if Path(f).name == "saved_model.pb"
                         and "checkpoints" not in Path(f).parts})
    if model_dirs:
        patterns = [f"/{d}/" for d in model_dirs]
        _git("sparse-checkout", "set", "--no-cone", *patterns, cwd=dest)
        notes.append(f"checked out only the {len(model_dirs)} model folder(s) of the repository")
    r = _git("checkout", cwd=dest)
    if r.returncode != 0:
        raise RuntimeError(f"git checkout failed: {r.stderr.strip()[-400:]}")
    return notes


def _choose(candidates: list, notes: list):
    usable = [c for c in candidates if not c["is_casted"]]
    if len(usable) == 1:
        return usable[0]["path"]
    if not usable and candidates:
        notes.append("only casted_model (fp32) folders found: these crash with fp64 inputs")
    elif len(usable) > 1:
        notes.append(f"{len(usable)} candidate models found: call `resolve <path>` with the one "
                     "you want (they are listed in candidates)")
    return None


def resolve(spec: str, workdir) -> dict:
    workdir = Path(workdir).expanduser()
    kind = classify(spec)
    result = {"spec": spec, "kind": kind, "model_dir": None, "candidates": [], "elements": None,
              "ok": False, "needs_fetch": False, "notes": []}
    notes = result["notes"]

    if kind == "builtin":
        path = workdir / "models" / "grace-off" / GRACE_OFF[spec]
        result.update(model_dir=str(path), elements=list(GRACE_OFF_ELEMENTS), ok=True)
        if not (path / "saved_model.pb").exists():
            result["needs_fetch"] = True
            notes.append("not downloaded yet: `python3 pipeline.py fetch` downloads it")
        else:
            result["candidates"] = find_candidates(path)

    elif kind == "grace_models":
        name = spec.split(":", 1)[1]
        path = grace_cache() / name
        result.update(model_dir=str(path), ok=bool(name))
        if (path / "saved_model.pb").exists():
            result["elements"] = read_metadata(path)["elements"]
            result["candidates"] = find_candidates(path)
        else:
            result.update(needs_fetch=True, elements=list(FOUNDATION_ELEMENTS))
            notes.append(f"not downloaded yet: `grace_models download {name}` (or pipeline.py fetch)")
        notes.append("foundation models carry no specialist knowledge: only useful to distill a "
                     "larger GRACE into the student")

    elif kind == "publication":
        notes.append("publications are not scraped: open the paper's 'Code/Data availability' "
                     "section, find the GitHub/GitLab/Zenodo/Hugging Face link to the model "
                     "weights, then call `resolve <that link>`")

    elif kind in ("git", "local"):
        if kind == "git":
            dest = workdir / "models" / "teachers" / repo_name(spec)
            try:
                notes += clone_repo(spec, dest)
            except RuntimeError as exc:
                notes.append(str(exc))
                return result
            root = dest
        else:
            root = Path(spec).expanduser()
            if not root.exists():
                notes.append(f"{root} does not exist")
                return result
        foreign = non_grace_files(root) if root.is_dir() else (
            [root.name] if any(root.match(p) for p in NOT_GRACE_PATTERNS) else [])
        candidates = find_candidates(root) if root.is_dir() else []
        result["candidates"] = candidates
        if not candidates:
            if foreign:
                notes.append(f"not a GRACE model: {', '.join(foreign)} look like MACE/PyTorch "
                             "files. Only GRACE/tensorpotential SavedModels are supported.")
            else:
                notes.append("no folder with saved_model.pb found")
            return result
        chosen = _choose(candidates, notes)
        if chosen:
            check = check_model(chosen)
            notes += check["warnings"]
            result.update(model_dir=chosen, elements=check["elements"], ok=check["ok"])
    return result


# --------------------------------------------------------------------------------------
# recommend
# --------------------------------------------------------------------------------------
def count_materials(rows, elements: set, max_atoms: int = 30, exclude_ids=()) -> dict:
    """rows: iterable of (numbers, mp_id|None). Counts distinct materials (by mp_id when
    given) with >=1 of the elements and made only of them, <= max_atoms."""
    with_any, only, seen = set(), set(), set()
    exclude_ids = set(exclude_ids)
    for i, (numbers, mp_id) in enumerate(rows):
        key = mp_id if mp_id is not None else i
        if key in seen or key in exclude_ids or len(numbers) > max_atoms:
            continue
        seen.add(key)
        symbols = {PERIODIC[z] if 0 < z < len(PERIODIC) else "?" for z in numbers}
        if symbols & elements:
            with_any.add(key)
            if symbols <= elements:
                only.add(key)
    return {"materials_with_elements": len(with_any), "only_these_elements": len(only)}


def replay_counts(workdir: Path, elements: set) -> dict:
    folder = workdir / "data" / "materials"
    shards = sorted(p for p in folder.glob("*.parquet") if p.name != "val.parquet") \
        if folder.is_dir() else []
    out = {"available": False, "materials_with_elements": None, "only_these_elements": None,
           "suggested": {"n_materials": 600, "n_only_adapted": 170}, "note": None}
    if not shards:
        out["note"] = "no MPtrj shards in data/materials yet (`pipeline.py prepare` downloads them)"
        return out
    try:
        import pyarrow.parquet as pq
    except ImportError:
        out["note"] = "pyarrow not available in this Python: counts unavailable (run with the venv)"
        return out
    exclude = set()
    val = folder / "val.parquet"
    if val.exists():
        exclude = set(pq.read_table(val, columns=["mp_id"]).column("mp_id").to_pylist())

    def rows():
        for shard in shards:
            cols = [c for c in ("numbers", "mp_id") if c in pq.read_schema(shard).names]
            table = pq.read_table(shard, columns=cols)
            numbers = table.column("numbers").to_pylist()
            ids = table.column("mp_id").to_pylist() if "mp_id" in cols else [None] * len(numbers)
            yield from zip(numbers, ids)

    counts = count_materials(rows(), elements, exclude_ids=exclude)
    out.update(available=True, **counts)
    out["suggested"] = {"n_materials": min(600, counts["materials_with_elements"]),
                        "n_only_adapted": min(170, int(0.3 * counts["only_these_elements"]))}
    return out


def local_teachers(workdir: Path) -> list:
    """Teachers already resolved/downloaded under <workdir>/models (not the builtin paths)."""
    root = workdir / "models"
    if not root.is_dir():
        return []
    builtin = re.compile(r"models/[12]l/b_off_(small|medium|large)/seed/1/saved_model$")
    out = []
    for c in find_candidates(root):
        path = Path(c["path"])
        if c["is_casted"] or not c["elements"] or builtin.search(c["path"].replace(os.sep, "/")):
            continue  # unusable, or the same model as a catalog GRACE-OFF entry
        if path.name == "final_model" and (path.parent / "saved_model").is_dir():
            continue  # duplicate export of the sibling saved_model
        parts = [q for q in path.relative_to(root).parts
                 if q not in ("teachers", "models", "seed", "1", "saved_model")]
        out.append({"spec": c["path"], "name": "/".join(parts) or path.name,
                    "elements": c["elements"], "domain": "user-provided teacher",
                    "keywords": [], "level_of_theory": "unknown (check the source)",
                    "source": "local", "size": "-", "default": False})
    return out


def recommend(elements: list, domain: str, workdir) -> dict:
    workdir = Path(workdir).expanduser()
    wanted = {e.strip() for e in elements if e.strip()}
    words = set(re.findall(r"[a-z]+", (domain or "").lower()))
    teachers = []
    for t in CATALOG + local_teachers(workdir):
        missing = sorted(wanted - set(t["elements"]))
        match = sum(1 for k in t["keywords"] if k in words or k.rstrip("s") in words)
        is_foundation = t["spec"].startswith("grace:")
        why = []
        why.append("covers all requested elements" if not missing
                   else f"missing {' '.join(missing)}")
        if match:
            why.append(f"domain match ({t['domain']})")
        if t.get("why_extra"):
            why.append(t["why_extra"])
        teachers.append({"spec": t["spec"], "name": t["name"], "covers_all": not missing,
                         "missing": missing, "domain": t["domain"],
                         "level_of_theory": t["level_of_theory"], "why": "; ".join(why),
                         "_rank": (not missing, match, not is_foundation, t.get("default", False),
                                   t["size"] == "medium")})
    teachers.sort(key=lambda t: t["_rank"], reverse=True)
    for t in teachers:
        t.pop("_rank")

    notes = []
    specialists = [t for t in teachers if t["covers_all"] and not t["spec"].startswith("grace:")]
    organic_like = {"C", "H"} <= wanted
    if wanted - set(FOUNDATION_ELEMENTS):
        notes.append(f"{' '.join(sorted(wanted - set(FOUNDATION_ELEMENTS)))} not in base GRACE: "
                     "the student itself cannot represent these elements")
    elif not organic_like:
        notes.append("these elements are base GRACE's home domain (inorganic materials): a "
                     "specialist teacher is only worth it for molecular systems; otherwise "
                     "fine-tune on DFT labels of your own structures")
    if not specialists:
        notes.append("no specialist teacher in the catalog covers all requested elements: "
                     "provide a teacher link (`resolve <url>`) or use DFT labels")
    replay = replay_counts(workdir, wanted)
    if replay.get("note"):
        notes.append(replay["note"])
    return {"elements": sorted(wanted), "domain": domain, "teachers": teachers,
            "replay": {k: replay[k] for k in ("available", "materials_with_elements",
                                              "only_these_elements", "suggested")},
            "notes": notes}


def format_recommendation(rec: dict) -> str:
    lines = [f"Elements: {' '.join(rec['elements'])}" + (f"   domain: {rec['domain']}" if rec["domain"] else ""),
             "", f"{'#':>2s} {'teacher':28s} {'covers':7s} {'level of theory':38s} why"]
    for i, t in enumerate(rec["teachers"], 1):
        label = t["spec"] if not t["spec"].startswith("/") else f"local:{t['name']}"
        lines.append(f"{i:2d} {label[:28]:28s} {'yes' if t['covers_all'] else 'no':7s} "
                     f"{t['level_of_theory'][:38]:38s} {t['why']}")
    r = rec["replay"]
    lines.append("")
    if r["available"]:
        lines.append(f"Replay materials (MPtrj, <= 30 atoms, val shard excluded): "
                     f"{r['materials_with_elements']} with these elements, "
                     f"{r['only_these_elements']} made only of them -> suggest "
                     f"n_materials={r['suggested']['n_materials']}, "
                     f"n_only_adapted={r['suggested']['n_only_adapted']}")
    else:
        lines.append("Replay materials: counts unavailable (defaults n_materials=600, "
                     "n_only_adapted=170)")
    lines += [f"NOTE: {n}" for n in rec["notes"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default=".", help="pipeline workdir (models/, data/)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve", help="turn a teacher spec into a SavedModel folder")
    r.add_argument("spec")
    c = sub.add_parser("check", help="validate a SavedModel folder (no loading)")
    c.add_argument("model_dir")
    rec = sub.add_parser("recommend", help="rank teachers for a set of elements")
    rec.add_argument("--elements", nargs="+", required=True)
    rec.add_argument("--domain", default="")
    args = p.parse_args(argv)

    if args.cmd == "resolve":
        result = resolve(args.spec, args.workdir)
        print(f"{result['kind']}: {result['model_dir'] or 'no model chosen'}"
              + (" (needs fetch)" if result["needs_fetch"] else ""))
        for cand in result["candidates"]:
            print(f"  candidate {cand['path']}  [{cand['precision']}"
                  f"{', casted' if cand['is_casted'] else ''}] elements "
                  f"{' '.join(cand['elements'] or ['?'])}")
        for note in result["notes"]:
            print(f"  NOTE: {note}")
    elif args.cmd == "check":
        result = check_model(args.model_dir)
        print(f"{'OK' if result['ok'] else 'NOT USABLE'}: {result['model_dir']} "
              f"({result['precision']}, cutoff {result['cutoff']}, "
              f"{len(result['elements'] or [])} elements)")
        for w in result["warnings"]:
            print(f"  WARNING: {w}")
    else:
        result = recommend(args.elements, args.domain, args.workdir)
        print(format_recommendation(result))
    print("RESULT:" + json.dumps(result))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
