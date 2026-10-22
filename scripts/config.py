"""Paths and settings for the GRACE LoRA pipeline.

Defaults work out of the box if you used `pipeline.py` (or the download scripts):
  - GRACE models: found by name in the GRACE cache (~/.cache/grace, or $GRACE_CACHE)
  - GRACE-OFF teacher, data, experiments, validation, report: under the work directory
    (pipeline.json "workdir"; default: this project folder)
Values written by `pipeline.py configure` into pipeline.json ("paths") override the defaults.
Every script also accepts the same values as command-line flags, which override these.
(On a Mac, ~/.cache is hidden in Finder: press Cmd+Shift+. to show hidden folders.)

This module must stay importable without TensorFlow (only `setup_device` imports it).

Runtime layout under the work directory (see docs/CONTRACTS.md):
    data/organic/{train,val,test}.pkl.gz         teacher-labelled organic pool
    data/materials/{val,test,train0}.parquet     MPtrj shards
    data/monitors/{organic,materials,only_adapted}.pkl.gz
    experiments/ledger.json, experiments/<exp_id>/...
    validation/<exp_id>/{organic,materials}/
    report/
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRACE_CACHE = Path(os.environ.get("GRACE_CACHE") or "~/.cache/grace").expanduser()
PIPELINE_SETTINGS = PROJECT_ROOT / "pipeline.json"


def _settings() -> dict:
    if PIPELINE_SETTINGS.exists():
        try:
            return json.loads(PIPELINE_SETTINGS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


_SETTINGS = _settings()
WORKDIR = str(Path(_SETTINGS.get("workdir") or PROJECT_ROOT).expanduser())
_W = Path(WORKDIR)

# --- Student: GRACE foundation model ---
STUDENT_NAME = "GRACE-2L-OMAT-medium-ft-AM"
# Folder containing model.yaml + checkpoint.index (from `grace_models checkpoint ...`)
STUDENT_CHECKPOINT_DIR = str(GRACE_CACHE / "checkpoints" / STUDENT_NAME)
# SavedModel of the same model (from `grace_models download ...`), used as the baseline
STUDENT_MODEL_DIR = str(GRACE_CACHE / STUDENT_NAME)

# --- Teacher: GRACE-OFF 2L-medium (heid-lab/grace-off, fetched by pipeline.py) ---
# Use the fp64 `saved_model` folder, NOT `casted_model` (casted_model rejects fp64 inputs)
TEACHER_MODEL_DIR = str(
    PROJECT_ROOT / "models/grace-off/models/2l/b_off_medium/seed/1/saved_model")

# --- Source data: Zenodo record 18662355 (fetched by pipeline.py) ---
# Training pool: 10k random SPICE v2 structures
TRAIN_POOL_XYZ = str(PROJECT_ROOT / "data/biased_test_set.xyz")
# Held-out evaluation set: 10k structures GRACE-OFF never saw
TEST_XYZ = str(PROJECT_ROOT / "data/GRACE_test_set.xyz")

# --- Runtime layout (outputs and prepared data) under the work directory ---
DATA_DIR = str(_W / "data")
ORGANIC_DIR = str(_W / "data/organic")
MATERIALS_DIR = str(_W / "data/materials")
MONITORS_DIR = str(_W / "data/monitors")
EXPERIMENTS_DIR = str(_W / "experiments")
VALIDATION_DIR = str(_W / "validation")
REPORT_DIR = str(_W / "report")
# legacy `pipeline.py run --plan` output folder
RUNS_DIR = str(_W / "runs")

# --- Settings written by pipeline.py (if present) override the defaults above ---
_PATH_KEYS = ("STUDENT_CHECKPOINT_DIR", "STUDENT_MODEL_DIR", "TEACHER_MODEL_DIR",
              "TRAIN_POOL_XYZ", "TEST_XYZ", "DATA_DIR", "RUNS_DIR", "ORGANIC_DIR",
              "MATERIALS_DIR", "MONITORS_DIR", "EXPERIMENTS_DIR", "VALIDATION_DIR", "REPORT_DIR")
globals().update({k: str(v) for k, v in _SETTINGS.get("paths", {}).items() if k in _PATH_KEYS})
# output folders: created on demand, never checked for existence by require()
_OUTPUT_KEYS = {"DATA_DIR", "RUNS_DIR", "ORGANIC_DIR", "MATERIALS_DIR", "MONITORS_DIR",
                "EXPERIMENTS_DIR", "VALIDATION_DIR", "REPORT_DIR"}

# MPtrj shard file names inside MATERIALS_DIR
MATERIALS_SHARDS = {"val": "val.parquet", "test": "test.parquet", "train0": "train0.parquet"}
# monitor file names inside MONITORS_DIR
MONITOR_FILES = {"organic": "organic.pkl.gz", "materials": "materials.pkl.gz",
                 "only_adapted": "only_adapted.pkl.gz"}

# --- Device ---
# The downloaded GRACE SavedModels are XLA-compiled, and XLA has no Apple Metal backend
# ("could not find registered platform"). Keep False: labelling and evaluation run on the CPU.
# Training can still use the GPU with `train_lora.py --gpu`.
USE_GPU = False


def workdir() -> Path:
    """The work directory (pipeline.json "workdir", default: the project folder)."""
    return Path(WORKDIR)


def paths() -> dict:
    """All resolved paths as Path objects (no existence checks)."""
    names = ("STUDENT_CHECKPOINT_DIR", "STUDENT_MODEL_DIR", "TEACHER_MODEL_DIR",
             "TRAIN_POOL_XYZ", "TEST_XYZ") + tuple(sorted(_OUTPUT_KEYS))
    out = {n: Path(globals()[n]).expanduser() for n in names}
    out["WORKDIR"] = workdir()
    return out


def materials_shard(name: str) -> Path:
    """Path of an MPtrj shard ('val', 'test' or 'train0') in MATERIALS_DIR."""
    return Path(MATERIALS_DIR).expanduser() / MATERIALS_SHARDS[name]


def monitor_paths() -> dict:
    """{monitor name: Path} for the fixed held-out monitor sets."""
    return {k: Path(MONITORS_DIR).expanduser() / f for k, f in MONITOR_FILES.items()}


def model_elements(model_dir) -> list:
    """Chemical symbols of a GRACE SavedModel, from its metadata.yaml (no yaml/TF needed).
    Returns [] if the file is missing."""
    meta = Path(model_dir).expanduser() / "metadata.yaml"
    if not meta.exists():
        return []
    symbols, inside = [], False
    for line in meta.read_text().splitlines():
        if line.startswith("chemical_symbols:"):
            inside = True
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("["):  # flow style: [Br, C, ...]
                return sorted(s.strip(" '\"") for s in rest.strip("[]").split(",") if s.strip())
            continue
        if inside:
            stripped = line.strip()
            if stripped.startswith("- "):
                symbols.append(stripped[2:].strip(" '\""))
            elif stripped and not line.startswith(" ") and not stripped.startswith("-"):
                break
    return sorted(symbols)


def teacher_elements(model_dir=None) -> list:
    """Elements the teacher model covers (= the elements LoRA adapters are trained for)."""
    return model_elements(model_dir or TEACHER_MODEL_DIR)


def student_elements(model_dir=None) -> list:
    return model_elements(model_dir or STUDENT_MODEL_DIR)


def setup_device(use_gpu: bool = USE_GPU):
    """Hide GPUs unless requested. Call before any model is loaded."""
    import tensorflow as tf

    if not use_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:  # devices already initialized
            print("WARNING: could not hide GPUs; call setup_device() before loading models")
    devices = tf.config.get_visible_devices()
    print(f"Devices: {[d.device_type + ':' + d.name.split(':')[-1] for d in devices]}")


def require(value: str, name: str) -> Path:
    """Return `value` as an expanded Path; exit with a clear message if unset or missing.
    Output folders (DATA_DIR, RUNS_DIR, EXPERIMENTS_DIR, ...) are not checked for existence."""
    if not value:
        raise SystemExit(
            f"Path `{name}` is not set. Fill it in scripts/config.py or pass it as a flag."
        )
    path = Path(value).expanduser()
    if name not in _OUTPUT_KEYS and not path.exists():
        hint = {
            "STUDENT_CHECKPOINT_DIR": "run: python3 pipeline.py fetch",
            "STUDENT_MODEL_DIR": "run: python3 pipeline.py fetch",
            "TEACHER_MODEL_DIR": "run: python3 pipeline.py fetch",
            "TRAIN_POOL_XYZ": "run: python3 pipeline.py fetch",
            "TEST_XYZ": "run: python3 pipeline.py fetch",
        }.get(name, "run: python3 pipeline.py prepare (or check the path in scripts/config.py)")
        raise SystemExit(f"`{name}` not found: {path}\n  -> {hint}")
    return path


# --- Datasets (pickled pandas DataFrames of ase.Atoms) ------------------------------------------
# Unpickling runs code, so a .pkl.gz must only come from a trusted source. As an integrity check,
# every dataset we write gets a "<file>.sha256" sidecar, and load_frame() refuses pickles whose
# sidecar is missing or does not match. (A sidecar proves the file is unchanged since it was
# recorded, not where it came from: anyone can supply both files.) A pickle you produced elsewhere can be accepted explicitly with
#   python scripts/config.py trust FILE...
# (or, for one run, GRACE_LORA_TRUST_PICKLES=1).
TRUST_PICKLES_ENV = "GRACE_LORA_TRUST_PICKLES"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def trust_file(path) -> Path:
    """Record the current content of `path` as produced by this pipeline (writes the sidecar)."""
    path = Path(path).expanduser()
    _sidecar(path).write_text(_sha256(path) + "\n")
    return path


def save_frame(df, path) -> Path:
    """df.to_pickle(path) plus its checksum sidecar."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return trust_file(path)


def load_frame(path):
    """pd.read_pickle(path), only for pickles with a matching checksum sidecar (integrity check)."""
    import pandas as pd

    path = Path(path).expanduser()
    if os.environ.get(TRUST_PICKLES_ENV, "").strip() not in ("", "0"):
        return pd.read_pickle(path)
    side = _sidecar(path)
    if not side.exists():
        raise SystemExit(
            f"Refusing to load {path}: no checksum sidecar ({side.name}), so it has not been recorded "
            f"as a trusted dataset. Unpickling runs code; only load files from trusted sources.\n"
            f"  If you created it yourself: python scripts/config.py trust {path}")
    if side.read_text().split()[0] != _sha256(path):
        raise SystemExit(
            f"Refusing to load {path}: its content changed since it was written (checksum mismatch). "
            f"Rebuild it, or if you changed it yourself: python scripts/config.py trust {path}")
    return pd.read_pickle(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "trust":
        for f in sys.argv[2:]:
            print(f"trusted: {trust_file(f)}")
    else:
        print("usage: python scripts/config.py trust FILE.pkl.gz...")
