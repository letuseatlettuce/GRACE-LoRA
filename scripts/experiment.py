"""Run ONE LoRA experiment of the autonomous loop and record it in the ledger.

Steps (docs/CONTRACTS.md):
  1. allocate an id (exp_001, ...) and write experiments/<id>/experiment.json
  2. build the training mix with make_replay_dataset.py (organics + replay materials)
  3. train with train_lora.py (replay flags, the 3 fixed monitors from data/monitors/)
  4. read history.csv: selected epoch = lowest validation force MAE (the exported model),
     base reference = epoch 0
  5. append the ledger entry; print `RESULT:{json}` as the last line

Stdlib only: the heavy work runs in subprocesses with --python (the venv interpreter).

    python scripts/experiment.py --workdir W --config '{"n_materials": 600, "rank": 8}' \\
        --rationale "replay baseline" [--parent exp_001] [--python .venv/bin/python] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger as L  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
DEVICE_FLAGS = {"cpu-jit": ["--jit"], "cpu": [], "gpu": ["--gpu"]}
DEFAULT_TEACHER_ELEMENTS = ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"]  # GRACE-OFF
DEFAULT_S_PER_STRUCTURE_EPOCH = 0.2  # measured on a laptop CPU with --jit, batch size 8
LOG_TAIL_LINES = 40
NOISE = re.compile(r"cuda_executor|computation_placer|absl::|oneDNN|cpu_feature_guard|"
                   r"rebuild TensorFlow|XLA service|StreamExecutor device|Compiled cluster|"
                   r"dlopen|gpu_device\.cc|Skipping registering|swapaxes|return bound|"
                   r"squeeze_dims|Instructions for updating|axis. argument instead|it/s\]|s/it\]")


# --------------------------------------------------------------------------------------
# config + settings
# --------------------------------------------------------------------------------------
def parse_config(text: str | None) -> dict:
    """JSON string or @file -> full config with defaults; unknown keys are an error."""
    raw = {}
    if text:
        if text.startswith("@"):
            text = Path(text[1:]).expanduser().read_text()
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise SystemExit("--config must be a JSON object")
    unknown = sorted(set(raw) - set(L.DEFAULT_CONFIG))
    if unknown:
        raise SystemExit(f"Unknown config key(s) {unknown}. Valid keys: {sorted(L.DEFAULT_CONFIG)}")
    config = {**L.DEFAULT_CONFIG, **raw}
    for key in ("n_organic", "n_materials", "n_materials_val", "n_only_adapted", "rank",
                "epochs", "batch_size", "max_buckets", "max_atoms", "seed"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 0:
            raise SystemExit(f"config '{key}' must be a non-negative integer, got {config[key]!r}")
    for key in ("n_organic", "rank", "epochs", "batch_size", "max_buckets", "max_atoms"):
        if config[key] == 0:
            raise SystemExit(f"config '{key}' must be > 0")
    if config["device"] not in DEVICE_FLAGS:
        raise SystemExit(f"config 'device' must be one of {sorted(DEVICE_FLAGS)}")
    if config["n_materials"] == 0:  # organic-only run (no replay)
        config["n_materials_val"] = 0
        config["n_only_adapted"] = 0
    if config["n_only_adapted"] > config["n_materials"]:
        raise SystemExit("config 'n_only_adapted' cannot exceed 'n_materials'")
    if config["adapt_elements"] is not None and not isinstance(config["adapt_elements"], list):
        raise SystemExit("config 'adapt_elements' must be a list of element symbols or null")
    return config


def load_settings(workdir: Path) -> dict:
    """pipeline.json (repo root first, then the workdir); {} if absent."""
    for path in (REPO / "pipeline.json", workdir / "pipeline.json"):
        if path.exists():
            return json.loads(path.read_text())
    return {}


def read_metadata_elements(model_dir) -> list | None:
    """chemical_symbols from a SavedModel's metadata.yaml, parsed without a YAML library."""
    meta = Path(model_dir).expanduser() / "metadata.yaml"
    if not meta.exists():
        return None
    symbols, inside = [], False
    for line in meta.read_text().splitlines():
        if line.startswith("chemical_symbols:"):
            inside = True
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("["):  # flow style: [Br, C, ...]
                return [s.strip().strip("'\"") for s in inline.strip("[]").split(",") if s.strip()]
            continue
        if inside:
            m = re.match(r"\s*-\s*(\S+)", line)
            if m:
                symbols.append(m.group(1).strip("'\""))
            elif line.strip():
                break
    return symbols or None


def teacher_elements(settings: dict) -> list:
    path = (settings.get("paths") or {}).get("TEACHER_MODEL_DIR")
    return (read_metadata_elements(path) if path else None) or list(DEFAULT_TEACHER_ELEMENTS)


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------
def build_commands(workdir: Path, exp_dir: Path, config: dict, settings: dict, python: str,
                   adapt_elements: list) -> tuple[list, list]:
    """(make_replay_dataset command, train_lora command) for this experiment."""
    data_dir = workdir / "data"
    materials = [str(data_dir / "materials" / f) for f in ("test.parquet", "train0.parquet")]
    paths = settings.get("paths") or {}
    data_cmd = [python, str(SCRIPTS / "make_replay_dataset.py"),
                "--organic-dir", str(data_dir / "organic"),
                "--materials", *materials,
                "--exclude", str(data_dir / "materials" / "val.parquet"),
                "--n-organic", str(config["n_organic"]),
                "--n-materials", str(config["n_materials"]),
                "--n-materials-val", str(config["n_materials_val"]),
                "--n-only-adapted", str(config["n_only_adapted"]),
                "--max-atoms", str(config["max_atoms"]),
                "--oversample", str(config["oversample"]),
                "--seed", str(config["seed"]),
                "--out", str(exp_dir / "data"),
                "--no-monitors"]
    energy_reference = data_dir / "monitors" / "energy_reference.json"
    if energy_reference.exists():
        # same per-element offsets as the organic monitor: no base inference per experiment
        data_cmd += ["--energy-reference", str(energy_reference)]
    if paths.get("TEACHER_MODEL_DIR"):
        data_cmd += ["--adapted-from", paths["TEACHER_MODEL_DIR"]]
    if paths.get("STUDENT_MODEL_DIR"):
        data_cmd += ["--base", paths["STUDENT_MODEL_DIR"]]

    train_cmd = [python, str(SCRIPTS / "train_lora.py"),
                 "--data-dir", str(exp_dir / "data"), "--runs-dir", str(exp_dir), "--name", "run",
                 "--mode", "lora", "--rank", str(config["rank"]), "--epochs", str(config["epochs"]),
                 "--batch-size", str(config["batch_size"]),
                 "--max-buckets", str(config["max_buckets"]), "--seed", str(config["seed"]),
                 *DEVICE_FLAGS[config["device"]],
                 "--no-energy-shift", "--no-train-shift",
                 "--adapt-elements", *adapt_elements]
    if config["alpha"] is not None:
        train_cmd += ["--alpha", str(config["alpha"])]
    if config["lr"] is not None:
        train_cmd += ["--lr", str(config["lr"])]
    if paths.get("STUDENT_CHECKPOINT_DIR"):
        train_cmd += ["--checkpoint-dir", paths["STUDENT_CHECKPOINT_DIR"]]
    monitors = [f"{name}={data_dir / 'monitors' / f'{name}.pkl.gz'}" for name in L.MONITORS]
    train_cmd += ["--monitor", *monitors]
    return data_cmd, train_cmd


def run_streamed(cmd: list, log_path: Path) -> int:
    """Run a command, stream output (minus TensorFlow noise) to the console, log everything."""
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONUNBUFFERED="1")
    with open(log_path, "a") as log:
        log.write(f"\n$ {shlex.join(cmd)}\n")
        log.flush()
        print(f"\n$ {shlex.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            if not NOISE.search(line):
                print(line, end="", flush=True)
        return proc.wait()


def log_tail(log_path: Path, n: int = LOG_TAIL_LINES) -> str:
    if not log_path.exists():
        return ""
    lines = [l for l in log_path.read_text(errors="replace").splitlines() if not NOISE.search(l)]
    return "\n".join(lines[-n:])


# --------------------------------------------------------------------------------------
# history.csv -> metrics
# --------------------------------------------------------------------------------------
def _num(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return None if value != value else value  # NaN -> None


def parse_history(path) -> tuple[dict, dict, int]:
    """(metrics of the selected epoch, base monitor values at epoch 0, selected epoch).

    Selected epoch = lowest validation force MAE over all rows (first occurrence), which is
    the state train_lora.py restores and exports."""
    with open(path) as fh:
        rows = [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"{path} has no rows")
    selected = min(rows, key=lambda r: (r["f_mae_meV_A"] if r["f_mae_meV_A"] is not None
                                        else float("inf")))
    epoch0 = next((r for r in rows if r["epoch"] == 0), rows[0])

    def monitor_values(row, kind):
        return {name: row.get(f"{name}_{kind}_mae_meV_{'A' if kind == 'f' else 'atom'}")
                for name in L.MONITORS}

    metrics = {f"{name}_f": v for name, v in monitor_values(selected, "f").items()}
    metrics.update({f"{name}_e": v for name, v in monitor_values(selected, "e").items()})
    metrics["val_f"] = selected["f_mae_meV_A"]
    metrics["val_e"] = selected["e_mae_meV_atom"]
    base = {f"{name}_f": v for name, v in monitor_values(epoch0, "f").items() if v is not None}
    return metrics, base, int(selected["epoch"])


# --------------------------------------------------------------------------------------
# ledger housekeeping
# --------------------------------------------------------------------------------------
def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def mark_stale(workdir: Path) -> list:
    """Entries left 'running' by a process that no longer exists -> 'interrupted'."""
    ledger = L.load(workdir)
    stale = [e["id"] for e in ledger["experiments"]
             if e.get("status") == "running"
             and not (e.get("host") == socket.gethostname() and _pid_alive(e.get("pid")))]
    for e in ledger["experiments"]:
        if e["id"] in stale:
            e["status"] = "interrupted"
            e["error"] = e.get("error") or "process ended without finishing (interrupted)"
    if stale:
        L.save(workdir, ledger)
    return stale


def estimate_seconds(config: dict, settings: dict) -> float:
    bench = settings.get("benchmark") or {}
    per = bench.get("train_s_per_structure_epoch") or DEFAULT_S_PER_STRUCTURE_EPOCH
    n_train = config["n_organic"] + config["n_materials"]
    overhead = 300 + 0.1 * config["n_organic"]  # data build (base energies) + compile + export
    return n_train * config["epochs"] * per + overhead


# --------------------------------------------------------------------------------------
# self-healing (optional, --self-heal)
# --------------------------------------------------------------------------------------
FAILURE_PATTERNS = [  # checked in order; first match wins
    ("missing_data", re.compile(r"Prepared data missing|run `?python3? pipeline\.py prepare|"
                                r"(No such file or directory|FileNotFoundError|not found)[^\n]*"
                                r"(monitors|organic|materials|\.pkl\.gz|\.parquet)", re.I)),
    ("oom", re.compile(r"out of memory|\bOOM\b|ResourceExhausted|MemoryError|\bKilled\b|"
                       r"std::bad_alloc|Cannot allocate memory", re.I)),
    ("device", re.compile(r"could not find registered platform|\bMetal\b|PluggableDevice|"
                          r"no (registered|supported) .*(XLA|device|kernel)|XLA[^\n]*(device|platform)|"
                          r"CUDA_ERROR|cudaError|failed to (create|initiali[sz]e)[^\n]*(device|GPU)|"
                          r"InternalError:[^\n]*GPU", re.I)),
    ("nan", re.compile(r"loss[^\n]*\b(nan|inf)\b|\bNaN\b|non-finite|contains (NaN|Inf)|"
                       r"Inf(inity)? (loss|in )", re.I)),
]
XLA_RELATED = re.compile(r"XLA|registered platform|jit", re.I)


def classify_failure(log_text: str, returncode=None) -> str:
    """missing_data | oom | device | nan | unknown, from a log tail and the exit code."""
    for name, pattern in FAILURE_PATTERNS:
        if pattern.search(log_text or ""):
            return name
    if returncode in (137, -9):  # SIGKILL: the kernel's out-of-memory killer
        return "oom"
    return "unknown"


def heal_config(config: dict, error_class: str, log_text: str = ""):
    """(new config, {key: [old, new]}) for a retry, or None when retrying cannot help."""
    new, change = dict(config), {}

    def set_(key, value):
        if new[key] != value:
            change[key] = [new[key], value]
            new[key] = value

    if error_class == "oom":
        set_("batch_size", max(2, config["batch_size"] // 2))
        set_("max_buckets", max(2, config["max_buckets"] - 1))
    elif error_class == "device":
        if config["device"] == "gpu":
            set_("device", "cpu-jit")
        elif config["device"] == "cpu-jit" and XLA_RELATED.search(log_text or ""):
            set_("device", "cpu")
    elif error_class == "nan":
        set_("lr", (config["lr"] if config["lr"] is not None else 2e-3) / 2)
    return (new, change) if change else None


# --------------------------------------------------------------------------------------
# predictions (optional, --hypothesis / --predict)
# --------------------------------------------------------------------------------------
PREDICT_KEYS = ("organic_f", "materials_f", "only_adapted_f")


def parse_predictions(items) -> dict:
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if key not in PREDICT_KEYS:
            raise SystemExit(f"--predict key must be one of {PREDICT_KEYS}, got '{key}'")
        try:
            out[key] = float(value)
        except ValueError:
            raise SystemExit(f"--predict {item}: value must be a number")
    return out


# --------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True, help="pipeline workdir (data/, experiments/)")
    p.add_argument("--config", default=None, help="JSON object (or @file) of config overrides; "
                   f"keys: {', '.join(L.DEFAULT_CONFIG)}")
    p.add_argument("--rationale", required=True, help="why this experiment (decision log)")
    p.add_argument("--parent", default=None, help="experiment this one builds on")
    p.add_argument("--python", default=sys.executable, help="interpreter for the subprocesses")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    p.add_argument("--self-heal", action="store_true",
                   help="retry a failed training with an adjusted config (OOM -> smaller batches, "
                        "GPU/XLA error -> CPU, NaN loss -> lower lr)")
    p.add_argument("--max-retries", type=int, default=2, help="retries with --self-heal")
    p.add_argument("--hypothesis", default=None, help="what you expect and why (stored)")
    p.add_argument("--predict", action="append", default=[], metavar="KEY=VALUE",
                   help=f"predicted monitor value; KEY in {', '.join(PREDICT_KEYS)} (repeatable)")
    args = p.parse_args(argv)

    workdir = Path(args.workdir).expanduser().resolve()
    config = parse_config(args.config)
    prediction = parse_predictions(args.predict)
    settings = load_settings(workdir)
    adapt = config["adapt_elements"] or teacher_elements(settings)
    ledger = L.load(workdir)
    if args.parent and L.get(ledger, args.parent) is None:
        raise SystemExit(f"--parent {args.parent} is not in the ledger")
    exp_id = L.next_id(ledger)
    exp_dir = workdir / "experiments" / exp_id
    data_cmd, train_cmd = build_commands(workdir, exp_dir, config, settings, args.python, adapt)
    est = estimate_seconds(config, settings)

    if args.dry_run:
        print(f"[dry-run] experiment {exp_id} in {exp_dir}")
        print(f"[dry-run] config: {json.dumps(config)}")
        print(f"[dry-run] estimated time: {est / 60:.0f} min")
        if args.self_heal:
            print(f"[dry-run] self-heal on: up to {args.max_retries} retries")
        print(shlex.join(data_cmd))
        print(shlex.join(train_cmd))
        return 0

    missing = [str(workdir / "data" / "monitors" / f"{n}.pkl.gz") for n in L.MONITORS
               if not (workdir / "data" / "monitors" / f"{n}.pkl.gz").exists()]
    if missing or not (workdir / "data" / "organic" / "train.pkl.gz").exists():
        raise SystemExit("Prepared data missing (" + ", ".join(missing or ["data/organic"])
                         + "). Run `python3 pipeline.py prepare` first.")

    L.sync_objective(workdir)  # objective limits from pipeline.json, if configured
    stale = mark_stale(workdir)
    if stale:
        print(f"Marked interrupted: {', '.join(stale)}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "experiment.log"
    started = time.time()
    entry = {"id": exp_id, "parent": args.parent, "time": time.strftime("%Y-%m-%d %H:%M"),
             "rationale": args.rationale, "config": config, "adapt_elements": adapt,
             "status": "running", "run_dir": f"experiments/{exp_id}/run", "duration_s": None,
             "selected_epoch": None, "metrics": None, "relative": None, "feasible": None,
             "score": None, "error": None, "pid": os.getpid(), "host": socket.gethostname(),
             "estimated_s": round(est)}
    if args.hypothesis:
        entry["hypothesis"] = args.hypothesis
    if prediction:
        entry["prediction"] = prediction
    if args.self_heal:
        entry["healing"] = []
    (exp_dir / "experiment.json").write_text(json.dumps(
        {k: entry.get(k) for k in ("id", "parent", "time", "rationale", "config",
                                   "adapt_elements", "hypothesis", "prediction")}, indent=2))
    L.append(workdir, entry)
    print(f"== {exp_id}: {args.rationale}")
    print(f"   config: {L.config_deltas(config)}  |  adapted elements: {' '.join(adapt)}")
    print(f"   estimated time: ~{est / 60:.0f} min "
          f"({config['n_organic'] + config['n_materials']} structures x {config['epochs']} epochs)")
    if args.hypothesis:
        print(f"   hypothesis: {args.hypothesis}")
    if prediction:
        print("   prediction: " + ", ".join(f"{k}={v:g}" for k, v in prediction.items()))

    healing, code = [], 1
    try:
        code = run_streamed(data_cmd, log_path)
        if code != 0:
            raise RuntimeError(f"data step failed (exit {code}): "
                               f"{classify_failure(log_tail(log_path), code)}")
        attempts = 1 + (max(0, args.max_retries) if args.self_heal else 0)
        for attempt in range(1, attempts + 1):
            offset = log_path.stat().st_size
            code = run_streamed(train_cmd, log_path)
            if code == 0:
                break
            tail = log_path.read_text(errors="replace")[offset:][-6000:]
            error_class = classify_failure(tail, code)
            healed = heal_config(config, error_class, tail) if attempt < attempts else None
            if not healed:
                hint = (" -> run `python3 pipeline.py prepare`" if error_class == "missing_data"
                        else "")
                raise RuntimeError(f"training failed (exit {code}, {error_class}){hint}")
            config, change = healed
            healing.append({"attempt": attempt, "error_class": error_class, "change": change,
                            "time": time.strftime("%Y-%m-%d %H:%M")})
            L.update_entry(workdir, exp_id, config=config, healing=healing)
            print(f"\n== self-heal: attempt {attempt} failed ({error_class}); retrying with "
                  + ", ".join(f"{k} {o}->{n}" for k, (o, n) in change.items()), flush=True)
            _, train_cmd = build_commands(workdir, exp_dir, config, settings, args.python, adapt)
        history = exp_dir / "run" / "history.csv"
        if not history.exists():
            raise RuntimeError(f"training finished but {history} is missing")
        metrics, base, selected = parse_history(history)
        ledger = L.load(workdir)
        if not ledger.get("base"):
            ledger["base"] = base
        entry = L.get(ledger, exp_id)
        entry.update(status="done", metrics=metrics, selected_epoch=selected, config=config,
                     duration_s=round(time.time() - started))
        if args.self_heal:
            entry["healing"] = healing
        L.evaluate_entry(entry, ledger)
        if prediction:
            entry["prediction_check"], entry["prediction_score"] = \
                L.prediction_check(prediction, metrics)
        L.save(workdir, ledger)
        code = 0
    except (RuntimeError, ValueError, OSError, KeyboardInterrupt) as exc:
        reason = "interrupted by user" if isinstance(exc, KeyboardInterrupt) else str(exc)
        fields = dict(status="failed", duration_s=round(time.time() - started), config=config,
                      error=f"{reason}\n--- log tail ---\n{log_tail(log_path)}")
        if args.self_heal:
            fields["healing"] = healing
        ledger = L.update_entry(workdir, exp_id, **fields)
        entry = L.get(ledger, exp_id)
        code = 1

    entry = {k: v for k, v in entry.items() if k not in ("pid", "host")}
    if entry["status"] == "done":
        m, r = entry["metrics"], entry["relative"]
        print(f"\n== {exp_id} done in {entry['duration_s'] / 60:.0f} min (epoch {entry['selected_epoch']}): "
              f"organic {m['organic_f']:.1f}, materials {m['materials_f']:.1f} "
              f"({100 * r.get('materials_increase', 0):+.0f}%), only_adapted "
              f"{m['only_adapted_f']:.1f} ({100 * r.get('only_adapted_increase', 0):+.0f}%), "
              f"feasible={entry['feasible']}")
        if healing:
            print(f"   self-healed after {len(healing)} failed attempt(s)")
        if entry.get("prediction_check"):
            print("   prediction vs actual: " + ", ".join(
                f"{k} {c['predicted']:g} vs {c['actual']:.1f} ({100 * c['rel_error']:+.0f}%)"
                for k, c in entry["prediction_check"].items()))
    else:
        print(f"\n== {exp_id} FAILED: {entry['error'].splitlines()[0]}  (log: {log_path})")
    print("RESULT:" + json.dumps(entry))
    return code


if __name__ == "__main__":
    sys.exit(main())
