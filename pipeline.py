#!/usr/bin/env python3
"""GRACE -> teacher LoRA distillation pipeline: one entry point from bare machine to report.

Standard library only, so it runs before the project venv exists. Heavy steps run the
scripts in scripts/ with the venv's Python. Choices are stored in pipeline.json, which
scripts/config.py reads, so no paths are ever edited by hand.

Just press Run (or `python3 pipeline.py`): everything below happens automatically, using the
choices in the YOUR CHOICES block right under this text. Finished steps are skipped, so
pressing Run again continues where it stopped. The only question is a yes/no before a
multi-hour run (set AUTO_RUN = True to skip it).

Individual steps (optional; the automatic mode runs them for you):
  sysinfo      detect OS, CPU, GPU, RAM, disk, Python versions; print recommendations
  setup-env    create .venv (Python 3.10-3.12) and install requirements
  catalog      list available student models, teacher models and datasets
  configure    choose student, teacher, data, device  -> pipeline.json
  fetch        download the chosen models and data (or check user-supplied paths)
  validate     check files, library versions and element coverage
  smoke        smoke test + timing/memory benchmark (--race: also time the GPU)
  estimate     time estimates for each plan, from the benchmark
  run          dataset -> training jobs -> evaluation -> report.md (resumable, fixed plans)
  prepare      organic pool (teacher labels) + MPtrj materials shards + monitor sets
  experiment   ONE LoRA experiment (--set key=value ...) -> experiments/ledger.json
  ledger       show the experiment ledger (--best: the best experiment)
  validate-exp organic + materials validation of an experiment (or of the base reference)
  validate-models  validate ANY models (SavedModel / run / adapter / checkpoint), no loop needed
  report       final report.md + figures from the ledger and validations
  settings     show ALL settings (training, quality, usage) with defaults; --set/--reset,
               --explain KEY, --confirm (auto and wizard show this panel and ask first)
  options      show / toggle optional agent features (subagents master switch +
               max_subagents cap, critic, self_heal, cockpit, cockpit_publish,
               predictions; all OFF by
               default; stored in pipeline.json "agent_options")
  cockpit      live HTML dashboard of the ledger and monitors (--watch SECONDS;
               --artifact: a version to publish as a Claude artifact)
  auto         sysinfo .. smoke (incl. prepare), then the chosen plan (the default)
  wizard       all of the above, asking every choice interactively

Examples:
  python3 pipeline.py              (= auto; also what "Run" in VS Code does)
  python3 pipeline.py wizard
  python3 pipeline.py configure --student GRACE-2L-OMAT-medium-ft-AM --teacher GRACE-OFF-2L-medium --data spice --yes
  python3 pipeline.py run --plan quick --yes
  python3 pipeline.py prepare
  python3 pipeline.py experiment --set n_materials=600 --set rank=8 --rationale "replay baseline"
  python3 pipeline.py validate-exp --exp best && python3 pipeline.py report
"""

from __future__ import annotations

# ======================================================================================
# YOUR CHOICES: edit if you want something else, then press Run. Nothing to type.
# ======================================================================================
STUDENT = "GRACE-2L-OMAT-medium-ft-AM"  # GRACE model name, or a local checkpoint folder
STUDENT_SAVED_MODEL = ""     # only with a local checkpoint folder: its SavedModel folder
TEACHER = "GRACE-OFF-2L-medium"  # GRACE-OFF-{1L,2L}-{small,medium,large}, "grace:<name>",
                                 # or a GRACE SavedModel folder (saved_model, not casted_model)
DATA = "spice"               # "spice" (downloaded automatically) or the path to your .xyz file
TEST_DATA = ""               # optional separate test .xyz for your own data
PLAN = "quick"               # pilot (~500 molecules) | quick (~1000) | solid (~2000, overnight)
AUTO_RUN = False             # True = start the long run without asking
RACE_GPU = True              # if a GPU exists: time it against the CPU in the smoke test
WORKDIR = ""                 # where data/ and runs/ go ("" = this folder)
# ======================================================================================

import argparse
import csv
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import checksums  # noqa: E402  (stdlib only)

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "pipeline.json"
STATE = ROOT / ".pipeline"
GRACE_CACHE = Path(os.environ.get("GRACE_CACHE") or "~/.cache/grace").expanduser()

DEFAULT_STUDENT = "GRACE-2L-OMAT-medium-ft-AM"
DEFAULT_TEACHER = "GRACE-OFF-2L-medium"
GRACE_OFF_REPO = "https://github.com/heid-lab/grace-off.git"
GRACE_OFF = {f"GRACE-OFF-{layers.upper()}-{size}": f"models/{layers}/b_off_{size}"
             for layers in ("1l", "2l") for size in ("small", "medium", "large")}
ZENODO_URL = "https://zenodo.org/api/records/18662355/files/{}/content"
SPICE_FILES = {"TRAIN_POOL_XYZ": "biased_test_set.xyz", "TEST_XYZ": "GRACE_test_set.xyz"}
SUPPORTED_PYTHON = ((3, 10), (3, 11), (3, 12))  # TensorFlow 2.19 wheels

PLANS = {
    "pilot": {"n_train": 500, "n_val": 100, "n_test": 300, "epochs": 10,
              "jobs": [{"name": "lora_r8", "mode": "lora", "rank": 8}]},
    "quick": {"n_train": 1000, "n_val": 200, "n_test": 1000, "epochs": 15,
              "jobs": [{"name": "lora_r8", "mode": "lora", "rank": 8},
                       {"name": "full_baseline", "mode": "full"}]},
    "solid": {"n_train": 2000, "n_val": 200, "n_test": 1000, "epochs": 30,
              "jobs": [{"name": f"lora_r{r}", "mode": "lora", "rank": r} for r in (8, 4, 16)]
                      + [{"name": "full_baseline", "mode": "full"}]},  # first job = main result
}
SMOKE = {"n_train": 96, "n_val": 24, "n_test": 40, "epochs": 3, "rank": 4}
DEVICE_FLAGS = {"cpu-jit": ["--jit"], "cpu": [], "gpu": ["--gpu"]}
NOISE = re.compile(r"cuda_executor|computation_placer|absl::|oneDNN|cpu_feature_guard|"
                   r"rebuild TensorFlow|XLA service|StreamExecutor device|Compiled cluster|"
                   r"dlopen|gpu_device\.cc|Skipping registering|swapaxes|return bound|"
                   r"squeeze_dims|Instructions for updating|axis. argument instead")


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def load_settings() -> dict:
    return json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}


def save_settings(settings: dict):
    SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")


def venv_dir(settings: dict) -> Path:
    return Path(os.environ.get("GRACE_PIPELINE_VENV") or settings.get("venv")
                or ROOT / ".venv").expanduser()


def vbin(settings: dict, name: str) -> str:
    return str(venv_dir(settings) / "bin" / name)


def capture(cmd, timeout=60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def keep_awake_prefix() -> list:
    """macOS: keep the machine awake while a long command runs."""
    if platform.system() == "Darwin" and shutil.which("caffeinate"):
        return ["caffeinate", "-i"]
    return []


def sh(cmd, log: Path | None = None, check=True, quiet_noise=True) -> str:
    """Run a command, stream its output (minus TensorFlow log noise), return all output."""
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONUNBUFFERED="1")
    lines = []
    logf = open(log, "a") if log else None
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            lines.append(line)
            if logf:
                logf.write(line)
                logf.flush()
            if not (quiet_noise and NOISE.search(line)):
                print(line, end="", flush=True)
        proc.wait()
    finally:
        if logf:
            logf.close()
    if check and proc.returncode != 0:
        raise SystemExit(f"\nFAILED (exit {proc.returncode}): {' '.join(cmd)}"
                         + (f"\nFull log: {log}" if log else ""))
    return "".join(lines)


def ask(question: str, default: str = "", choices=None) -> str:
    hint = f" [{default}]" if default else ""
    while True:
        answer = input(f"{question}{hint}: ").strip() or default
        if not choices or answer in choices:
            return answer
        print(f"  choose one of: {', '.join(choices)}")


def confirm(question: str, default=True, assume_yes=False) -> bool:
    if assume_yes:
        return True
    answer = input(f"{question} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not answer else answer.startswith("y")


def inspect(settings: dict, *args) -> dict:
    """Run scripts/inspect_setup.py in the venv and parse its JSON line."""
    out = capture([vbin(settings, "python"), "scripts/inspect_setup.py", *args], timeout=600)
    for line in out.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    raise SystemExit(f"inspect_setup.py {' '.join(args)} returned no result:\n{out[-2000:]}")


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{max(m, 1)} min"


# --------------------------------------------------------------------------------------
# sysinfo
# --------------------------------------------------------------------------------------
def python_version(path: str):
    out = capture([path, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"])
    return tuple(int(x) for x in out.split(".")) if re.fullmatch(r"\d+\.\d+\.\d+", out) else None


def find_pythons() -> dict:
    candidates = [shutil.which(n) for n in ("python3.11", "python3.12", "python3.10", "python3")]
    candidates += ["/opt/homebrew/bin/python3.11", "/usr/local/bin/python3.11",
                   *map(str, Path("~/.pyenv/versions").expanduser().glob("3.1[0-2]*/bin/python"))]
    found = {}
    for path in filter(None, candidates):
        if not Path(path).exists():
            continue
        real = os.path.realpath(path)
        version = python_version(path)
        if version and real not in {v["real"] for v in found.values()}:
            found[path] = {"version": ".".join(map(str, version)), "real": real,
                           "supported": version[:2] in SUPPORTED_PYTHON}
    return found


def detect_system() -> dict:
    system = platform.system()
    info = {
        "os": system, "arch": platform.machine(), "os_version": platform.release(),
        "wsl": "microsoft" in platform.release().lower(),
        "cpu": platform.processor() or platform.machine(), "cpu_threads": os.cpu_count(),
        "cpu_performance_cores": None, "ram_gb": None, "gpus": [],
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1),
    }
    if system == "Darwin":
        info["os_version"] = "macOS " + capture(["sw_vers", "-productVersion"])
        info["cpu"] = capture(["sysctl", "-n", "machdep.cpu.brand_string"]) or info["cpu"]
        info["ram_gb"] = round(int(capture(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1)
        perf = capture(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        info["cpu_performance_cores"] = int(perf) if perf.isdigit() else None
        try:
            displays = json.loads(capture(["system_profiler", "SPDisplaysDataType", "-json"]))
            for d in displays.get("SPDisplaysDataType", []):
                info["gpus"].append({"vendor": "apple" if "Apple" in d.get("sppci_model", "")
                                     else "other", "name": d.get("sppci_model", "?"),
                                     "cores": d.get("sppci_cores"), "memory_gb": "shared"})
        except (json.JSONDecodeError, TypeError):
            pass
    elif system == "Linux":
        try:
            distro = re.search(r'PRETTY_NAME="([^"]+)"', Path("/etc/os-release").read_text())
            info["os_version"] = distro.group(1) if distro else f"Linux {platform.release()}"
            cpuinfo = Path("/proc/cpuinfo").read_text()
            m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            info["cpu"] = m.group(1).strip() if m else info["cpu"]
            mem = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text())
            info["ram_gb"] = round(int(mem.group(1)) / 2**20, 1) if mem else None
        except OSError:
            pass
        smi = capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                       "--format=csv,noheader,nounits"])
        for row in filter(None, smi.splitlines()):
            name, memory, driver = [x.strip() for x in row.split(",")]
            info["gpus"].append({"vendor": "nvidia", "name": name,
                                 "memory_gb": round(float(memory) / 1024, 1), "driver": driver})
    info["pythons"] = find_pythons()
    return info


def recommend(info: dict) -> dict:
    blockers, notes = [], []
    if info["os"] == "Darwin" and info["arch"] != "arm64":
        blockers.append("Intel Macs are not supported: TensorFlow 2.19 ships Apple Silicon wheels only.")
    if info["os"] == "Windows":
        blockers.append("Run inside WSL2 (Ubuntu) on Windows; native Windows is not supported.")
    pythons = [p for p, v in info["pythons"].items() if v["supported"]]
    pythons.sort(key=lambda p: (info["pythons"][p]["version"].split(".")[1] != "11", p))
    if not pythons:
        blockers.append("No Python 3.10-3.12 found. Install Python 3.11 "
                        "(macOS: `brew install python@3.11`; Ubuntu: `sudo apt install python3.11-venv`).")
    if info["disk_free_gb"] < 8:
        blockers.append(f"Only {info['disk_free_gb']} GB free disk; need about 8 GB.")

    ram = info["ram_gb"] or 8
    batch_size, max_buckets = (8, 4) if ram < 24 else (16, 8)
    notes.append(f"Batch size {batch_size}, max {max_buckets} batch shapes "
                 f"(memory-safe for {ram} GB RAM; XLA compilation needs ~6 GB at batch size 8, "
                 ">14 GB at batch size 16).")
    if ram < 12:
        notes.append("Less than 12 GB RAM: keep other programs closed during training.")

    has_apple = any(g["vendor"] == "apple" for g in info["gpus"])
    has_nvidia = any(g["vendor"] == "nvidia" for g in info["gpus"])
    if has_apple:
        notes.append("Apple GPU detected: labelling/evaluation must use the CPU (XLA has no Metal "
                     "backend). Training may use Metal: run `smoke --race` to compare.")
    if has_nvidia:
        notes.append("NVIDIA GPU detected: CUDA TensorFlow will be installed; run `smoke --race` "
                     "to see whether the GPU beats CPU+XLA for training.")
        if any(re.search(r"RTX\s*50\d\d", g["name"]) for g in info["gpus"] if g["vendor"] == "nvidia"):
            notes.append("RTX 50xx (Blackwell) GPU: TensorFlow 2.19 support is uncertain; the tested "
                         "path is CPU+XLA (`setup-env --cpu-only`, device cpu-jit).")
    if info["wsl"]:
        notes.append("WSL: keep the project and outputs in the Linux file system (~), not under "
                     "/mnt/c or a OneDrive folder.")
    if info["os"] == "Darwin":
        notes.append("macOS: long runs are wrapped in `caffeinate -i` so the Mac does not sleep.")

    # which sections of references/hardware.md apply to this machine (known issues + fixes)
    sections = []
    if info["wsl"] or info["os"] == "Windows":
        sections.append("windows--wsl2")
    if has_nvidia:
        sections.append("nvidia-gpus-incl-rtx-50xx-blackwell")
    if info["os"] == "Darwin":
        sections.append("macos-apple-silicon-m1m4")
    if info["os"] == "Linux" and not has_nvidia and not info["wsl"]:
        sections.append("linux-cpu-only")
    sections.append("memory-sizing-all-platforms")
    hardware_doc = ".claude/skills/grace-lora/references/hardware.md"
    notes.append("Known issues and fixes for this machine: " + ", ".join(
        f"{hardware_doc}#{s}" for s in sections))
    return {"blockers": blockers, "notes": notes, "python": pythons[0] if pythons else None,
            "batch_size": batch_size, "max_buckets": max_buckets, "device": "cpu-jit",
            "cuda_install": has_nvidia, "hardware_notes": [f"{hardware_doc}#{s}" for s in sections]}


def print_sysinfo(info: dict, rec: dict):
    print("\n== System")
    print(f"  OS        {info['os_version']} ({info['arch']}){'  [WSL]' if info['wsl'] else ''}")
    cores = f"{info['cpu_threads']} threads"
    if info["cpu_performance_cores"]:
        cores += f", {info['cpu_performance_cores']} performance cores"
    print(f"  CPU       {info['cpu']} ({cores})")
    print(f"  RAM       {info['ram_gb']} GB")
    gpus = ", ".join(f"{g['name']}" + (f" ({g['cores']} cores)" if g.get("cores") else "")
                     + (f" {g['memory_gb']} GB" if isinstance(g.get("memory_gb"), float) else "")
                     for g in info["gpus"]) or "none detected"
    print(f"  GPU       {gpus}")
    print(f"  Disk free {info['disk_free_gb']} GB")
    for path, v in info["pythons"].items():
        print(f"  Python    {v['version']:8s} {path}{'' if v['supported'] else '  (unsupported)'}")
    print("\n== Recommendations")
    for b in rec["blockers"]:
        print(f"  BLOCKER: {b}")
    print(f"  Python for the venv: {rec['python']}")
    print(f"  Training device: {rec['device']} (CPU with XLA)")
    for n in rec["notes"]:
        print(f"  - {n}")


def cmd_sysinfo(args):
    info = detect_system()
    rec = recommend(info)
    print_sysinfo(info, rec)
    settings = load_settings()
    settings["system"] = info
    settings["recommendation"] = rec
    save_settings(settings)
    if args.json:
        print(json.dumps({"system": info, "recommendation": rec}, indent=2))
    return rec


# --------------------------------------------------------------------------------------
# setup-env
# --------------------------------------------------------------------------------------
def cmd_setup_env(args):
    settings = load_settings()
    info = settings.get("system") or detect_system()
    rec = settings.get("recommendation") or recommend(info)
    if rec["blockers"]:
        raise SystemExit("Fix these first:\n  " + "\n  ".join(rec["blockers"]))
    vdir = venv_dir(settings)
    vpy = vdir / "bin" / "python"
    if not vpy.exists():
        python = args.python or rec["python"]
        print(f"Creating venv {vdir} with {python}")
        sh([python, "-m", "venv", vdir])
    version = python_version(str(vpy))
    if not version or version[:2] not in SUPPORTED_PYTHON:
        raise SystemExit(f"{vdir} uses Python {version}; need 3.10-3.12. Remove it or use --venv.")

    requirements = ROOT / "requirements.txt"
    cpu_only = info["os"] == "Linux" and (args.cpu_only or not rec.get("cuda_install"))
    tp_pin = None
    if cpu_only:
        # No NVIDIA GPU: install the plain (CPU) TensorFlow wheel, not the ~4 GB CUDA bundle.
        # tensorpotential itself declares tensorflow[and-cuda] on Linux, so it is installed
        # with --no-deps and its remaining dependencies are listed explicitly.
        STATE.mkdir(exist_ok=True)
        lines = []
        for line in requirements.read_text().splitlines():
            if line.strip().startswith("tensorpotential"):
                tp_pin = line.split("#")[0].strip()
                continue
            lines.append(line.replace("tensorflow[and-cuda]==", "tensorflow=="))
        lines += ["# tensorpotential dependencies (installed with --no-deps below)",
                  "matscipy", "sympy", "pyyaml>=6.0.2", "tqdm", "requests", "questionary>=2.0",
                  "rich>=13.0", "scikit-learn>=1.1"]
        requirements = STATE / "requirements-cpu.txt"
        requirements.write_text("\n".join(lines) + "\n")
        print("Linux without NVIDIA GPU: installing CPU-only TensorFlow (no CUDA packages)")
    sh([vpy, "-m", "pip", "install", "--upgrade", "pip"])
    sh([vpy, "-m", "pip", "install", "-r", requirements])
    if tp_pin:
        sh([vpy, "-m", "pip", "install", "--no-deps", tp_pin])

    report = inspect(settings, "versions")
    print("\n== Installed versions")
    for pkg, ver in report["versions"].items():
        want = report["pinned"].get(pkg)
        status = "" if want is None else ("OK" if ver == want else f"MISMATCH (want {want})")
        print(f"  {pkg:18s} {str(ver):10s} {status}")
    conflicts = capture([str(vpy), "-m", "pip", "check"])
    if cpu_only:  # expected: tensorpotential's declared CUDA extra is deliberately not installed
        conflicts = "\n".join(line for line in conflicts.splitlines()
                              if not ("tensorpotential" in line and ("nvidia" in line or "and-cuda" in line)))
    print(f"\n== pip check\n  {conflicts or 'no broken requirements'}")
    if not report["ok"]:
        raise SystemExit("Pinned versions do not match; see table above.")
    if args.venv:
        settings["venv"] = str(vdir)
    save_settings(settings)


# --------------------------------------------------------------------------------------
# catalog + configure
# --------------------------------------------------------------------------------------
def student_catalog(settings: dict) -> list:
    if Path(vbin(settings, "python")).exists():
        try:
            return [s["name"] for s in inspect(settings, "catalog")["students"]]
        except SystemExit:
            pass
    return [DEFAULT_STUDENT, "GRACE-2L-OMAT-large-ft-AM", "GRACE-1L-OMAT-medium-ft-AM",
            "GRACE-2L-SMAX-OMAT-medium"]


def cmd_catalog(args):
    settings = load_settings()
    print("\n== Student models (GRACE foundation models with a training checkpoint)")
    for name in student_catalog(settings):
        print(f"  {name}{'   (default)' if name == DEFAULT_STUDENT else ''}")
    print("  <path>   a local checkpoint folder (model.yaml + checkpoint.index); "
          "also give --student-model <SavedModel dir>")
    print("\n== Teacher models")
    for name, sub in GRACE_OFF.items():
        print(f"  {name:22s} heid-lab/grace-off {sub}"
              f"{'   (default)' if name == DEFAULT_TEACHER else ''}")
    print("  grace:<name>           any GRACE foundation model above, e.g. grace:GRACE-2L-OMAT-large-ft-AM")
    print("  <path>                 a local GRACE SavedModel folder (fp64 `saved_model`, not `casted_model`)")
    print("\n== Data")
    print("  spice                  SPICE v2 sets from Zenodo 18662355 (training pool + held-out test)  (default)")
    print("  <pool.xyz>             your own extended-XYZ structures (+ --test <test.xyz>, optional;")
    print("                         without it the test set is held out from the pool)")


def resolve_student(spec: str, model_dir: str | None) -> dict:
    path = Path(spec).expanduser()
    if path.exists():
        if not (path / "model.yaml").exists() or not (path / "checkpoint.index").exists():
            raise SystemExit(f"{path} must contain model.yaml and checkpoint.index")
        if not model_dir:
            raise SystemExit("A local student checkpoint also needs --student-model <SavedModel dir> "
                             "(used as the untrained baseline in the evaluation)")
        return {"source": "local", "name": path.name, "checkpoint_dir": str(path),
                "model_dir": str(Path(model_dir).expanduser())}
    return {"source": "grace_models", "name": spec,
            "checkpoint_dir": str(GRACE_CACHE / "checkpoints" / spec),
            "model_dir": str(GRACE_CACHE / spec)}


def resolve_teacher(spec: str) -> dict:
    if spec in GRACE_OFF:
        return {"source": "grace-off", "name": spec, "subdir": GRACE_OFF[spec],
                "model_dir": str(ROOT / "models/grace-off" / GRACE_OFF[spec] / "seed/1/saved_model")}
    if spec.startswith("grace:"):
        return {"source": "grace_models", "name": spec[6:], "model_dir": str(GRACE_CACHE / spec[6:])}
    path = Path(spec).expanduser()
    if "casted_model" in str(path):
        raise SystemExit("Use the fp64 `saved_model` folder, not `casted_model` (it rejects fp64 inputs)")
    return {"source": "local", "name": path.name, "model_dir": str(path)}


def resolve_data(spec: str, test: str | None) -> dict:
    if spec == "spice":
        return {"source": "spice", **{k: str(ROOT / "data" / f) for k, f in SPICE_FILES.items()}}
    return {"source": "local", "TRAIN_POOL_XYZ": str(Path(spec).expanduser()),
            "TEST_XYZ": str(Path(test).expanduser()) if test else ""}


def cmd_configure(args):
    settings = load_settings()
    rec = settings.get("recommendation") or recommend(settings.get("system") or detect_system())
    interactive = not args.yes and sys.stdin.isatty()
    if interactive and not (args.student and args.teacher and args.data):
        cmd_catalog(args)
        print()
    student = args.student or (ask("Student model (name or checkpoint path)", DEFAULT_STUDENT)
                               if interactive else DEFAULT_STUDENT)
    student_model = args.student_model
    if interactive and Path(student).expanduser().exists() and not student_model:
        student_model = ask("SavedModel folder of the same student (baseline)")
    teacher = args.teacher or (ask("Teacher model (name, grace:<name> or SavedModel path)",
                                   DEFAULT_TEACHER) if interactive else DEFAULT_TEACHER)
    data = args.data or (ask("Data ('spice' or path to your .xyz pool)", "spice")
                         if interactive else "spice")
    test = args.test
    if interactive and data != "spice" and test is None:
        test = ask("Separate test .xyz (empty = hold out part of the pool)", "") or None
    device = args.device or settings.get("device") or rec["device"]
    workdir = Path(args.workdir or settings.get("workdir") or ROOT).expanduser().resolve()

    st, te, da = resolve_student(student, student_model), resolve_teacher(teacher), resolve_data(data, test)
    settings.update({
        "student": st, "teacher": te, "data": da, "device": device,
        "batch_size": args.batch_size or settings.get("batch_size") or rec["batch_size"],
        "max_buckets": args.max_buckets or settings.get("max_buckets") or rec["max_buckets"],
        "workdir": str(workdir),
        "paths": {
            "STUDENT_CHECKPOINT_DIR": st["checkpoint_dir"], "STUDENT_MODEL_DIR": st["model_dir"],
            "TEACHER_MODEL_DIR": te["model_dir"],
            "TRAIN_POOL_XYZ": da["TRAIN_POOL_XYZ"], "TEST_XYZ": da["TEST_XYZ"],
            "DATA_DIR": str(workdir / "data"), "RUNS_DIR": str(workdir / "runs"),
            **layout_paths(workdir),
        },
    })
    save_settings(settings)
    print(f"\nSaved {SETTINGS}:")
    print(f"  student  {st['name']} ({st['source']})")
    print(f"  teacher  {te['name']} ({te['source']})")
    print(f"  data     {da['source']}: {da['TRAIN_POOL_XYZ']}"
          + (f" | test {da['TEST_XYZ']}" if da["TEST_XYZ"] else " | test held out from pool"))
    print(f"  device   {device}, batch size {settings['batch_size']}, max buckets {settings['max_buckets']}")
    print(f"  outputs  {workdir}/data, {workdir}/experiments, {workdir}/validation, {workdir}/report")


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------
def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {dest.name}: {pct:3d}% of {total / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, progress)
    print()
    try:
        checksums.verify_file(tmp, checksums.expected_file(dest))
    except checksums.ChecksumError:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(dest)


def verify_downloads(paths) -> None:
    """Check files/model folders with pinned SHA-256s (scripts/checksums.py); others are skipped."""
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            known = checksums.verify_tree(p) if p.is_dir() else checksums.verify_file(p)
        except checksums.ChecksumError as exc:
            raise SystemExit(f"ERROR: {exc}")
        print(f"  {'checksum OK' if known else 'no pinned checksum (not verified)'}: {p}")


def fetch_grace_off(subdir: str):
    dest = ROOT / "models/grace-off"
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", GRACE_OFF_REPO, dest])
    current = capture(["git", "-C", str(dest), "sparse-checkout", "list"]).split()
    patterns = sorted({p for p in current if p.startswith("/models/")} | {f"/{subdir}/"})
    # --no-cone: cone mode would also pull ~80 MB of MACE models from the parent folder
    sh(["git", "-C", dest, "sparse-checkout", "set", "--no-cone", *patterns])


def require_configured(settings: dict):
    if "paths" not in settings:
        raise SystemExit("Not configured yet: run `python3 pipeline.py configure` first.")


def cmd_fetch(args):
    settings = load_settings()
    require_configured(settings)
    st, te, da = settings["student"], settings["teacher"], settings["data"]
    grace_models = vbin(settings, "grace_models")

    print(f"\n== Student {st['name']}")
    if st["source"] == "grace_models":
        sh([grace_models, "download", st["name"]])
        sh([grace_models, "checkpoint", st["name"]])
        verify_downloads([GRACE_CACHE / st["name"], GRACE_CACHE / "checkpoints" / st["name"]])
    print(f"\n== Teacher {te['name']}")
    if te["source"] == "grace-off":
        fetch_grace_off(te["subdir"])
        verify_downloads([settings["paths"]["TEACHER_MODEL_DIR"]])
    elif te["source"] == "grace_models":
        sh([grace_models, "download", te["name"]])
        verify_downloads([GRACE_CACHE / te["name"]])
    print(f"\n== Data ({da['source']})")
    if da["source"] == "spice":
        for key, name in SPICE_FILES.items():
            dest = Path(da[key])
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  exists: {dest}")
                verify_downloads([dest])
            else:
                try:
                    download(ZENODO_URL.format(name), dest)
                except checksums.ChecksumError as exc:
                    raise SystemExit(f"ERROR: {exc}")
                print(f"  checksum OK: {dest}")
    cmd_validate(args, quick=True)


# --------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------
def cmd_validate(args, quick=False):
    settings = load_settings()
    require_configured(settings)
    paths, problems, warnings = settings["paths"], [], []
    checks = {"STUDENT_CHECKPOINT_DIR": ["model.yaml", "checkpoint.index"],
              "STUDENT_MODEL_DIR": ["saved_model.pb"], "TEACHER_MODEL_DIR": ["saved_model.pb"],
              "TRAIN_POOL_XYZ": [], "TEST_XYZ": []}
    print("\n== Files")
    for key, needed in checks.items():
        value = paths.get(key, "")
        if key == "TEST_XYZ" and not value:
            print(f"  --   {key:24s} (none: test set held out from the pool)")
            continue
        p = Path(value).expanduser()
        ok = p.exists() and all((p / f).exists() for f in needed)
        print(f"  {'OK  ' if ok else 'FAIL'} {key:24s} {p}")
        if not ok:
            problems.append(f"{key} missing or incomplete: {p}")
    if "casted_model" in paths.get("TEACHER_MODEL_DIR", ""):
        problems.append("TEACHER_MODEL_DIR points to casted_model; use saved_model")
    if quick or problems:
        if problems:
            raise SystemExit("\nProblems:\n  " + "\n  ".join(problems))
        return

    versions = inspect(settings, "versions")
    print("\n== Library versions: " + ("OK" if versions["ok"] else "MISMATCH"))
    if not versions["ok"]:
        problems.append(f"pinned versions differ: {versions['versions']}")

    xyz = [paths["TRAIN_POOL_XYZ"]] + ([paths["TEST_XYZ"]] if paths.get("TEST_XYZ") else [])
    print("\n== Elements (reading the datasets, ~10-30 s)")
    el = inspect(settings, "elements", "--student-model", paths["STUDENT_MODEL_DIR"],
                 "--teacher", paths["TEACHER_MODEL_DIR"], "--xyz", *xyz)
    data_elements = sorted({e for d in el["data"].values() for e in d["elements"]})
    print(f"  data     {' '.join(data_elements)}")
    for role in ("teacher", "student"):
        covered = el[role]
        if covered is None:
            warnings.append(f"{role} has no metadata.yaml; element coverage not checked")
            continue
        missing = sorted(set(data_elements) - set(covered))
        print(f"  {role:8s} {len(covered)} elements" + (f"; MISSING {' '.join(missing)}" if missing else "; covers all"))
        if missing:
            problems.append(f"{role} does not cover data elements {missing}")
    for path, d in el["data"].items():
        print(f"  {Path(path).name}: {d['structures']} structures, {d['mean_atoms']:.0f} atoms on "
              f"average, {d['with_dft_forces']} with DFT forces")
    test_info = el["data"].get(paths.get("TEST_XYZ") or paths["TRAIN_POOL_XYZ"])
    if test_info and test_info["with_dft_forces"] < test_info["structures"]:
        warnings.append("test structures lack DFT labels: evaluation reports agreement with the "
                        "teacher only (no DFT errors)")
    pool = el["data"][paths["TRAIN_POOL_XYZ"]]["structures"]
    for name, plan in PLANS.items():
        need = plan["n_train"] + plan["n_val"] + (0 if paths.get("TEST_XYZ") else plan["n_test"])
        if need > pool:
            warnings.append(f"plan '{name}' needs {need} pool structures, pool has {pool}")
    for w in warnings:
        print(f"  WARNING: {w}")
    settings["validation"] = {"ok": not problems, "warnings": warnings, "elements": data_elements,
                              "time": time.strftime("%Y-%m-%d %H:%M")}
    save_settings(settings)
    if problems:
        raise SystemExit("\nProblems:\n  " + "\n  ".join(problems))
    print("\nValidation passed.")


# --------------------------------------------------------------------------------------
# smoke test + benchmark
# --------------------------------------------------------------------------------------
def read_history(run_dir: Path) -> list:
    with open(run_dir / "history.csv") as f:
        return [{k: float(v) if v not in ("", "nan") else float("nan") for k, v in row.items()}
                for row in csv.DictReader(f)]


def peak_child_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak / (2**30 if platform.system() == "Darwin" else 2**20)  # bytes on macOS, KiB on Linux


def train_cmd(settings, data_dir, runs_dir, name, job, epochs, device):
    cmd = [vbin(settings, "python"), "scripts/train_lora.py", "--data-dir", data_dir,
           "--runs-dir", runs_dir, "--name", name, "--epochs", epochs,
           "--batch-size", settings["batch_size"], "--max-buckets", settings["max_buckets"],
           *DEVICE_FLAGS[device]]
    if job.get("mode") == "full":
        cmd += ["--mode", "full"]
    else:
        cmd += ["--rank", job.get("rank", 8)]
    return cmd


def cmd_smoke(args):
    settings = load_settings()
    require_configured(settings)
    vpy, device = vbin(settings, "python"), settings["device"]
    base = STATE / "smoke"
    data, runs, log = base / "data", base / "runs", base / "smoke.log"
    base.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    checks = []

    print("\n== 1/4 Models load (check_models.py)")
    sh([vpy, "scripts/check_models.py"], log=log)

    print("\n== 2/4 Label a tiny dataset with the teacher")
    out = sh([vpy, "scripts/make_dataset.py", "--out", data, "--n-train", SMOKE["n_train"],
              "--n-val", SMOKE["n_val"], "--n-test", SMOKE["n_test"]], log=log)
    m = re.search(r"Done in (\d+) s", out)
    n_labelled = SMOKE["n_train"] + SMOKE["n_val"] + SMOKE["n_test"]
    label_s = int(m.group(1)) / n_labelled if m else float("nan")

    print(f"\n== 3/4 Train LoRA for {SMOKE['epochs']} epochs ({device})")
    t0 = time.time()
    sh(train_cmd(settings, data, runs, "smoke_lora", {"rank": SMOKE["rank"]}, SMOKE["epochs"], device),
       log=log)
    train_wall = time.time() - t0
    peak_gb = peak_child_rss_gb()
    hist = read_history(runs / "smoke_lora")
    times = [h["time_s"] for h in hist if h["epoch"] >= 1]
    steady = sum(times[1:]) / max(len(times) - 1, 1)
    train_s = steady / SMOKE["n_train"]
    compile_s = max(times[0] - steady, 0.0)
    overhead_s = max(train_wall - sum(times), 0.0)
    checks.append(("validation force error decreased",
                   hist[-1]["f_mae_meV_A"] < hist[0]["f_mae_meV_A"]))

    print("\n== 4/4 Evaluate base vs LoRA vs teacher")
    sh([vpy, "scripts/evaluate.py", "--data-dir", data, "--runs-dir", runs, "--runs", "smoke_lora"],
       log=log)
    ev = json.loads((runs / "eval.json").read_text())
    base_f = ev["student_base"]["force_mae_vs_teacher_meV_A"]
    lora_f = ev["student_smoke_lora"]["force_mae_vs_teacher_meV_A"]
    checks.append(("LoRA closer to teacher than base model", lora_f < base_f))
    checks.append(("teacher reproduces its own labels", ev["teacher"]["force_mae_vs_teacher_meV_A"] < 0.01))
    drift = ev["student_smoke_lora"].get("materials_drift_energy_meV_atom")
    if drift is not None and drift == drift:  # not NaN
        checks.append(("no energy drift on inorganic crystals", drift < 0.5))
    eval_s = sum(v["ms_per_structure"] for v in ev.values()) / len(ev) / 1000

    bench = {"device": device, "batch_size": settings["batch_size"],
             "max_buckets": settings["max_buckets"], "label_s_per_structure": label_s,
             "train_s_per_structure_epoch": train_s, "compile_s": compile_s,
             "process_overhead_s": overhead_s, "eval_s_per_structure_model": eval_s,
             "peak_train_rss_gb": round(peak_gb, 2),
             "smoke": {"base_force_vs_teacher": base_f, "lora_force_vs_teacher": lora_f},
             "time": time.strftime("%Y-%m-%d %H:%M")}

    if args.race:
        bench["race"] = gpu_race(settings, data, runs, log, steady)
        if bench["race"].get("winner") == "gpu":
            settings["device"] = "gpu"
            bench["train_s_per_structure_epoch"] = bench["race"]["gpu_epoch_s"] / SMOKE["n_train"]

    print("\n== Smoke test checks")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    ram = (settings.get("system") or {}).get("ram_gb")
    print(f"  peak training memory: {peak_gb:.1f} GB" + (f" of {ram} GB RAM" if ram else ""))
    if ram and peak_gb > 0.7 * ram:
        print("  WARNING: training uses >70% of RAM; lower --batch-size/--max-buckets "
              "via `configure` before a full run")
    bench["passed"] = all(ok for _, ok in checks)
    STATE.mkdir(exist_ok=True)
    (STATE / "benchmark.json").write_text(json.dumps(bench, indent=2))
    settings["benchmark"] = bench
    save_settings(settings)
    print_estimates(bench)
    if not bench["passed"]:
        raise SystemExit(f"Smoke test FAILED; full log: {log}")


def gpu_race(settings, data, runs, log, cpu_epoch_s) -> dict:
    gpus = (settings.get("system") or {}).get("gpus", [])
    if not gpus:
        print("\n== GPU race skipped: no GPU detected")
        return {"skipped": "no GPU"}
    print("\n== GPU race: 2 epochs on the GPU")
    out = sh(train_cmd(settings, data, runs, "smoke_gpu", {"rank": SMOKE["rank"]}, 2, "gpu"),
             log=log, check=False)
    hist_path = runs / "smoke_gpu" / "history.csv"
    if "Traceback" in out or not hist_path.exists() or len(read_history(runs / "smoke_gpu")) < 3:
        print("  GPU run failed: keeping CPU+XLA (details in the smoke log)")
        return {"winner": "cpu-jit", "gpu_error": out[-1500:]}
    gpu_epoch = read_history(runs / "smoke_gpu")[2]["time_s"]
    winner = "gpu" if gpu_epoch < cpu_epoch_s else "cpu-jit"
    print(f"  epoch 2: GPU {gpu_epoch:.1f} s vs CPU+XLA {cpu_epoch_s:.1f} s -> {winner}")
    return {"winner": winner, "gpu_epoch_s": gpu_epoch, "cpu_epoch_s": cpu_epoch_s}


# --------------------------------------------------------------------------------------
# estimate
# --------------------------------------------------------------------------------------
def estimate_plan(plan: dict, bench: dict) -> dict:
    n_label = plan["n_train"] + plan["n_val"] + plan["n_test"]
    label = n_label * bench["label_s_per_structure"] + 30
    per_job = (plan["epochs"] * plan["n_train"] * bench["train_s_per_structure_epoch"]
               + bench["compile_s"] + bench["process_overhead_s"])
    train = per_job * len(plan["jobs"])
    n_models = len(plan["jobs"]) + 2  # + base student + teacher
    evaluation = n_models * plan["n_test"] * bench["eval_s_per_structure_model"] + 30
    return {"label": label, "train_per_job": per_job, "train": train, "eval": evaluation,
            "total": label + train + evaluation}


def print_estimates(bench: dict):
    print(f"\n== Time estimates (device {bench['device']}, batch size {bench['batch_size']}; "
          "expect -25% / +40%)")
    print(f"  {'plan':7s} {'molecules':>10s} {'epochs':>6s} {'jobs':>4s} {'labelling':>10s} "
          f"{'per job':>10s} {'evaluation':>10s} {'TOTAL':>10s}")
    for name, plan in PLANS.items():
        e = estimate_plan(plan, bench)
        print(f"  {name:7s} {plan['n_train']:>10d} {plan['epochs']:>6d} {len(plan['jobs']):>4d} "
              f"{fmt_duration(e['label']):>10s} {fmt_duration(e['train_per_job']):>10s} "
              f"{fmt_duration(e['eval']):>10s} {fmt_duration(e['total']):>10s}")
    print("  jobs: pilot = LoRA r8; quick = LoRA r8 + full baseline; "
          "solid = LoRA r8/r4/r16 + full baseline")


def cmd_estimate(args):
    settings = load_settings()
    if "benchmark" not in settings:
        raise SystemExit("No benchmark yet: run `python3 pipeline.py smoke` first.")
    print_estimates(settings["benchmark"])


# --------------------------------------------------------------------------------------
# run + report
# --------------------------------------------------------------------------------------
def write_report(plan_name: str, plan: dict, settings: dict, data_dir: Path, runs_dir: Path):
    ev = json.loads((runs_dir / "eval.json").read_text())
    lines = [f"# GRACE LoRA distillation report: plan `{plan_name}`", "",
             f"- Student: `{settings['student']['name']}`  ",
             f"- Teacher: `{settings['teacher']['name']}`  ",
             f"- Data: {settings['data']['source']}; {plan['n_train']} train / {plan['n_val']} val / "
             f"{plan['n_test']} test structures; {plan['epochs']} epochs  ",
             f"- Device: {settings['device']}, batch size {settings['batch_size']}  ",
             f"- Generated: {time.strftime('%Y-%m-%d %H:%M')}", "",
             "## Held-out test set", "",
             "| Model | Force MAE vs DFT (meV/Å) | Force MAE vs teacher (meV/Å) | Energy MAE vs DFT "
             "(meV/atom) | Materials force drift (meV/Å) | Materials energy drift (meV/atom) |",
             "|---|---|---|---|---|---|"]

    def cell(d, k):
        v = d.get(k)
        return "–" if v is None or v != v else f"{v:.2f}"

    for model, d in ev.items():
        lines.append(f"| {model} | {cell(d, 'force_mae_vs_dft_meV_A')} | "
                     f"{cell(d, 'force_mae_vs_teacher_meV_A')} | {cell(d, 'energy_mae_vs_dft_meV_atom')} | "
                     f"{cell(d, 'materials_drift_force_meV_A')} | {cell(d, 'materials_drift_energy_meV_atom')} |")
    key = "force_mae_vs_dft_meV_A" if "force_mae_vs_dft_meV_A" in ev["teacher"] else "force_mae_vs_teacher_meV_A"
    base, teacher = ev["student_base"][key], ev["teacher"][key]
    lines += ["", "## Training runs", "",
              "| Run | Trainable params | Best val force MAE (meV/Å) | Best epoch | Train time | "
              "Gap to teacher closed |", "|---|---|---|---|---|---|"]
    for job in plan["jobs"]:
        summary = json.loads((runs_dir / job["name"] / "summary.json").read_text())
        hist = read_history(runs_dir / job["name"])
        best = min(hist, key=lambda h: h["f_mae_meV_A"])
        test_f = ev[f"student_{job['name']}"][key]
        closed = 100 * (base - test_f) / (base - teacher) if base != teacher else float("nan")
        lines.append(f"| {job['name']} | {summary['trainable_params']:,} "
                     f"({100 * summary['trainable_params'] / summary['total_params']:.1f}%) | "
                     f"{summary['val_best_f_mae_meV_A']:.2f} | {int(best['epoch'])} | "
                     f"{fmt_duration(summary['train_time_s'])} | {closed:.0f}% |")
    lines += ["", "Gap closed = (base − run) / (base − teacher) on the test force error "
              f"({'vs DFT' if key.endswith('dft_meV_A') else 'vs teacher'}).",
              "Materials drift compares each run with the base student on rattled inorganic "
              "crystals (energy drift only on crystals without the trained elements).", ""]
    (runs_dir / "report.md").write_text("\n".join(lines))
    print(f"\nReport: {runs_dir / 'report.md'}")


def cmd_run(args):
    settings = load_settings()
    require_configured(settings)
    plan = dict(PLANS[args.plan])
    if getattr(args, "jobs", None):
        plan["jobs"] = args.jobs
    for key in ("n_train", "n_val", "n_test", "epochs"):
        if getattr(args, key) is not None:
            plan[key] = getattr(args, key)
    vpy, device = vbin(settings, "python"), settings["device"]
    data_dir = Path(settings["paths"]["DATA_DIR"]) / args.plan
    runs_dir = Path(settings["paths"]["RUNS_DIR"]) / args.plan
    runs_dir.mkdir(parents=True, exist_ok=True)
    log = runs_dir / "pipeline.log"
    if "benchmark" in settings:
        e = estimate_plan(plan, settings["benchmark"])
        print(f"Plan '{args.plan}': estimated {fmt_duration(e['total'])} "
              f"(labelling {fmt_duration(e['label'])}, training {fmt_duration(e['train'])}, "
              f"evaluation {fmt_duration(e['eval'])})")
    else:
        print("No benchmark yet (run `smoke`); duration unknown.")
    if not confirm(f"Start plan '{args.plan}' now?", assume_yes=args.yes):
        return
    awake = keep_awake_prefix()
    t_start = time.time()

    marker = data_dir / "dataset.json"
    spec = {**{k: plan[k] for k in ("n_train", "n_val", "n_test")}, "paths": settings["paths"]}
    if marker.exists() and json.loads(marker.read_text()) == spec:
        print(f"\n== Dataset: reusing {data_dir}")
    else:
        print("\n== Dataset: labelling with the teacher")
        sh(awake + [vpy, "scripts/make_dataset.py", "--out", data_dir, "--n-train", plan["n_train"],
                    "--n-val", plan["n_val"], "--n-test", plan["n_test"]], log=log)
        marker.write_text(json.dumps(spec, indent=2))

    for job in plan["jobs"]:
        if (runs_dir / job["name"] / "summary.json").exists() and not args.force:
            print(f"\n== Training {job['name']}: already finished, skipping")
            continue
        print(f"\n== Training {job['name']}")
        sh(awake + train_cmd(settings, data_dir, runs_dir, job["name"], job, plan["epochs"], device),
           log=log)

    print("\n== Evaluation")
    sh(awake + [vpy, "scripts/evaluate.py", "--data-dir", data_dir, "--runs-dir", runs_dir,
                "--runs", *[j["name"] for j in plan["jobs"]], "--out", runs_dir / "eval.json"], log=log)
    write_report(args.plan, plan, settings, data_dir, runs_dir)
    print(f"\nPlan '{args.plan}' finished in {fmt_duration(time.time() - t_start)}. Log: {log}")


# --------------------------------------------------------------------------------------
# automatic mode
# --------------------------------------------------------------------------------------
def cmd_auto(args):
    print("GRACE LoRA distillation pipeline: automatic mode")
    print(f"Choices (top of pipeline.py): student {STUDENT} | teacher {TEACHER} | data {DATA} | "
          f"plan {PLAN}\n")
    ensure_settings_confirmed()

    rec = cmd_sysinfo(argparse.Namespace(json=False))
    if rec["blockers"]:
        raise SystemExit("\nFix the BLOCKER(s) above, then press Run again.")

    settings = load_settings()
    env_ready = Path(vbin(settings, "python")).exists()
    if env_ready:
        try:
            env_ready = inspect(settings, "versions")["ok"]
        except SystemExit:
            env_ready = False
    if env_ready:
        print("\n== Environment: ready")
    else:
        print("\n== Environment: creating .venv and installing libraries (several GB, 5-15 min)")
        cmd_setup_env(argparse.Namespace(python=None, cpu_only=False, venv=None))

    wanted = {"student": STUDENT, "student_model": STUDENT_SAVED_MODEL, "teacher": TEACHER,
              "data": DATA, "test": TEST_DATA, "workdir": WORKDIR}
    settings = load_settings()
    if settings.get("choices") != wanted or "paths" not in settings:
        print("\n== Configuration")
        cmd_configure(argparse.Namespace(
            student=STUDENT, student_model=STUDENT_SAVED_MODEL or None, teacher=TEACHER,
            data=DATA, test=TEST_DATA or None, device=None, batch_size=None, max_buckets=None,
            workdir=WORKDIR or None, yes=True))
        settings = load_settings()
        settings["choices"] = wanted
        for stale in ("validation", "benchmark"):  # different models or data: redo these
            settings.pop(stale, None)
        save_settings(settings)
    else:
        print("\n== Configuration: unchanged")

    print("\n== Models and data (downloads only what is missing)")
    cmd_fetch(argparse.Namespace())
    if not load_settings().get("validation", {}).get("ok"):
        cmd_validate(argparse.Namespace())
    else:
        print("\n== Validation: passed earlier")

    print("\n== Prepared data (organic pool, materials shards, monitors; reused if present)")
    cmd_prepare(argparse.Namespace(n_organic_pool=2000, n_val=100, n_test=300,
                                   materials_from=None, force=False))

    settings = load_settings()
    if not settings.get("benchmark", {}).get("passed"):
        has_gpu = bool((settings.get("system") or {}).get("gpus"))
        print("\n== Smoke test and timing benchmark (~10-15 min)")
        cmd_smoke(argparse.Namespace(race=RACE_GPU and has_gpu))
        settings = load_settings()
    else:
        print("\n== Smoke test: passed earlier")
        print_estimates(settings["benchmark"])

    plan = PLANS[PLAN]
    est = estimate_plan(plan, settings["benchmark"])["total"]
    finish = time.strftime("%H:%M", time.localtime(time.time() + est))
    print(f"\n== Plan '{PLAN}': {plan['n_train']} train / {plan['n_val']} val / {plan['n_test']} test "
          f"molecules, {plan['epochs']} epochs, jobs: {', '.join(j['name'] for j in plan['jobs'])}")
    print(f"   estimated {fmt_duration(est)} -> finishes around {finish}")

    start = AUTO_RUN or (sys.stdin.isatty() and confirm("\nStart it now?", default=False))
    if not start:
        print("\nNot started. Press Run again and answer 'y', or set AUTO_RUN = True.")
        return
    cmd_run(argparse.Namespace(plan=PLAN, yes=True, force=False, n_train=plan["n_train"],
                               n_val=plan["n_val"], n_test=plan["n_test"], epochs=plan["epochs"],
                               jobs=plan["jobs"]))


# --------------------------------------------------------------------------------------
# prepared data + autonomous experiments (see docs/CONTRACTS.md)
# --------------------------------------------------------------------------------------
# optional agent features: all OFF unless enabled with `pipeline.py options --enable NAME`
AGENT_OPTIONS = {
    "subagents": "master switch: may the skill launch helper agents at all (limited-usage "
                 "accounts: keep off)",
    "critic": "after each experiment, a critic agent questions the result before the next "
              "decision (needs subagents)",
    "self_heal": "experiment.py retries failed runs automatically (e.g. out of memory -> smaller "
                 "batch size) instead of stopping",
    "cockpit": "keep a live HTML dashboard of the ledger and monitors (pipeline.py cockpit)",
    "cockpit_publish": "republish the cockpit as a private Claude artifact after each experiment "
                       "(Claude Code with the Artifact tool only)",
    "predictions": "before each experiment the agent writes a hypothesis and predicted monitor "
                   "values (--hypothesis/--predict), scored against the outcome",
}
NUMERIC_OPTIONS = {"max_subagents": 3}  # cap on helper agents per loop (integer >= 0)
NEEDS_SUBAGENTS = ("critic",)
PREDICT_KEYS = ("organic_f", "materials_f", "only_adapted_f")


def get_agent_options(settings: dict) -> dict:
    """{name: bool} for every feature plus max_subagents; missing features mean False."""
    stored = settings.get("agent_options") or {}
    options = {name: bool(stored.get(name, False)) for name in AGENT_OPTIONS}
    for name, default in NUMERIC_OPTIONS.items():
        try:
            options[name] = max(int(stored.get(name, default)), 0)
        except (TypeError, ValueError):
            options[name] = default
    return options


def set_agent_options(settings: dict, enable=(), disable=(), sets=()) -> tuple:
    """Validate and apply --enable/--disable/--set; store the full table in settings.
    Returns (options, warnings)."""
    enable, disable = list(enable or ()), list(disable or ())
    unknown = sorted(n for n in set(enable) | set(disable) if n not in AGENT_OPTIONS)
    if unknown:
        raise SystemExit(f"unknown option(s) {unknown}; choose from {', '.join(AGENT_OPTIONS)}")
    both = sorted(set(enable) & set(disable))
    if both:
        raise SystemExit(f"option(s) {both} both enabled and disabled")
    options = get_agent_options(settings)
    options.update({n: True for n in enable})
    options.update({n: False for n in disable})
    for item in sets or ():
        key, sep, value = item.partition("=")
        if not sep or key not in NUMERIC_OPTIONS:
            raise SystemExit(f"--set expects KEY=VALUE with KEY in {', '.join(NUMERIC_OPTIONS)}; "
                             f"got '{item}'")
        try:
            number = int(value)
        except ValueError:
            raise SystemExit(f"{key} must be an integer >= 0, got '{value}'")
        if number < 0:
            raise SystemExit(f"{key} must be an integer >= 0, got {number}")
        options[key] = number
    warnings = [f"WARNING: '{n}' is enabled but needs subagents, which are OFF: it stays "
                "inactive until you run `python3 pipeline.py options --enable subagents`"
                for n in NEEDS_SUBAGENTS if options[n] and not options["subagents"]]
    settings["agent_options"] = options
    return options, warnings


def subagents_line(options: dict) -> str:
    if not options["subagents"]:
        return "Subagents: OFF (the skill will not launch any agents)"
    return f"Subagents: ON (at most {options['max_subagents']} per loop)"


def cmd_options(args):
    settings = load_settings()
    before = settings.get("agent_options")
    options, warnings = set_agent_options(settings, args.enable, args.disable, args.set)
    if options != before:
        if before is not None:
            settings["settings_confirmed"] = False
        save_settings(settings)
    if args.json:
        print(json.dumps(options, indent=2))
    else:
        print(subagents_line(options))
        print(f"  max_subagents = {options['max_subagents']}  "
              "(pipeline.py options --set max_subagents=N)")
        print("Agent options (all off by default; pipeline.py options --enable NAME / --disable NAME):")
        for name in AGENT_OPTIONS:
            print(f"  [{'on ' if options[name] else 'off'}] {name:12s} {AGENT_OPTIONS[name]}")
    for w in warnings:
        print(w)


# --------------------------------------------------------------------------------------
# settings: everything that affects training, quality and usage (pipeline.py settings)
# --------------------------------------------------------------------------------------
SETTINGS_GROUPS = ("Models & data", "Training & quality", "Objective & budget", "Validation",
                   "Usage & agent features")
AFFECTS = ("quality", "time", "tokens", "memory")


def _s(group, default, kind, short, long, affects, allowed=None, lo=None, hi=None,
       nullable=False, read_only=False):
    return {"group": group, "default": default, "type": kind, "allowed": allowed, "min": lo,
            "max": hi, "nullable": nullable, "affects": affects, "short": short, "long": long,
            "read_only": read_only}


SETTING_SPECS = {
    # ---- Models & data (read-only here; change with `configure`) ----
    "student": _s("Models & data", DEFAULT_STUDENT, "str", "base GRACE model that gets the adapter",
                  "The pretrained GRACE foundation model whose weights stay frozen; the LoRA adapter "
                  "is added on top of it. This project used GRACE-2L-OMAT-medium-ft-AM (21.1M "
                  "parameters, 89 elements). Larger students are slower to train and evaluate. "
                  "Change it with `python3 pipeline.py configure --student NAME`.",
                  ["quality", "time", "memory"], read_only=True),
    "teacher": _s("Models & data", DEFAULT_TEACHER, "str", "model whose labels the student learns",
                  "The specialist model that labels the training molecules (distillation). The "
                  "default GRACE-OFF 2L-medium is ~15 meV/Å from DFT on organic molecules, so its "
                  "labels are not the bottleneck. Its elements define which element rows of the "
                  "student are adapted. Change it with `python3 pipeline.py configure --teacher ...`.",
                  ["quality"], read_only=True),
    "data": _s("Models & data", "spice", "str", "where training/test structures come from",
               "The structure source: 'spice' (Zenodo SPICE v2 sets) or your own extended-XYZ file. "
               "The test set is held out and never trained on. Change it with "
               "`python3 pipeline.py configure --data ...`.", ["quality"], read_only=True),
    # ---- Training & quality ----
    "experiment_defaults.n_organic": _s(
        "Training & quality", 1000, "int", "teacher-labelled molecules per experiment",
        "How many organic molecules (teacher labels) each experiment trains on. 500 gave 106 "
        "meV/Å on held-out molecules; 1000 gave 97 (run L2). More molecules improve organic "
        "accuracy but training time grows linearly. Must not exceed the prepared pool "
        "(`prepare --n-organic-pool`, default 2000).", ["quality", "time"], lo=10, hi=20000),
    "experiment_defaults.n_materials": _s(
        "Training & quality", 600, "int", "replay materials (DFT) mixed into training",
        "Materials with the adapted elements, with DFT labels, mixed in so base GRACE does not "
        "forget materials (replay). Without replay, materials error rose 2.3x (46 -> 105 meV/Å); "
        "with 600 it stayed at 51. 0 disables replay (organic-only, faster, forgets). Raise it if "
        "the materials monitor degrades.", ["quality", "time"], lo=0, hi=20000),
    "experiment_defaults.n_materials_val": _s(
        "Training & quality", 100, "int", "replay materials in the validation split",
        "Materials held out from training to pick the best epoch together with the validation "
        "molecules. 100 worked well; fewer makes epoch selection noisier, more costs a little "
        "evaluation time per epoch.", ["time"], lo=0, hi=5000),
    "experiment_defaults.n_only_adapted": _s(
        "Training & quality", 170, "int", "quota of crystals made only of adapted elements",
        "How many replay materials must be crystals made ONLY of adapted elements (molecular "
        "crystals, C/P/S solids) - the hardest group for forgetting. Raising it from ~10 to 170 "
        "cut that group's error from 148 to 124 meV/Å. Raise it further if the only_adapted "
        "monitor stays high; it competes with organic accuracy.", ["quality"], lo=0, hi=5000),
    "experiment_defaults.rank": _s(
        "Training & quality", 8, "int", "LoRA adapter rank (capacity)",
        "Rank of the low-rank update W = W0 + (alpha/r) B A. Rank 8 trains 1.36M parameters "
        "(6.4% of the model) and was used for all results. Higher rank adds capacity for both "
        "domains at the cost of memory and adapter size; try 16 when organic and materials "
        "errors both plateau.", ["quality", "memory", "time"], lo=1, hi=64),
    "experiment_defaults.alpha": _s(
        "Training & quality", None, "float", "LoRA scale (null = 2 x rank)",
        "Scaling of the adapter update (alpha/rank). null means 2 x rank, which was used "
        "throughout this project. Larger alpha makes the adapter learn faster but less stably. "
        "Rarely needs changing.", ["quality"], lo=1e-6, nullable=True),
    "experiment_defaults.epochs": _s(
        "Training & quality", 10, "int", "training passes over the data",
        "Epochs per experiment; the best epoch on the validation split is exported. 10 epochs "
        "with cosine decay gave the reported results, and the monitors still improved ~1 meV/Å "
        "per epoch at the end. Time grows linearly; 12-15 squeezes out a little more.",
        ["quality", "time"], lo=1, hi=100),
    "experiment_defaults.lr": _s(
        "Training & quality", None, "float", "peak learning rate (null = 2e-3 for LoRA)",
        "Peak learning rate of the cosine schedule. null means 2e-3 for LoRA (5e-4 for full "
        "fine-tuning), the values used here. Too high destabilises training; too low wastes "
        "epochs.", ["quality"], lo=1e-7, hi=1.0, nullable=True),
    "experiment_defaults.max_atoms": _s(
        "Training & quality", 30, "int", "largest replay material (atoms)",
        "Replay materials larger than this are skipped. 30 kept training memory around 6 GB at "
        "batch size 8 on an 11.7 GB laptop. Raising it adds larger crystals (more realistic) but "
        "costs memory and time per batch.", ["memory", "time", "quality"], lo=5, hi=200),
    "experiment_defaults.oversample": _s(
        "Training & quality", 3.0, "float", "extra weight for H/F (and O) containing materials",
        "Sampling weight for replay materials containing the most damaged elements (H, F; also "
        "O for only-adapted crystals). 3.0 was used for runs L1 and L2. Raise it if H/F/O errors stay "
        "high on materials; 1.0 samples uniformly.", ["quality"], lo=1.0, hi=100.0),
    "experiment_defaults.seed": _s(
        "Training & quality", 0, "int", "random seed for data sampling",
        "Seed for which molecules and materials are sampled. Results are deterministic for a "
        "given seed. Change it to check that an improvement is not a sampling accident.",
        ["quality"], lo=0, hi=2**31 - 1),
    "device": _s(
        "Training & quality", "cpu-jit", "choice", "training device",
        "cpu-jit = CPU with XLA (about 2.2x faster than plain CPU on this project's laptop); cpu "
        "= plain CPU; gpu = try the GPU for training (Metal on Macs is untested; the downloaded "
        "models always run on the CPU). Keep cpu-jit unless the smoke test's GPU race says "
        "otherwise.", ["time"], allowed=("cpu-jit", "cpu", "gpu")),
    "batch_size": _s(
        "Training & quality", 8, "int", "structures per training step",
        "8 with at most 4 batch shapes peaked at ~6.3 GB; 16 exceeded 14 GB and swapped on an "
        "11.7 GB machine. Use 16 only with >= 24 GB RAM. Smaller batches use less memory but "
        "are slower.", ["memory", "time"], lo=1, hi=64),
    "max_buckets": _s(
        "Training & quality", 4, "int", "distinct batch shapes (XLA compilations)",
        "Each batch shape is compiled separately with XLA; fewer shapes means less memory and "
        "compile time, a little more padding. 4 was memory-safe here; raise to 8 with plenty of "
        "RAM.", ["memory", "time"], lo=1, hi=16),
    # ---- Objective & budget ----
    "objective.max_materials_increase": _s(
        "Objective & budget", 0.15, "float", "allowed materials error increase vs base (0.15 = 15%)",
        "An experiment is feasible only if its materials monitor error is at most this fraction "
        "above base GRACE. Run L2 was at about +9% (54.6 vs 50.0 meV/Å). Tighten it to protect "
        "materials more, loosen it to favour organic accuracy.", ["quality"], lo=0.0, hi=10.0),
    "objective.max_only_adapted_increase": _s(
        "Objective & budget", 0.75, "float", "allowed increase on only-adapted crystals (0.75 = 75%)",
        "Same as above for crystals made only of adapted elements, the group that degrades most "
        "(L2: 107.7 vs 62.7 meV/Å, +72%). Lower it to force the agent to protect molecular "
        "crystals and C/P/S solids, at the expense of organic accuracy.", ["quality"],
        lo=0.0, hi=10.0),
    "budget.time_budget_min": _s(
        "Objective & budget", 240, "int", "total minutes for the autonomous loop",
        "The agent plans experiments so their estimated total stays within this time. One L2-size "
        "experiment took ~55 min on the test laptop's CPU. Raise it for overnight runs.",
        ["time", "tokens"], lo=1, hi=100000),
    "budget.max_experiments": _s(
        "Objective & budget", 4, "int", "maximum experiments in the loop",
        "Upper bound on experiments the agent runs, whatever the time budget. Each experiment is "
        "one training run plus monitors; more experiments explore more settings but cost time "
        "and agent tokens.", ["time", "tokens"], lo=1, hi=50),
    # ---- Validation ----
    "validation.mid_loop": _s(
        "Validation", "quick", "choice", "validation size during the loop",
        "quick = 100 organic molecules and 50/150/60 materials (A/B/C), a few minutes; full = all "
        "300 molecules and 200/500/all materials, ~12 min. Quick is enough to steer the loop; "
        "the monitors already track progress every epoch.", ["time"], allowed=("quick", "full")),
    "validation.final": _s(
        "Validation", "full", "choice", "validation size for the final/best model",
        "Size of the validation of the base reference and the best experiment, which go into the "
        "report and heatmap. Keep full for numbers you will cite; quick only for a fast demo.",
        ["time", "quality"], allowed=("quick", "full")),
    # ---- Usage & agent features ----
    "agent_options.subagents": _s(
        "Usage & agent features", False, "bool", "allow the skill to launch helper agents",
        "Master switch: when off, the skill never launches extra agents (important for "
        "limited-usage accounts). Features that need agents (critic) stay inactive while it is "
        "off.", ["tokens"]),
    "agent_options.max_subagents": _s(
        "Usage & agent features", 3, "int", "cap on helper agents per loop",
        "Upper bound on helper agents launched in one loop when subagents are on. Each agent "
        "costs tokens; 0 is equivalent to subagents off.", ["tokens"], lo=0, hi=20),
    "agent_options.critic": _s(
        "Usage & agent features", False, "bool", "critic agent reviews each result",
        "After each experiment, a critic agent questions the result before the next decision "
        "(e.g. 'is this gain within noise?'). More careful decisions at the cost of tokens; "
        "needs subagents on.", ["tokens", "quality"]),
    "agent_options.self_heal": _s(
        "Usage & agent features", False, "bool", "retry failed runs automatically",
        "Failed experiments (e.g. out of memory) are retried with safer settings (smaller batch "
        "size) instead of stopping the loop. Saves babysitting; failures still appear in the "
        "ledger.", ["time"]),
    "agent_options.predictions": _s(
        "Usage & agent features", False, "bool", "agent predicts results before each experiment",
        "Before each experiment the agent records a hypothesis and predicted monitor values, "
        "later scored against the outcome - makes the reasoning auditable. Costs a few tokens "
        "per experiment.", ["tokens"]),
    "agent_options.cockpit": _s(
        "Usage & agent features", False, "bool", "live HTML dashboard of the loop",
        "Keeps report/cockpit.html updated with the ledger and monitor curves while the loop "
        "runs (`pipeline.py cockpit`). Small overhead; useful for demos.", ["time"]),
    "agent_options.cockpit_publish": _s(
        "Usage & agent features", False, "bool", "republish the cockpit as a web artifact",
        "After each finished experiment the agent rebuilds the cockpit (`pipeline.py cockpit "
        "--artifact`) and republishes it to the same private claude.ai link, so others can follow "
        "the loop from a browser or phone. Only in Claude Code with the Artifact tool; publishing "
        "sends results to claude.ai (private until shared). A few tokens per experiment.",
        ["tokens"]),
}
DEFAULT_S_PER_STRUCTURE_EPOCH = 0.2


def resolve_setting(key: str) -> str:
    """Full registry key from a dotted or unambiguous short key ('rank', 'critic')."""
    if key in SETTING_SPECS:
        return key
    matches = [k for k in SETTING_SPECS if k.rsplit(".", 1)[-1] == key]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise SystemExit(f"'{key}' is ambiguous: {', '.join(matches)}")
    raise SystemExit(f"unknown setting '{key}'. Known: {', '.join(SETTING_SPECS)}")


def get_setting(settings: dict, key: str):
    key = resolve_setting(key)
    spec = SETTING_SPECS[key]
    if key == "student":
        return (settings.get("student") or {}).get("name", spec["default"])
    if key == "teacher":
        return (settings.get("teacher") or {}).get("name", spec["default"])
    if key == "data":
        return (settings.get("data") or {}).get("source", spec["default"])
    node = settings
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return spec["default"]
        node = node[part]
    return node


def coerce_setting(key: str, raw):
    """Parse and validate a value for a setting; SystemExit with a clear message if invalid."""
    spec = SETTING_SPECS[key]
    value = parse_value(raw) if isinstance(raw, str) else raw
    if value is None:
        if spec["nullable"]:
            return None
        raise SystemExit(f"{key} cannot be null")
    kind = spec["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise SystemExit(f"{key} must be true or false, got {raw!r}")
        return value
    if kind == "choice":
        if value not in spec["allowed"]:
            raise SystemExit(f"{key} must be one of {', '.join(spec['allowed'])}, got {raw!r}")
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise SystemExit(f"{key} must be an integer, got {raw!r}")
        value = int(value)
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"{key} must be a number, got {raw!r}")
        value = float(value)
    if spec["min"] is not None and value < spec["min"]:
        raise SystemExit(f"{key} must be >= {spec['min']}, got {value}")
    if spec["max"] is not None and value > spec["max"]:
        raise SystemExit(f"{key} must be <= {spec['max']}, got {value}")
    return value


def put_setting(settings: dict, key: str, value):
    parts = key.split(".")
    node = settings
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_settings_changes(settings: dict, sets=(), reset=None) -> list:
    """Apply --reset (None = no reset, [] = all, [keys] = these) then --set KEY=VALUE.
    Any change sets settings_confirmed back to false. Returns the changed keys."""
    changed = []
    if reset is not None:
        keys = [resolve_setting(k) for k in reset] or [k for k in SETTING_SPECS]
        for key in keys:
            if SETTING_SPECS[key]["read_only"]:
                if reset:
                    raise SystemExit(f"{key} is read-only here: change it with `pipeline.py configure`")
                continue
            put_setting(settings, key, SETTING_SPECS[key]["default"])
            changed.append(key)
    for item in sets or ():
        name, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got '{item}'")
        key = resolve_setting(name.strip())
        if SETTING_SPECS[key]["read_only"]:
            raise SystemExit(f"{key} is read-only here: change it with `pipeline.py configure`")
        put_setting(settings, key, coerce_setting(key, raw.strip()))
        changed.append(key)
    if changed:
        settings["settings_confirmed"] = False
    return changed


def experiment_defaults(settings: dict) -> dict:
    """experiment_defaults + device/batch_size/max_buckets (stored or default)."""
    config = {k.split(".", 1)[1]: get_setting(settings, k) for k in SETTING_SPECS
              if k.startswith("experiment_defaults.")}
    for key in ("device", "batch_size", "max_buckets"):
        config[key] = get_setting(settings, key)
    return config


def estimate_minutes(settings: dict) -> dict:
    """Training-time estimate from the benchmark (s/structure-epoch) or 0.2 s."""
    bench = settings.get("benchmark") or {}
    s = bench.get("train_s_per_structure_epoch")
    source = "smoke-test benchmark" if s else "default 0.2 s"
    s = float(s) if s else DEFAULT_S_PER_STRUCTURE_EPOCH
    n = get_setting(settings, "n_organic") + get_setting(settings, "n_materials")
    per = n * get_setting(settings, "epochs") * s / 60
    count = get_setting(settings, "max_experiments")
    return {"per_experiment_min": per, "total_min": per * count, "max_experiments": count,
            "budget_min": get_setting(settings, "time_budget_min"), "s_per_structure_epoch": s,
            "structures": n, "source": source}


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def settings_panel(settings: dict) -> str:
    lines = ["Settings that affect training, quality and usage (✱ = changed from default)"]
    for group in SETTINGS_GROUPS:
        keys = [k for k, v in SETTING_SPECS.items() if v["group"] == group]
        lines += ["", f"== {group}" + ("  (change with `pipeline.py configure`)"
                                       if group == "Models & data" else "")]
        lines.append(f"  {'setting':36s} {'current':>16s} {'default':>16s}  {'affects':22s} meaning")
        for key in keys:
            spec, value = SETTING_SPECS[key], get_setting(settings, key)
            mark = "✱" if value != spec["default"] else " "
            current = _fmt(value)
            if len(current) > 15:
                current = current[:14] + "…"
            default = _fmt(spec["default"])
            if len(default) > 15:
                default = default[:14] + "…"
            lines.append(f"  {key:36s} {mark}{current:>15s} {default:>16s}  "
                         f"{','.join(spec['affects']):22s} {spec['short']}")
    est = estimate_minutes(settings)
    lines += ["", f"≈ {est['per_experiment_min']:.0f} min per experiment ({est['structures']} "
                  f"structures × {get_setting(settings, 'epochs')} epochs × "
                  f"{est['s_per_structure_epoch']:.2f} s/structure-epoch, {est['source']}), "
                  f"≈ {est['total_min']:.0f} min for {est['max_experiments']} experiments, "
                  f"vs time budget {est['budget_min']} min"
                  + ("  (over budget!)" if est['total_min'] > est['budget_min'] else "")]
    if not get_setting(settings, "agent_options.subagents"):
        lines.append("Subagents: OFF (no extra agents launched)")
    else:
        lines.append(f"Subagents: ON (at most {get_setting(settings, 'max_subagents')} per loop)")
    lines.append("Settings confirmed: " + ("yes" if settings.get("settings_confirmed") else "NO"))
    lines += ["", "Details: .claude/skills/grace-lora/references/settings.md, or ask Claude to "
                  "explain any setting. Confirm with `python3 pipeline.py settings --confirm`."]
    return "\n".join(lines)


def settings_json(settings: dict) -> dict:
    return {"settings": {k: {"value": get_setting(settings, k), "default": v["default"],
                             "group": v["group"], "affects": v["affects"], "short": v["short"],
                             "changed": get_setting(settings, k) != v["default"],
                             "read_only": v["read_only"]} for k, v in SETTING_SPECS.items()},
            "settings_confirmed": bool(settings.get("settings_confirmed")),
            "estimate": estimate_minutes(settings)}


def cmd_settings(args):
    settings = load_settings()
    if args.explain:
        key = resolve_setting(args.explain)
        spec = SETTING_SPECS[key]
        print(f"{key}  (group: {spec['group']}; default: {_fmt(spec['default'])}; current: "
              f"{_fmt(get_setting(settings, key))}; affects: {', '.join(spec['affects'])})")
        print(spec["long"])
        return
    changed = apply_settings_changes(settings, args.set, args.reset)
    if args.confirm:
        settings["settings_confirmed"] = True
    if changed or args.confirm:
        save_settings(settings)
    if changed:
        print(f"Changed: {', '.join(changed)} (settings need confirming again)")
    if args.json:
        print(json.dumps(settings_json(settings), indent=2))
    else:
        print(settings_panel(settings))
    if args.confirm:
        print("\nSettings confirmed.")


def ensure_settings_confirmed():
    """auto/wizard: show the panel; ask for confirmation in a terminal, else stop."""
    settings = load_settings()
    print(settings_panel(settings))
    if settings.get("settings_confirmed"):
        return
    if sys.stdin.isatty() and confirm("\nUse these settings?", default=True):
        settings["settings_confirmed"] = True
        save_settings(settings)
        return
    raise SystemExit("\nSettings not confirmed. Review them, then run "
                     "`python3 pipeline.py settings --confirm` (or change them with --set).")


LAYOUT = {"ORGANIC_DIR": "data/organic", "MATERIALS_DIR": "data/materials",
          "MONITORS_DIR": "data/monitors", "EXPERIMENTS_DIR": "experiments",
          "VALIDATION_DIR": "validation", "REPORT_DIR": "report"}
MONITOR_FILES = ("organic.pkl.gz", "materials.pkl.gz", "only_adapted.pkl.gz")


def layout_paths(workdir: Path) -> dict:
    return {k: str(workdir / v) for k, v in LAYOUT.items()}


def wpath(settings: dict, key: str) -> Path:
    """A runtime-layout path (pipeline.json "paths", else <workdir>/<default>)."""
    workdir = Path(settings.get("workdir") or ROOT)
    return Path(settings.get("paths", {}).get(key) or workdir / LAYOUT[key]).expanduser()


def script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def parse_value(text: str):
    """--set values: JSON first (numbers, true/false/null, lists, quoted strings), else a
    plain string. Python-style True/False/None are accepted too."""
    alias = {"True": True, "False": False, "None": None}
    if text in alias:
        return alias[text]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_sets(items) -> dict:
    """['rank=16', 'epochs=12', 'adapt_elements=["C","H"]'] -> {'rank': 16, ...}"""
    config = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set expects key=value, got '{item}'")
        config[key.strip()] = parse_value(value.strip())
    return config


def load_ledger(settings: dict) -> dict:
    path = wpath(settings, "EXPERIMENTS_DIR") / "ledger.json"
    return json.loads(path.read_text()) if path.exists() else {"experiments": []}


def best_experiment(ledger: dict):
    """docs/CONTRACTS.md: the feasible finished experiment with the lowest score, else the one
    with the lowest materials_increase. None if nothing finished."""
    done = [e for e in ledger.get("experiments", []) if e.get("status") == "done"]
    feasible = [e for e in done if e.get("feasible")]
    if feasible:
        return min(feasible, key=lambda e: e.get("score", float("inf")))
    if done:
        return min(done, key=lambda e: (e.get("relative") or {}).get("materials_increase",
                                                                      float("inf")))
    return None


def cmd_prepare(args):
    """Organic pool (teacher labels), MPtrj shards and the three monitor sets. Idempotent."""
    settings = load_settings()
    require_configured(settings)
    vpy = vbin(settings, "python")
    organic, materials, monitors = (wpath(settings, k) for k in
                                    ("ORGANIC_DIR", "MATERIALS_DIR", "MONITORS_DIR"))
    log = wpath(settings, "EXPERIMENTS_DIR") / "prepare.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    spec = {"n_train": args.n_organic_pool, "n_val": args.n_val, "n_test": args.n_test,
            "pool": settings["paths"].get("TRAIN_POOL_XYZ"),
            "test": settings["paths"].get("TEST_XYZ"),
            "teacher": settings["paths"].get("TEACHER_MODEL_DIR")}
    marker = organic / "pool.json"
    have = all((organic / f"{s}.pkl.gz").exists() for s in ("train", "val", "test"))
    if have and marker.exists() and json.loads(marker.read_text()) == spec and not args.force:
        print(f"\n== Organic pool: reusing {organic}")
    else:
        print(f"\n== Organic pool: labelling {args.n_organic_pool} + {args.n_val} + {args.n_test} "
              "molecules with the teacher")
        sh(keep_awake_prefix() + [vpy, script("make_dataset.py"), "--out", organic,
                                  "--n-train", args.n_organic_pool, "--n-val", args.n_val,
                                  "--n-test", args.n_test], log=log)
        marker.write_text(json.dumps(spec, indent=2))

    print("\n== Materials (MPtrj shards)")
    cmd = [vpy, script("fetch_materials.py"), "--out", materials]
    if args.materials_from:
        cmd += ["--from-dir", args.materials_from]
    sh(cmd, log=log)

    mon_marker = monitors / "monitors.json"
    mon_spec = {"organic_pool": spec, "val_shard": str(materials / "val.parquet")}
    have = all((monitors / f).exists() for f in MONITOR_FILES)
    if have and mon_marker.exists() and json.loads(mon_marker.read_text()) == mon_spec \
            and not args.force:
        print(f"\n== Monitors: reusing {monitors}")
    else:
        print("\n== Monitors: building the held-out organic / materials / only-adapted sets")
        sh([vpy, script("make_replay_dataset.py"), "--monitors-only", "--organic-dir", organic,
            "--exclude", materials / "val.parquet", "--out", monitors], log=log)
        mon_marker.write_text(json.dumps(mon_spec, indent=2))
    settings["prepared"] = {"organic": spec, "time": time.strftime("%Y-%m-%d %H:%M")}
    save_settings(settings)
    print(f"\nPrepared. Next: python3 pipeline.py experiment --set ... --rationale \"...\"")


def parse_predictions(items) -> list:
    """Validate --predict KEY=VALUE items (keys organic_f/materials_f/only_adapted_f, numeric
    values); returned unchanged for pass-through to experiment.py."""
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or key not in PREDICT_KEYS:
            raise SystemExit(f"--predict expects KEY=VALUE with KEY in {', '.join(PREDICT_KEYS)}; "
                             f"got '{item}'")
        try:
            float(value)
        except ValueError:
            raise SystemExit(f"--predict value must be a number: '{item}'")
    return list(items or [])


def experiment_command(vpy: str, settings: dict, args) -> list:
    """The scripts/experiment.py command line for `pipeline.py experiment` (pure)."""
    # precedence: experiment_defaults < pipeline.json device/batch_size/max_buckets < --set
    config = {**experiment_defaults(settings), **parse_sets(args.set)}
    workdir = Path(settings.get("workdir") or ROOT)
    cmd = [vpy, script("experiment.py"), "--workdir", workdir, "--config", json.dumps(config),
           "--rationale", args.rationale or "", "--python", vpy]
    if args.parent:
        cmd += ["--parent", args.parent]
    self_heal = getattr(args, "self_heal", None)
    if self_heal is None:
        self_heal = get_agent_options(settings)["self_heal"]
    if self_heal:
        cmd += ["--self-heal"]
    if getattr(args, "hypothesis", None):
        cmd += ["--hypothesis", args.hypothesis]
    for item in parse_predictions(getattr(args, "predict", None)):
        cmd += ["--predict", item]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def cmd_experiment(args):
    settings = load_settings()
    require_configured(settings)
    cmd = experiment_command(vbin(settings, "python"), settings, args)
    log = wpath(settings, "EXPERIMENTS_DIR") / "experiments.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    sh(([] if args.dry_run else keep_awake_prefix()) + cmd, log=log)


def cockpit_command(vpy: str, settings: dict, args) -> list:
    workdir = Path(settings.get("workdir") or ROOT)
    out = Path(args.out).expanduser() if args.out else wpath(settings, "REPORT_DIR") / "cockpit.html"
    if getattr(args, "artifact", False):
        out = (Path(args.out).expanduser() if args.out
               else wpath(settings, "REPORT_DIR") / "cockpit_artifact" / "cockpit.html")
        return [vpy, script("cockpit.py"), "--workdir", workdir, "--out", out, "--artifact"]
    cmd = [vpy, script("cockpit.py"), "--workdir", workdir, "--out", out]
    if args.watch:
        cmd += ["--watch", args.watch]
    return cmd


def cmd_cockpit(args):
    settings = load_settings()
    require_configured(settings)
    sh(cockpit_command(vbin(settings, "python"), settings, args))


def cmd_ledger(args):
    settings = load_settings()
    require_configured(settings)
    workdir = Path(settings.get("workdir") or ROOT)
    action = ["best"] if args.best else ["show"] + (["--json"] if args.json else [])
    sh([vbin(settings, "python"), script("ledger.py"), "--workdir", workdir, *action])


QUICK_SIZES = {"organic": ["--n", 100], "materials": ["--n-a", 50, "--n-b", 150, "--n-c", 60]}


def detect_model(path: Path) -> dict:
    """What kind of model a user-supplied path is (see `validate-models --help`)."""
    path = Path(path).expanduser()
    if path.name.endswith(".index") and path.exists():
        path = path.with_suffix("")
    if (path / "saved_model.pb").exists():
        return {"kind": "saved_model", "model_dir": path, "source": path}
    if (path / "saved_model" / "saved_model.pb").exists():
        return {"kind": "run", "model_dir": path / "saved_model", "source": path}
    if (path / "best_trainable_state.npz").exists() and \
            ((path / "adapter.json").exists() or (path / "args.json").exists()):
        return {"kind": "adapter", "model_dir": None, "source": path}
    if Path(str(path) + ".index").exists():
        return {"kind": "checkpoint", "model_dir": None, "source": path}
    if (path / "best_checkpoint.index").exists():
        return {"kind": "checkpoint", "model_dir": None, "source": path / "best_checkpoint"}
    if (path / "checkpoint.index").exists():
        return {"kind": "checkpoint", "model_dir": None, "source": path / "checkpoint"}
    raise SystemExit(f"Cannot use {path}: not a SavedModel (saved_model.pb), a train_lora run "
                     "(saved_model/), a LoRA adapter (adapter.json + best_trainable_state.npz) or "
                     "a checkpoint (<prefix>.index / best_checkpoint.index)")


KIND_TEXT = {"saved_model": "SavedModel", "run": "train_lora run (uses its saved_model/)",
             "adapter": "LoRA adapter (base + adapter -> SavedModel via apply_adapter.py)",
             "checkpoint": "raw checkpoint (exported via export_checkpoint.py)"}


def resolve_models(specs, out: Path, vpy: str) -> list:
    """['name=path', ...] -> [{name, kind, source, model_dir, build}] where `build` is the
    command creating model_dir (None if it already exists or nothing must be built)."""
    resolved, names = [], set()
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--model expects NAME=PATH, got '{spec}'")
        if name in names or name in ("base", "teacher", "GRACE-OFF"):
            raise SystemExit(f"model name '{name}' is duplicated or reserved (base/teacher/GRACE-OFF)")
        names.add(name)
        info = {"name": name, **detect_model(Path(path)), "build": None}
        if info["model_dir"] is None:
            info["model_dir"] = out / "models" / name
            if not (info["model_dir"] / "saved_model.pb").exists():
                tool = "apply_adapter.py" if info["kind"] == "adapter" else "export_checkpoint.py"
                info["build"] = [vpy, script(tool), info["source"], "--out", info["model_dir"]]
        resolved.append(info)
    return resolved


def validation_commands(vpy: str, models: list, organic: Path, materials: Path, out: Path,
                        paths: dict, workdir: Path, quick=False, no_base=False,
                        no_teacher=False) -> list:
    """[(label, cmd)] for validate-models: organic, materials, report."""
    organic_cmd = [vpy, script("validate_model.py"), "--trained",
                   *[f"{m['name']}={m['model_dir']}" for m in models],
                   "--structures", organic, "--n", 100 if quick else 0, "--out", out / "organic"]
    mat_models = ([] if no_base else [f"base={paths['STUDENT_MODEL_DIR']}"]) \
        + [f"{m['name']}={m['model_dir']}" for m in models] \
        + ([] if no_teacher else [f"GRACE-OFF={paths['TEACHER_MODEL_DIR']}"])
    materials_cmd = [vpy, script("validate_materials.py"), "--data", materials,
                     "--models", *mat_models, "--out", out / "materials"] \
        + (QUICK_SIZES["materials"] if quick else [])
    report_cmd = [vpy, script("report.py"), "--workdir", workdir, "--validation-only", out]
    return [("organic", organic_cmd), ("materials", materials_cmd), ("report", report_cmd)]


def adhoc_dir(settings: dict, now=None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M", time.localtime(now))
    return wpath(settings, "VALIDATION_DIR") / f"adhoc_{stamp}"


def cmd_validate_models(args):
    """Validate any models (no experiment loop needed): organic + materials + report."""
    settings = load_settings()
    require_configured(settings)
    vpy = vbin(settings, "python")
    workdir = Path(settings.get("workdir") or ROOT)
    out = Path(args.out).expanduser() if args.out else adhoc_dir(settings)
    organic = Path(args.organic).expanduser() if args.organic \
        else wpath(settings, "ORGANIC_DIR") / "test.pkl.gz"
    materials = Path(args.materials).expanduser() if args.materials \
        else wpath(settings, "MATERIALS_DIR") / "val.parquet"
    for need, flag in ((organic, "--organic"), (materials, "--materials")):
        if not need.exists():
            raise SystemExit(f"{need} not found: run `python3 pipeline.py prepare` first, "
                             f"or pass {flag} <file>")
    models = resolve_models(args.model, out, vpy)
    print(f"Validation folder: {out}{'  (quick: reduced sizes)' if args.quick else ''}")
    for m in models:
        action = "build once" if m["build"] else "use"
        print(f"  {m['name']:16s} {KIND_TEXT[m['kind']]}\n  {'':16s} {action}: {m['model_dir']}")
    if args.no_base or args.no_teacher:
        print("  (organic validation always includes base and teacher rows: the teacher is the "
              "reference; --no-base/--no-teacher apply to the materials validation)")
    commands = validation_commands(vpy, models, organic, materials, out, settings["paths"],
                                   workdir, args.quick, args.no_base, args.no_teacher)
    if args.dry_run:
        for m in models:
            if m["build"]:
                print("$ " + " ".join(map(str, m["build"])))
        for _, cmd in commands:
            print("$ " + " ".join(map(str, cmd)))
        return
    out.mkdir(parents=True, exist_ok=True)
    log = out / "validate.log"
    for m in models:
        if m["build"]:
            print(f"\n== Building {m['name']} ({m['kind']})")
            sh(m["build"], log=log)
    for label, cmd in commands:
        result_file = {"organic": out / "organic" / "validation.json",
                       "materials": out / "materials" / "materials_validation.json"}.get(label)
        if result_file and result_file.exists() and not args.force:
            print(f"\n== {label} validation: exists ({result_file.parent}), skipping (--force to redo)")
            continue
        print(f"\n== {label.capitalize()}" + (" validation" if label != "report" else ""))
        if label == "report":
            try:
                sh(cmd, log=log)
            except SystemExit:
                print("Note: the report step did not complete; the validation results are "
                      f"complete in {out}")
        else:
            sh(cmd, log=log)
    print(f"\nValidation results: {out}")


def validation_mode(settings: dict, quick=False, full=False, final=False) -> str:
    if quick and full:
        raise SystemExit("--quick and --full are mutually exclusive")
    if quick:
        return "quick"
    if full:
        return "full"
    return get_setting(settings, "validation.final" if final else "validation.mid_loop")


def cmd_validate_exp(args):
    """Organic + materials validation of one experiment (or the base/teacher reference)."""
    settings = load_settings()
    require_configured(settings)
    vpy = vbin(settings, "python")
    paths = settings["paths"]
    exp = args.exp
    if exp == "best":
        best = best_experiment(load_ledger(settings))
        if best is None:
            raise SystemExit("No finished experiment in the ledger yet.")
        exp = best["id"]
        print(f"Best experiment: {exp}")
    # --quick/--full win; otherwise settings validation.final (final calls: --final, base,
    # best) or validation.mid_loop (any other experiment id)
    final = getattr(args, "final", False) or args.exp in ("base", "best")
    mode = validation_mode(settings, getattr(args, "quick", False), getattr(args, "full", False),
                           final)
    args.quick = mode == "quick"
    print(f"Validation mode: {mode}")
    # quick validations go to their own folder so they never hide a full one
    out = wpath(settings, "VALIDATION_DIR") / (f"{exp}_quick" if args.quick else exp)
    organic_test = wpath(settings, "ORGANIC_DIR") / "test.pkl.gz"
    val_shard = wpath(settings, "MATERIALS_DIR") / "val.parquet"
    for need in (organic_test, val_shard):
        if not need.exists():
            raise SystemExit(f"{need} missing: run `python3 pipeline.py prepare` first")
    log = out / "validate.log"
    out.mkdir(parents=True, exist_ok=True)

    if exp == "base":
        organic_cmd = [vpy, script("validate_model.py"), "--reference-only"]
        models = [f"base={paths['STUDENT_MODEL_DIR']}", f"GRACE-OFF={paths['TEACHER_MODEL_DIR']}"]
    else:
        model = wpath(settings, "EXPERIMENTS_DIR") / exp / "run" / "saved_model"
        if not (model / "saved_model.pb").exists():
            raise SystemExit(f"No trained model at {model} (is {exp} finished? see `pipeline.py ledger`)")
        organic_cmd = [vpy, script("validate_model.py"), "--trained", f"{exp}={model}"]
        models = [f"base={paths['STUDENT_MODEL_DIR']}", f"{exp}={model}"]

    if (out / "organic" / "validation.json").exists() and not args.force:
        print(f"\n== Organic validation: exists ({out / 'organic'}), skipping (--force to redo)")
    else:
        print(f"\n== Organic validation ({exp})")
        sh(organic_cmd + ["--structures", organic_test, "--n", 100 if args.quick else 0,
                          "--out", out / "organic"], log=log)
    if (out / "materials" / "materials_validation.json").exists() and not args.force:
        print(f"\n== Materials validation: exists ({out / 'materials'}), skipping")
    else:
        print(f"\n== Materials validation ({exp}, groups A/B/C)")
        sh([vpy, script("validate_materials.py"), "--data", val_shard, "--models", *models,
            "--out", out / "materials"] + (QUICK_SIZES["materials"] if args.quick else []), log=log)
    print(f"\nValidation of {exp}: {out}")


def cmd_report(args):
    settings = load_settings()
    require_configured(settings)
    workdir = Path(settings.get("workdir") or ROOT)
    cmd = [vbin(settings, "python"), script("report.py"), "--workdir", workdir]
    if args.exps:
        cmd += ["--exps", *args.exps]
    if args.out:
        cmd += ["--out", args.out]
    sh(cmd)


# --------------------------------------------------------------------------------------
# wizard
# --------------------------------------------------------------------------------------
def cmd_wizard(args):
    print("GRACE LoRA distillation: guided setup\n")
    ensure_settings_confirmed()
    rec = cmd_sysinfo(argparse.Namespace(json=False))
    if rec["blockers"]:
        raise SystemExit("\nResolve the blockers above, then run the wizard again.")
    settings = load_settings()
    vpy = Path(vbin(settings, "python"))
    if not vpy.exists() or not inspect(settings, "versions")["ok"]:
        if confirm(f"\nCreate/repair the Python environment in {venv_dir(settings)} "
                   "(several GB download)?"):
            cmd_setup_env(argparse.Namespace(python=None, cpu_only=False, venv=None))
        else:
            raise SystemExit("An environment is required.")
    print()
    cmd_configure(argparse.Namespace(student=None, student_model=None, teacher=None, data=None,
                                     test=None, device=None, batch_size=None, max_buckets=None,
                                     workdir=None, yes=False))
    if confirm("\nDownload the chosen models and data now?"):
        cmd_fetch(argparse.Namespace())
    cmd_validate(argparse.Namespace())
    has_gpu = bool((load_settings().get("system") or {}).get("gpus"))
    race = has_gpu and confirm("\nAlso race the GPU against the CPU during the smoke test?",
                               default=True)
    if confirm("\nRun the smoke test now (~10-15 min)?"):
        cmd_smoke(argparse.Namespace(race=race))
    plan = ask("\nWhich plan to run? (pilot / quick / solid / none)", "none",
               choices=["pilot", "quick", "solid", "none"])
    if plan == "none":
        print("Later: python3 pipeline.py run --plan quick")
        return
    cmd_run(argparse.Namespace(plan=plan, yes=False, force=False, n_train=None, n_val=None,
                               n_test=None, epochs=None))


# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venv", help="use this venv instead of ./.venv (also env GRACE_PIPELINE_VENV)")
    sub = parser.add_subparsers(dest="cmd")  # no subcommand (e.g. "Run" in an editor) = auto

    p = sub.add_parser("sysinfo", help="detect hardware and recommend settings")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("setup-env", help="create .venv and install requirements")
    p.add_argument("--python", help="interpreter for the venv (default: recommended)")
    p.add_argument("--cpu-only", action="store_true", help="Linux: skip the CUDA TensorFlow bundle")
    sub.add_parser("catalog", help="list student/teacher models and datasets")
    p = sub.add_parser("configure", help="choose student, teacher, data -> pipeline.json")
    p.add_argument("--student", help=f"GRACE model name or checkpoint folder (default {DEFAULT_STUDENT})")
    p.add_argument("--student-model", help="SavedModel folder (only for a local student checkpoint)")
    p.add_argument("--teacher", help=f"GRACE-OFF variant, grace:<name> or SavedModel folder (default {DEFAULT_TEACHER})")
    p.add_argument("--data", help="'spice' or path to an extended-XYZ pool")
    p.add_argument("--test", help="optional held-out test .xyz for custom data")
    p.add_argument("--device", choices=list(DEVICE_FLAGS), help="training device")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-buckets", type=int)
    p.add_argument("--workdir", help="where data/ and runs/ go (default: project folder)")
    p.add_argument("--yes", action="store_true", help="non-interactive: use defaults for unset options")
    sub.add_parser("fetch", help="download models and data")
    sub.add_parser("validate", help="check files, versions and element coverage")
    p = sub.add_parser("smoke", help="smoke test + timing benchmark")
    p.add_argument("--race", action="store_true", help="also time 2 training epochs on the GPU")
    sub.add_parser("estimate", help="time estimates per plan")
    p = sub.add_parser("run", help="run a plan: dataset, training, evaluation, report")
    p.add_argument("--plan", choices=list(PLANS), default="quick")
    p.add_argument("--n-train", type=int)
    p.add_argument("--n-val", type=int)
    p.add_argument("--n-test", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--force", action="store_true", help="retrain jobs that already finished")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p = sub.add_parser("prepare", help="organic pool + MPtrj shards + monitor sets (idempotent)")
    p.add_argument("--n-organic-pool", type=int, default=2000,
                   help="teacher-labelled training molecules (the largest n_organic you will use)")
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--materials-from", help="folder with MPtrj shards already on disk (no download)")
    p.add_argument("--force", action="store_true", help="rebuild even if present")
    p = sub.add_parser("experiment", help="run ONE LoRA experiment and append it to the ledger")
    p.add_argument("--set", action="extend", nargs="+", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable (e.g. --set rank=16 --set n_materials=0)")
    p.add_argument("--rationale", default="", help="why this experiment (goes to the decision log)")
    p.add_argument("--parent", help="experiment id this one builds on")
    heal = p.add_mutually_exclusive_group()
    heal.add_argument("--self-heal", dest="self_heal", action="store_const", const=True,
                      default=None, help="retry failed runs automatically (default: agent option)")
    heal.add_argument("--no-self-heal", dest="self_heal", action="store_const", const=False,
                      help="never retry, even if the self_heal option is on")
    p.add_argument("--hypothesis", help="what this experiment is expected to show (predictions option)")
    p.add_argument("--predict", action="append", default=[], metavar="KEY=VALUE",
                   help=f"predicted monitor value, KEY in {', '.join(PREDICT_KEYS)} (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    p = sub.add_parser("ledger", help="show the experiment ledger")
    p.add_argument("--json", action="store_true")
    p.add_argument("--best", action="store_true", help="print the best experiment")
    p = sub.add_parser("validate-exp", help="organic + materials validation of an experiment")
    p.add_argument("--exp", required=True, help="experiment id, 'best', or 'base' (reference rows)")
    p.add_argument("--force", action="store_true", help="redo existing validations")
    p.add_argument("--quick", action="store_true",
                   help="reduced sizes (organic 100; materials A 50 / B 150 / C 60), into <id>_quick/")
    p.add_argument("--full", action="store_true", help="all structures (overrides the settings)")
    p.add_argument("--final", action="store_true",
                   help="use the settings' validation.final size (base and best are final anyway; "
                        "other ids use validation.mid_loop)")
    p = sub.add_parser(
        "validate-models",
        help="validate any models without the experiment loop (organic + materials + report)",
        description="PATH may be a SavedModel folder, a train_lora run folder, a LoRA adapter "
                    "folder (adapter.json + best_trainable_state.npz; built via apply_adapter.py) "
                    "or a checkpoint (<prefix>.index or a folder with best_checkpoint.index; "
                    "exported via export_checkpoint.py). Built models go to <out>/models/<NAME>.")
    p.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                   help="model to validate (repeatable)")
    p.add_argument("--no-base", action="store_true", help="materials: leave out base GRACE")
    p.add_argument("--no-teacher", action="store_true", help="materials: leave out GRACE-OFF")
    p.add_argument("--quick", action="store_true",
                   help="reduced sizes (organic 100; materials A 50 / B 150 / C 60)")
    p.add_argument("--organic", help="organic structures with DFT labels (.xyz/.pkl.gz; default: "
                                     "<workdir>/data/organic/test.pkl.gz)")
    p.add_argument("--materials", help="materials with DFT labels (.parquet/.xyz; default: "
                                       "<workdir>/data/materials/val.parquet)")
    p.add_argument("--out", help="output folder (default: <workdir>/validation/adhoc_<date_time>)")
    p.add_argument("--force", action="store_true", help="redo existing validations")
    p.add_argument("--dry-run", action="store_true", help="print what would run")
    p = sub.add_parser("settings", help="show / change / confirm ALL settings (training, quality, usage)")
    p.add_argument("--json", action="store_true", help="dump everything as JSON")
    p.add_argument("--set", action="extend", nargs="+", default=[], metavar="KEY=VALUE",
                   help="change a setting: dotted (experiment_defaults.rank=16) or short when "
                        "unambiguous (rank=16, critic=true)")
    p.add_argument("--reset", nargs="*", metavar="KEY",
                   help="back to defaults: the given keys, or all with no keys")
    p.add_argument("--explain", metavar="KEY", help="what a setting does, tested value, trade-off")
    p.add_argument("--confirm", action="store_true", help="mark the settings as reviewed")
    p = sub.add_parser("options", help="show / toggle the optional agent features (all off by default)")
    p.add_argument("--enable", nargs="+", default=[], metavar="NAME",
                   help=f"turn on: {', '.join(AGENT_OPTIONS)}")
    p.add_argument("--disable", nargs="+", default=[], metavar="NAME", help="turn off")
    p.add_argument("--set", action="extend", nargs="+", default=[], metavar="KEY=VALUE",
                   help="numeric option, e.g. max_subagents=2")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("cockpit", help="live HTML dashboard of the experiments (scripts/cockpit.py)")
    p.add_argument("--watch", type=int, metavar="SECONDS", help="keep refreshing every SECONDS")
    p.add_argument("--out", help="HTML file (default: <workdir>/report/cockpit.html)")
    p.add_argument("--artifact", action="store_true",
                   help="write a publishable artifact version to <workdir>/report/cockpit_artifact/")
    p = sub.add_parser("report", help="final report.md + figures")
    p.add_argument("--exps", nargs="*", help="experiments to include (default: all validated)")
    p.add_argument("--out", help="output folder (default: <workdir>/report)")
    sub.add_parser("auto", help="everything, non-interactive, from the YOUR CHOICES block (default)")
    sub.add_parser("wizard", help="interactive: everything above, asking every choice")

    args = parser.parse_args()
    if args.venv:
        os.environ["GRACE_PIPELINE_VENV"] = str(Path(args.venv).expanduser())
    handlers = {"sysinfo": cmd_sysinfo, "setup-env": cmd_setup_env, "catalog": cmd_catalog,
                "configure": cmd_configure, "fetch": cmd_fetch, "validate": cmd_validate,
                "smoke": cmd_smoke, "estimate": cmd_estimate, "run": cmd_run, "wizard": cmd_wizard,
                "prepare": cmd_prepare, "experiment": cmd_experiment, "ledger": cmd_ledger,
                "validate-exp": cmd_validate_exp, "validate-models": cmd_validate_models,
                "report": cmd_report, "options": cmd_options, "cockpit": cmd_cockpit,
                "settings": cmd_settings}
    handlers["auto"] = cmd_auto
    handlers[args.cmd or "auto"](args)


if __name__ == "__main__":
    main()
