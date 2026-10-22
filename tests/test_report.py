"""Tests for scripts/report.py and the plot modules (no TensorFlow, no training).

A fake <workdir> is assembled from tests/fixtures: real run histories and validation outputs
of the reference runs, cut down to a few structures and stripped of local paths.

    python -m pytest tests/test_report.py -q
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import report  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
RUNS = {exp_id: FIXTURES / "runs" / exp_id for exp_id in ("exp_001", "exp_002", "exp_003")}
VALIDATION_KEYS = ("base", "exp_002", "exp_003")  # fixtures/validation/<key>/{organic,materials}

BASE = {"organic_f": 301.5, "materials_f": 50.0, "only_adapted_f": 62.7}
OBJECTIVE = {"primary": "organic_f", "max_materials_increase": 0.15, "max_only_adapted_increase": 0.75}
CONFIG = {"n_organic": 500, "n_materials": 0, "n_materials_val": 100, "n_only_adapted": 0, "rank": 8,
          "alpha": None, "epochs": 10, "batch_size": 8, "max_buckets": 4, "lr": None, "max_atoms": 30,
          "oversample": 3.0, "seed": 0, "device": "cpu-jit", "adapt_elements": None}
EXPERIMENTS = [
    ("exp_001", None, {},
     "Baseline: plain LoRA on 500 teacher-labelled molecules, no replay, to measure gain and forgetting.",
     {"organic_f": 108.0, "materials_f": 125.0, "only_adapted_f": 298.6}, 1080),
    ("exp_002", "exp_001", {"n_materials": 500},
     "Materials monitor degraded 2.5x with H/O/F worst: add replay of 500 MPtrj structures that "
     "contain adapted elements, H/F oversampled.",
     {"organic_f": 101.6, "materials_f": 55.4, "only_adapted_f": 135.4}, 2100),
    ("exp_003", "exp_002", {"n_organic": 1000, "n_materials": 600, "n_only_adapted": 170},
     "Materials recovered but only-adapted crystals are 2.2x base: add 170 only-adapted crystals and "
     "double the organic data.",
     {"organic_f": 93.0, "materials_f": 54.6, "only_adapted_f": 107.7}, 3300),
]


def make_entry(exp_id, parent, delta, rationale, metrics, seconds, with_feasible=True):
    cfg = {**CONFIG, **delta}
    entry = {"id": exp_id, "parent": parent, "time": "2026-10-23 10:00", "rationale": rationale,
             "config": cfg, "status": "done", "run_dir": f"experiments/{exp_id}/run",
             "duration_s": seconds, "selected_epoch": 10, "metrics": metrics,
             "relative": report.relative(metrics, BASE), "score": metrics["organic_f"], "error": None}
    if with_feasible:
        entry["feasible"] = report.is_feasible({"metrics": metrics}, BASE, OBJECTIVE)
    return entry


def fake_workdir(tmp, with_validation=True):
    wd = Path(tmp) / "workdir"
    for exp_id, src in RUNS.items():
        run = wd / "experiments" / exp_id / "run"
        run.mkdir(parents=True)
        for f in ("history.csv", "summary.json", "args.json", "adapter.json"):
            if (src / f).exists():
                shutil.copy(src / f, run / f)
    ledger = {"version": 1, "objective": OBJECTIVE, "base": BASE,
              "experiments": [make_entry(*e) for e in EXPERIMENTS]}
    (wd / "experiments" / "ledger.json").write_text(json.dumps(ledger, indent=2))
    if with_validation:
        for exp_id in VALIDATION_KEYS:
            for kind in ("organic", "materials"):
                src = FIXTURES / "validation" / exp_id / kind
                dst = wd / "validation" / exp_id / kind
                dst.mkdir(parents=True)
                for f in src.glob("*"):
                    if f.suffix in (".json", ".npz"):
                        shutil.copy(f, dst / f.name)
    return wd


def run_report(wd, *extra):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "report.py"), "--workdir", str(wd), *extra],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


# ---- pure-logic tests -------------------------------------------------------------------------
def test_feasibility_and_best():
    entries = [make_entry(*e, with_feasible=False) for e in EXPERIMENTS]
    assert [report.is_feasible(e, BASE, OBJECTIVE) for e in entries] == [False, False, True]
    assert report.pick_best(entries, BASE, OBJECTIVE)["id"] == "exp_003"
    # nothing feasible -> least materials degradation
    tight = {**OBJECTIVE, "max_materials_increase": 0.01}
    assert report.pick_best(entries, BASE, tight)["id"] == "exp_003"


def test_config_delta():
    a, b = {**CONFIG}, {**CONFIG, "n_materials": 600, "rank": 16}
    assert report.config_delta(b, a) == "n_materials 0→600; rank 8→16"
    assert report.config_delta(a, a) == "same config"
    assert "no replay" in report.config_delta(a, None)


def test_minimal_report_without_ledger(tmp_path):
    run_report(tmp_path)
    assert "No experiments" in (tmp_path / "report" / "report.md").read_text()


# ---- end-to-end on real fixtures --------------------------------------------------------------
def test_full_report(tmp_path):
    wd = fake_workdir(tmp_path)
    adhoc = wd / "validation" / "adhoc_checkpoints"
    adhoc.mkdir(parents=True)
    (adhoc / "summary.md").write_text("# Checkpoint comparison\n")
    stdout = run_report(wd)
    md = (wd / "report" / "report.md").read_text()
    assert "## Additional validations" in md
    assert "[adhoc_checkpoints](../validation/adhoc_checkpoints/summary.md): Checkpoint comparison" in md
    assert "Best experiment: `exp_003`" in md
    for section in ("## Summary", "## Experiment timeline", "## Decision log", "## Figures", "## Limitations"):
        assert section in md
    expected = {"heatmap_force_mae.png", "tradeoff.png", "monitor_curves.png", "training_history.png",
                "best_parity_forces.png", "best_parity_forces_vs_dft.png", "best_parity_materials.png",
                "pareto.png"}
    written = {p.name for p in (wd / "report" / "figures").glob("*.png")}
    assert expected <= written, f"missing: {expected - written}\n{stdout}"
    # every image link is relative and points to an existing file
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md):
        assert not link.startswith("/") and (wd / "report" / link).exists(), link
    assert "group C" in md and "teacher" in md  # auto-generated limitations


def fake_validation_out(tmp, organic=True, materials=True):
    """<out>/organic + <out>/materials as written by `pipeline.py validate-models`."""
    out = Path(tmp) / "adhoc_compare"
    for kind, src, on in (("organic", FIXTURES / "validation/base/organic", organic),
                          ("materials", FIXTURES / "validation/base/materials", materials)):
        if on:
            (out / kind).mkdir(parents=True)
            for f in src.glob("*"):
                if f.suffix in (".json", ".npz"):
                    shutil.copy(f, out / kind / f.name)
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_validation_only(tmp_path):
    out = fake_validation_out(tmp_path)
    stdout = run_report(tmp_path, "--validation-only", str(out), "--title", "Checkpoint comparison")
    md = (out / "summary.md").read_text()
    assert md.startswith("# Checkpoint comparison")
    for section in ("## Data", "## Results", "## Verdict", "## Figures"):
        assert section in md
    for model in ("base", "smoke test", "C10", "C11", "GRACE-OFF"):
        assert f"| {model} |" in md or f"| **{model}** |" in md
    assert "better than base on organic molecules" in md
    assert "cannot run on 98.3% of materials" in md
    written = {p.name for p in (out / "figures").glob("*.png")}
    assert {"heatmap_force_mae.png", "parity_forces.png", "parity_forces_vs_dft.png",
            "parity_materials.png"} <= written, stdout
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md):
        assert (out / link).exists(), link


def test_validation_only_partial(tmp_path):
    out = fake_validation_out(tmp_path, organic=False)
    run_report(tmp_path, "--validation-only", str(out))
    md = (out / "summary.md").read_text()
    assert "## Verdict" in md and "Organic:" not in md
    written = {p.name for p in (out / "figures").glob("*.png")}
    assert {"heatmap_force_mae.png", "parity_materials.png"} <= written
    assert "parity_forces.png" not in written


def test_validation_only_empty(tmp_path):
    run_report(tmp_path, "--validation-only", str(tmp_path / "nothing"))
    assert "No validation outputs" in (tmp_path / "nothing" / "summary.md").read_text()


def test_report_without_validation(tmp_path):
    wd = fake_workdir(tmp_path, with_validation=False)
    run_report(wd, "--exps", "exp_002", "exp_003")
    md = (wd / "report" / "report.md").read_text()
    timeline = md.split("## Experiment timeline")[1].split("## Decision log")[0]
    assert not re.search(r"^\| (\*\*)?exp_001", timeline, re.M)  # filtered out (still a parent)
    written = {p.name for p in (wd / "report" / "figures").glob("*.png")}
    assert {"tradeoff.png", "monitor_curves.png"} <= written
    assert "Not fully validated" in md


# ---- round 3: cockpit + critic / healing / prediction fields ---------------------------------
def add_agent_fields(wd):
    """Add hypothesis/prediction/review/healing fields and a RUNNING exp_004 (partial history)."""
    import time as _time
    path = wd / "experiments" / "ledger.json"
    ledger = json.loads(path.read_text())
    e2, e3 = ledger["experiments"][1], ledger["experiments"][2]
    e2["hypothesis"] = "Replaying materials that contain H/O/F keeps their weight rows close to base."
    e2["prediction"] = {"organic_f": 105.0, "materials_f": 60.0}
    e2["prediction_check"] = {"organic_f": {"predicted": 105.0, "actual": 101.6, "error": -3.4, "rel_error": -0.032},
                              "materials_f": {"predicted": 60.0, "actual": 55.4, "error": -4.6, "rel_error": -0.077}}
    e2["prediction_score"] = 0.95
    e2["review"] = [{"time": "2026-10-23 10:40", "verdict": "approve",
                     "text": "Sound: replay targets exactly the elements that degraded."}]
    e3["review"] = [{"time": "2026-10-23 11:20", "verdict": "concern",
                     "text": "Two changes at once (more organics + only-adapted quota) blur attribution."}]
    e3["healing"] = [{"attempt": 1, "error_class": "OutOfMemory", "change": {"batch_size": [16, 8]},
                      "time": "2026-10-23 11:05"}]
    run = wd / "experiments" / "exp_004" / "run"
    run.mkdir(parents=True)
    lines = (RUNS["exp_003"] / "history.csv").read_text().splitlines()
    (run / "history.csv").write_text("\n".join(lines[:5]) + "\n")  # epochs 0-3 of 10
    ledger["experiments"].append({
        "id": "exp_004", "parent": "exp_003", "time": _time.strftime("%Y-%m-%d %H:%M"),
        "rationale": "Organics still improving at the end: try rank 16 with the same data.",
        "hypothesis": "Rank is the bottleneck for organics now that replay protects materials.",
        "prediction": {"organic_f": 88.0, "materials_f": 55.0},
        "config": {**CONFIG, "n_organic": 1000, "n_materials": 600, "n_only_adapted": 170, "rank": 16},
        "status": "running", "run_dir": "experiments/exp_004/run", "duration_s": None,
        "healing": [{"attempt": 1, "error_class": "DataBuildTimeout", "change": {"max_atoms": [30, 24]},
                     "time": _time.strftime("%Y-%m-%d %H:%M")}]})
    path.write_text(json.dumps(ledger, indent=2))
    return wd


def run_cockpit(wd, *extra):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "cockpit.py"), "--workdir", str(wd), *extra],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return wd / "report" / "cockpit.html"


def test_cockpit_live(tmp_path):
    wd = add_agent_fields(fake_workdir(tmp_path, with_validation=False))
    page = run_cockpit(wd).read_text()
    assert not re.search(r"""(src|href)\s*=\s*["']?https?://""", page), "external resource in cockpit"
    assert page.count("data:image/png;base64,") >= 2
    assert 'http-equiv="refresh"' in page and "prefers-color-scheme:dark" in page
    for text in ("Running now", "exp_004", "epoch 3/10", "Rank is the bottleneck", "critic: approve",
                 "self-healing attempt", "OutOfMemory", "Predicted vs actual", "Ledger", "Decision log"):
        assert text in page, text


def test_cockpit_empty(tmp_path):
    page = run_cockpit(tmp_path).read_text()
    assert "No experiment started yet" in page and "<html" in page


def test_report_agent_sections(tmp_path):
    wd = add_agent_fields(fake_workdir(tmp_path))
    run_report(wd)
    md = (wd / "report" / "report.md").read_text()
    for text in ("## Predictions vs outcomes", "Mean |relative error|", "## Self-healing events",
                 "`OutOfMemory`", "batch_size 16→8", "**Critic (approve):**", "_Hypothesis:_"):
        assert text in md, text
