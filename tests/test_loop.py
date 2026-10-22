"""Pure-Python tests for the autonomous loop (ledger.py, experiment.py).

No training, no TensorFlow. Run with pytest, or directly:
    python tests/test_loop.py
"""

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import experiment as X  # noqa: E402
import ledger as L  # noqa: E402

# isolate from a real pipeline.json in the repository root (objective lookup)
L.REPO = Path(tempfile.mkdtemp(prefix="test_loop_repo_"))
HISTORY_FIXTURE = Path(__file__).resolve().parent / "fixtures/runs/exp_003/history.csv"
BASE = {"organic_f": 300.0, "materials_f": 50.0, "only_adapted_f": 60.0}


def _tmp():
    return Path(tempfile.mkdtemp(prefix="test_loop_"))


def _done_entry(exp_id, organic, materials, only, ledger):
    entry = {"id": exp_id, "parent": None, "status": "done", "config": dict(L.DEFAULT_CONFIG),
             "metrics": {"organic_f": organic, "materials_f": materials, "only_adapted_f": only}}
    return L.evaluate_entry(entry, ledger)


# ---------------------------------------------------------------------------- ledger
def test_ledger_roundtrip_and_atomic_write():
    wd = _tmp()
    ledger = L.load(wd)
    assert ledger == L.new_ledger()
    ledger["base"] = dict(BASE)
    ledger["experiments"].append(_done_entry("exp_001", 100, 55, 100, ledger))
    path = L.save(wd, ledger)
    assert path == wd / "experiments" / "ledger.json"
    assert L.load(wd) == json.loads(path.read_text()) == ledger
    leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".ledger.")]
    assert leftovers == [], "temp files must be renamed away"


def test_next_id():
    ledger = L.new_ledger()
    assert L.next_id(ledger) == "exp_001"
    ledger["experiments"] = [{"id": "exp_001"}, {"id": "exp_007"}, {"id": "weird"}]
    assert L.next_id(ledger) == "exp_008"


def test_relative_metrics_and_feasibility():
    rel = L.relative_metrics({"organic_f": 100, "materials_f": 55, "only_adapted_f": 100}, BASE)
    assert abs(rel["organic_gain"] - (1 - 100 / 300)) < 1e-12
    assert abs(rel["materials_increase"] - 0.10) < 1e-12
    assert abs(rel["only_adapted_increase"] - (100 / 60 - 1)) < 1e-12
    obj = dict(L.DEFAULT_OBJECTIVE)
    assert L.is_feasible(rel, obj)                       # +10% <= 15%, +67% <= 75%
    assert not L.is_feasible({"materials_increase": 0.2, "only_adapted_increase": 0.1}, obj)
    assert not L.is_feasible({"materials_increase": 0.1, "only_adapted_increase": 0.9}, obj)
    assert L.is_feasible({}, obj)                        # missing monitors do not block


def test_best_selection_feasible_and_fallback():
    ledger = L.new_ledger()
    ledger["base"] = dict(BASE)
    ledger["experiments"] = [
        _done_entry("exp_001", 108, 125, 298, ledger),   # infeasible (+150% materials)
        _done_entry("exp_002", 106, 55, 100, ledger),    # feasible
        _done_entry("exp_003", 97, 54, 102, ledger),     # feasible, best score
        {"id": "exp_004", "status": "failed", "metrics": None, "score": None},
    ]
    assert L.best(ledger)["id"] == "exp_003"
    # no feasible experiment -> lowest materials increase among finished ones
    ledger["objective"]["max_materials_increase"] = 0.01
    L.rescore(ledger)
    assert not any(e.get("feasible") for e in ledger["experiments"])
    assert L.best(ledger)["id"] == "exp_003"            # +8% materials is the smallest
    assert L.best(L.new_ledger()) is None


def test_append_update_and_table():
    wd = _tmp()
    ledger = L.load(wd)
    ledger["base"] = dict(BASE)
    L.save(wd, ledger)
    L.append(wd, {"id": "exp_002", "status": "running", "config": {"rank": 16}})
    L.append(wd, {"id": "exp_001", "status": "running", "config": {}})
    L.append(wd, {"id": "exp_002", "status": "running", "config": {"rank": 16, "epochs": 5}})
    ledger = L.load(wd)
    assert [e["id"] for e in ledger["experiments"]] == ["exp_001", "exp_002"]
    assert ledger["experiments"][1]["config"]["epochs"] == 5
    L.update_entry(wd, "exp_001", status="done",
                   metrics={"organic_f": 100, "materials_f": 52, "only_adapted_f": 90})
    ledger = L.load(wd)
    L.evaluate_entry(L.get(ledger, "exp_001"), ledger)
    L.save(wd, ledger)
    table = L.format_table(L.load(wd))
    assert "exp_001 *" in table and "rank=16 epochs=5" in table and "+4%" in table


def test_ledger_cli_set_objective_rescores():
    wd = _tmp()
    ledger = L.new_ledger()
    ledger["base"] = dict(BASE)
    ledger["experiments"] = [_done_entry("exp_001", 100, 56, 80, ledger)]  # +12% materials
    L.save(wd, ledger)
    assert L.load(wd)["experiments"][0]["feasible"] is True
    with contextlib.redirect_stdout(io.StringIO()):
        L.main(["--workdir", str(wd), "set-objective", "max_materials_increase=0.10"])
    after = L.load(wd)
    assert after["objective"]["max_materials_increase"] == 0.10
    assert after["experiments"][0]["feasible"] is False
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        L.main(["--workdir", str(wd), "best"])
    assert json.loads(out.getvalue())["id"] == "exp_001"


# ---------------------------------------------------------------------------- experiment
def test_parse_history_real_fixture():
    metrics, base, selected = X.parse_history(HISTORY_FIXTURE)
    assert selected == 10
    assert abs(metrics["val_f"] - 83.81) < 0.01
    assert abs(metrics["organic_f"] - 93.0) < 0.1
    assert abs(metrics["materials_f"] - 54.6) < 0.1
    assert abs(metrics["only_adapted_f"] - 107.7) < 0.1
    assert abs(base["organic_f"] - 301.54) < 0.01
    assert abs(base["materials_f"] - 50.03) < 0.01
    assert abs(base["only_adapted_f"] - 62.70) < 0.01


def test_parse_history_without_monitors_and_epoch0_best():
    wd = _tmp()
    path = wd / "history.csv"
    path.write_text("epoch,train_loss,e_mae_meV_atom,f_mae_meV_A,time_s\n"
                    "0,,10.0,50.0,0.0\n1,0.1,11.0,60.0,10.0\n2,0.1,12.0,55.0,10.0\n")
    metrics, base, selected = X.parse_history(path)
    assert selected == 0 and metrics["val_f"] == 50.0       # nothing beat the base
    assert metrics["organic_f"] is None and base == {}


def test_parse_config_validation():
    config = X.parse_config(None)
    assert config == L.DEFAULT_CONFIG
    config = X.parse_config('{"rank": 16, "epochs": 5}')
    assert config["rank"] == 16 and config["n_organic"] == 1000
    organic_only = X.parse_config('{"n_materials": 0}')
    assert organic_only["n_materials_val"] == 0 and organic_only["n_only_adapted"] == 0
    for bad in ('{"rnak": 16}', '{"device": "tpu"}', '{"epochs": 0}', '{"rank": -1}',
                '{"n_materials": 100, "n_only_adapted": 200}', '[1, 2]'):
        try:
            X.parse_config(bad)
        except SystemExit as exc:
            if "rnak" in bad:
                assert "Valid keys" in str(exc)
        else:
            raise AssertionError(f"config {bad} should be rejected")
    wd = _tmp()
    (wd / "cfg.json").write_text('{"n_organic": 500}')
    assert X.parse_config(f"@{wd / 'cfg.json'}")["n_organic"] == 500


def test_metadata_elements_and_teacher_fallback():
    wd = _tmp()
    (wd / "metadata.yaml").write_text("chemical_symbols:\n- Br\n- C\n- H\ncutoff: 6.0\n")
    assert X.read_metadata_elements(wd) == ["Br", "C", "H"]
    flow = wd / "flow"
    flow.mkdir()
    (flow / "metadata.yaml").write_text("chemical_symbols: [O, 'N']\ncutoff: 6.0\n")
    assert X.read_metadata_elements(flow) == ["O", "N"]
    assert X.teacher_elements({"paths": {"TEACHER_MODEL_DIR": str(wd)}}) == ["Br", "C", "H"]
    assert X.teacher_elements({}) == X.DEFAULT_TEACHER_ELEMENTS


def test_build_commands():
    wd = _tmp()
    settings = {"paths": {"TEACHER_MODEL_DIR": "/t/saved_model", "STUDENT_MODEL_DIR": "/s",
                          "STUDENT_CHECKPOINT_DIR": "/s/ckpt"}}
    config = X.parse_config('{"rank": 16, "device": "gpu", "lr": 0.001}')
    data_cmd, train_cmd = X.build_commands(wd, wd / "experiments" / "exp_001", config,
                                           settings, "/venv/python", ["C", "H", "O"])
    assert data_cmd[0] == "/venv/python" and data_cmd[1].endswith("make_replay_dataset.py")
    for flag in ("--n-organic", "--n-materials", "--n-only-adapted", "--no-monitors",
                 "--exclude", "--adapted-from", "--base"):
        assert flag in data_cmd, flag
    assert data_cmd[data_cmd.index("--n-organic") + 1] == "1000"
    assert "--energy-reference" not in data_cmd          # no reference file yet -> fit offsets
    (wd / "data" / "monitors").mkdir(parents=True)
    (wd / "data" / "monitors" / "energy_reference.json").write_text("{}")
    data_cmd2, _ = X.build_commands(wd, wd / "experiments" / "exp_001", config, settings,
                                    "/venv/python", ["C", "H", "O"])
    assert data_cmd2[data_cmd2.index("--energy-reference") + 1] == \
        str(wd / "data" / "monitors" / "energy_reference.json")
    assert train_cmd[1].endswith("train_lora.py")
    assert train_cmd[train_cmd.index("--rank") + 1] == "16"
    assert "--gpu" in train_cmd and "--jit" not in train_cmd
    assert train_cmd[train_cmd.index("--lr") + 1] == "0.001"
    assert "--alpha" not in train_cmd                    # default alpha = 2*rank in train_lora
    for flag in ("--no-energy-shift", "--no-train-shift", "--checkpoint-dir"):
        assert flag in train_cmd, flag
    i = train_cmd.index("--adapt-elements")
    assert train_cmd[i + 1:i + 4] == ["C", "H", "O"]
    monitors = train_cmd[train_cmd.index("--monitor") + 1:]
    assert [m.split("=")[0] for m in monitors] == list(L.MONITORS)
    assert train_cmd[train_cmd.index("--runs-dir") + 1] == str(wd / "experiments" / "exp_001")
    assert train_cmd[train_cmd.index("--name") + 1] == "run"


def test_dry_run_writes_nothing():
    wd = _tmp()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = X.main(["--workdir", str(wd), "--config", '{"epochs": 3}',
                       "--rationale", "test", "--dry-run", "--python", "/venv/python"])
    text = out.getvalue()
    assert code == 0 and "exp_001" in text and "train_lora.py" in text and "--epochs 3" in text
    assert not (wd / "experiments").exists(), "dry-run must not create files"


def test_missing_prepared_data_is_a_clear_error():
    wd = _tmp()
    try:
        X.main(["--workdir", str(wd), "--rationale", "test", "--python", "/venv/python"])
    except SystemExit as exc:
        assert "pipeline.py prepare" in str(exc)
    else:
        raise AssertionError("must refuse without prepared data")


def test_mark_stale_running_entries():
    wd = _tmp()
    ledger = L.new_ledger()
    ledger["experiments"] = [
        {"id": "exp_001", "status": "running", "pid": 999_999_999, "host": socket.gethostname()},
        {"id": "exp_002", "status": "running", "pid": os.getpid(), "host": socket.gethostname()},
        {"id": "exp_003", "status": "done"},
    ]
    L.save(wd, ledger)
    assert X.mark_stale(wd) == ["exp_001"]
    status = {e["id"]: e["status"] for e in L.load(wd)["experiments"]}
    assert status == {"exp_001": "interrupted", "exp_002": "running", "exp_003": "done"}


def test_estimate_uses_benchmark_when_present():
    config = X.parse_config('{"n_organic": 1000, "n_materials": 600, "epochs": 10}')
    default = X.estimate_seconds(config, {})
    faster = X.estimate_seconds(config, {"benchmark": {"train_s_per_structure_epoch": 0.1}})
    assert abs((default - faster) - 1600 * 10 * 0.1) < 1e-6


# ---------------------------------------------------------------------------- round 3
HISTORY_OK = ("epoch,train_loss,e_mae_meV_atom,f_mae_meV_A,time_s,organic_f_mae_meV_A,"
              "organic_e_mae_meV_atom,materials_f_mae_meV_A,materials_e_mae_meV_atom,"
              "only_adapted_f_mae_meV_A,only_adapted_e_mae_meV_atom\n"
              "0,,14.5,221.0,0.0,300.0,15.0,50.0,16.0,60.0,25.0\n"
              "1,0.01,14.0,90.0,100.0,110.0,14.0,55.0,17.0,100.0,30.0\n")


def _prepared_workdir():
    wd = _tmp()
    for name in L.MONITORS:
        (wd / "data" / "monitors").mkdir(parents=True, exist_ok=True)
        (wd / "data" / "monitors" / f"{name}.pkl.gz").write_bytes(b"x")
    (wd / "data" / "organic").mkdir(parents=True)
    (wd / "data" / "organic" / "train.pkl.gz").write_bytes(b"x")
    return wd


def _mock_runs(script):
    """script: list of (exit code, log text) for the TRAINING calls; data calls succeed.
    A successful training call writes history.csv into the run folder."""
    calls = {"train": 0, "data": 0, "train_cmds": []}

    def fake(cmd, log_path):
        if cmd[1].endswith("make_replay_dataset.py"):
            calls["data"] += 1
            with open(log_path, "a") as fh:
                fh.write("data ok\n")
            return 0
        code, text = script[calls["train"]]
        calls["train"] += 1
        calls["train_cmds"].append(list(cmd))
        with open(log_path, "a") as fh:
            fh.write(text + "\n")
        if code == 0:
            run = Path(cmd[cmd.index("--runs-dir") + 1]) / "run"
            run.mkdir(parents=True, exist_ok=True)
            (run / "history.csv").write_text(HISTORY_OK)
        return code
    return fake, calls


def _run_main(argv, fake):
    real = X.run_streamed
    X.run_streamed = fake
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            code = X.main(argv)
    finally:
        X.run_streamed = real
    last = out.getvalue().strip().splitlines()[-1]
    assert last.startswith("RESULT:")
    return code, json.loads(last[len("RESULT:"):])


def test_classify_failure():
    assert X.classify_failure("epoch 1 ...\nKilled\n", -9) == "oom"
    assert X.classify_failure("ResourceExhaustedError: OOM when allocating tensor", 1) == "oom"
    assert X.classify_failure("nothing useful", 137) == "oom"
    assert X.classify_failure("NotFoundError: could not find registered platform with id: 0x1", 1) == "device"
    assert X.classify_failure("epoch   3  val E nan meV/atom  F nan meV/A  loss nan  12s", 1) == "nan"
    assert X.classify_failure("Prepared data missing (data/monitors/organic.pkl.gz)", 1) == "missing_data"
    assert X.classify_failure("FileNotFoundError: [Errno 2] No such file or directory: "
                              "'/w/data/organic/train.pkl.gz'", 1) == "missing_data"
    assert X.classify_failure("ValueError: something else", 1) == "unknown"


def test_heal_config_changes():
    cfg = X.parse_config(None)                                   # batch 8, buckets 4, lr None
    new, change = X.heal_config(cfg, "oom")
    assert change == {"batch_size": [8, 4], "max_buckets": [4, 3]}
    new, change = X.heal_config(new, "oom")
    assert new["batch_size"] == 2 and new["max_buckets"] == 2
    assert X.heal_config(new, "oom") is None                     # already at the minimum
    gpu = dict(cfg, device="gpu")
    assert X.heal_config(gpu, "device")[1] == {"device": ["gpu", "cpu-jit"]}
    assert X.heal_config(cfg, "device", "could not find registered platform")[1] == \
        {"device": ["cpu-jit", "cpu"]}
    assert X.heal_config(cfg, "device", "some CUDA_ERROR") is None   # not XLA-related
    assert X.heal_config(dict(cfg, device="cpu"), "device", "XLA") is None
    assert X.heal_config(cfg, "nan")[1] == {"lr": [None, 1e-3]}
    assert X.heal_config(dict(cfg, lr=1e-3), "nan")[1] == {"lr": [1e-3, 5e-4]}
    assert X.heal_config(cfg, "missing_data") is None and X.heal_config(cfg, "unknown") is None


def test_self_heal_fail_then_succeed_with_predictions():
    wd = _prepared_workdir()
    fake, calls = _mock_runs([(137, "Killed"), (0, "epoch 1 ok")])
    code, entry = _run_main(["--workdir", str(wd), "--rationale", "heal test", "--self-heal",
                             "--hypothesis", "replay keeps materials", "--predict", "organic_f=100",
                             "--predict", "materials_f=55", "--python", "/venv/python"], fake)
    assert code == 0 and entry["status"] == "done" and calls["train"] == 2 and calls["data"] == 1
    assert len(entry["healing"]) == 1
    h = entry["healing"][0]
    assert h["attempt"] == 1 and h["error_class"] == "oom"
    assert h["change"] == {"batch_size": [8, 4], "max_buckets": [4, 3]}
    assert entry["config"]["batch_size"] == 4                       # final config = the one that ran
    second = calls["train_cmds"][1]
    assert second[second.index("--batch-size") + 1] == "4"
    assert entry["hypothesis"] == "replay keeps materials"
    assert entry["prediction"] == {"organic_f": 100.0, "materials_f": 55.0}
    pc = entry["prediction_check"]
    assert pc["organic_f"]["actual"] == 110.0 and abs(pc["organic_f"]["rel_error"] - 10 / 110) < 1e-12
    assert pc["materials_f"]["error"] == 0.0
    assert abs(entry["prediction_score"] - (10 / 110) / 2) < 1e-12
    stored = L.get(L.load(wd), "exp_001")
    assert stored["healing"] == entry["healing"] and stored["config"]["batch_size"] == 4


def test_no_self_heal_means_no_retry():
    wd = _prepared_workdir()
    fake, calls = _mock_runs([(137, "Killed"), (0, "never reached")])
    code, entry = _run_main(["--workdir", str(wd), "--rationale", "x", "--python", "/p"], fake)
    assert code == 1 and entry["status"] == "failed" and calls["train"] == 1
    assert "healing" not in entry and "oom" in entry["error"]


def test_self_heal_stops_on_missing_data_and_after_max_retries():
    wd = _prepared_workdir()
    fake, calls = _mock_runs([(1, "FileNotFoundError: No such file or directory: "
                                  "'/w/data/monitors/materials.pkl.gz'")])
    code, entry = _run_main(["--workdir", str(wd), "--rationale", "x", "--self-heal",
                             "--python", "/p"], fake)
    assert code == 1 and calls["train"] == 1 and entry["healing"] == []
    assert "pipeline.py prepare" in entry["error"]
    wd = _prepared_workdir()
    fake, calls = _mock_runs([(1, "loss nan")] * 5)
    code, entry = _run_main(["--workdir", str(wd), "--rationale", "x", "--self-heal",
                             "--max-retries", "2", "--python", "/p"], fake)
    assert code == 1 and calls["train"] == 3 and len(entry["healing"]) == 2
    assert [h["change"]["lr"] for h in entry["healing"]] == [[None, 1e-3], [1e-3, 5e-4]]


def test_predict_validation_and_prediction_check():
    assert X.parse_predictions(["organic_f=95", "only_adapted_f=110.5"]) == \
        {"organic_f": 95.0, "only_adapted_f": 110.5}
    for bad in (["val_f=10"], ["organic_f=abc"]):
        try:
            X.parse_predictions(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(bad)
    check, score = L.prediction_check({"organic_f": 100, "materials_f": 50},
                                      {"organic_f": 80.0, "materials_f": 50.0})
    assert check["organic_f"]["error"] == -20.0 and check["organic_f"]["rel_error"] == -0.25
    assert score == 0.125
    assert L.prediction_check({"organic_f": 1}, {"materials_f": 2}) == ({}, None)


def test_annotate_review_and_tables():
    wd = _tmp()
    ledger = L.new_ledger()
    ledger["base"] = dict(BASE)
    entry = _done_entry("exp_001", 97, 54, 100, ledger)
    entry.update(hypothesis="more C-type data fixes group C",
                 prediction={"organic_f": 100.0}, healing=[
                     {"attempt": 1, "error_class": "oom", "change": {"batch_size": [8, 4]},
                      "time": "2026-10-23 10:00"}])
    entry["prediction_check"], entry["prediction_score"] = L.prediction_check(
        entry["prediction"], entry["metrics"])
    ledger["experiments"] = [entry]
    L.save(wd, ledger)
    with contextlib.redirect_stdout(io.StringIO()):
        L.main(["--workdir", str(wd), "annotate", "exp_001", "review",
                "gain consistent with v2; plausible", "--verdict", "accept"])
        L.main(["--workdir", str(wd), "annotate", "exp_001", "notes", "free note"])
    e = L.get(L.load(wd), "exp_001")
    assert e["review"][0]["verdict"] == "accept" and e["notes"][0]["verdict"] is None
    table = L.format_table(L.load(wd))
    assert "exp_001 *✎" in table and "healed: oom" in table and "review accept" in table
    assert "pred 3%" in table
    preds = L.format_predictions(L.load(wd))
    assert "organic_f" in preds and "100.0" in preds and "97.0" in preds
    for bad in (["annotate", "exp_009", "review", "x"],
                ["annotate", "exp_001", "status", "x"]):                # unknown id / non-list field
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                L.main(["--workdir", str(wd)] + bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(bad)


# ---------------------------------------------------------------------------- objective config
def _write_objective(wd, **objective):
    (wd / "pipeline.json").write_text(json.dumps({"objective": objective, "paths": {}}))


def test_new_ledger_takes_objective_from_pipeline_json():
    wd = _tmp()
    assert L.load(wd)["objective"] == L.DEFAULT_OBJECTIVE          # nothing configured
    _write_objective(wd, max_materials_increase=0.05, unknown_key=1)
    obj = L.load(wd)["objective"]
    assert obj["max_materials_increase"] == 0.05
    assert obj["max_only_adapted_increase"] == L.DEFAULT_OBJECTIVE["max_only_adapted_increase"]
    assert "unknown_key" not in obj and obj["primary"] == "organic_f"


def test_sync_objective_rescores_and_prints_once():
    wd = _tmp()
    ledger = L.new_ledger()
    ledger["base"] = dict(BASE)
    ledger["experiments"] = [_done_entry("exp_001", 100, 55, 80, ledger),   # +10% materials
                             _done_entry("exp_002", 120, 51, 70, ledger)]   # +2% materials
    L.save(wd, ledger)
    assert L.best(L.load(wd))["id"] == "exp_001"
    _write_objective(wd, max_materials_increase=0.05)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        synced = L.sync_objective(wd)
    assert "Objective updated from pipeline.json: max_materials_increase 0.15 -> 0.05" in out.getvalue()
    stored = L.load(wd)
    assert stored["objective"]["max_materials_increase"] == 0.05 == synced["objective"]["max_materials_increase"]
    assert L.get(stored, "exp_001")["feasible"] is False and L.best(stored)["id"] == "exp_002"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        L.sync_objective(wd)
    assert out.getvalue() == ""                                       # already in sync


def test_experiment_start_syncs_objective():
    wd = _prepared_workdir()
    _write_objective(wd, max_materials_increase=0.05)
    L.save(wd, L.new_ledger())                                       # stored defaults (0.15)
    fake, _ = _mock_runs([(0, "ok")])
    real = X.run_streamed
    X.run_streamed = fake
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            code = X.main(["--workdir", str(wd), "--rationale", "x", "--python", "/p"])
    finally:
        X.run_streamed = real
    assert code == 0 and "Objective updated from pipeline.json" in out.getvalue()
    stored = L.load(wd)
    assert stored["objective"]["max_materials_increase"] == 0.05
    assert L.get(stored, "exp_001")["feasible"] is False             # +10% > 5%


def test_set_objective_still_works_and_warns_when_configured():
    wd = _tmp()
    L.save(wd, L.new_ledger())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        L.main(["--workdir", str(wd), "set-objective", "max_only_adapted_increase=0.5"])
    assert L.load(wd)["objective"]["max_only_adapted_increase"] == 0.5 and "NOTE" not in out.getvalue()
    _write_objective(wd, max_materials_increase=0.1)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        L.main(["--workdir", str(wd), "set-objective", "max_only_adapted_increase=0.6"])
    assert L.load(wd)["objective"]["max_only_adapted_increase"] == 0.6
    assert "pipeline.json also sets an objective" in out.getvalue()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
