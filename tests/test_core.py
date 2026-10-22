"""Core pipeline tests: no models, no TensorFlow inference.

    python -m pytest tests/test_core.py -q
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
import fetch_materials  # noqa: E402
import make_replay_dataset as mrd  # noqa: E402


def load_pipeline():
    spec = importlib.util.spec_from_file_location("pipeline", ROOT / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_pipeline()


# ---------------------------------------------------------------- pipeline.py helpers
def test_parse_sets_types():
    cfg = pipeline.parse_sets(["rank=16", "lr=2e-3", "replay=true", "alpha=null",
                               "device=cpu-jit", 'adapt_elements=["C","H"]', "flag=False",
                               "name=\"quoted\"", "n_materials=0"])
    assert cfg == {"rank": 16, "lr": 0.002, "replay": True, "alpha": None, "device": "cpu-jit",
                   "adapt_elements": ["C", "H"], "flag": False, "name": "quoted",
                   "n_materials": 0}


def test_parse_sets_rejects_missing_equals():
    with pytest.raises(SystemExit):
        pipeline.parse_sets(["rank16"])


def test_parse_sets_value_with_equals():
    assert pipeline.parse_sets(["note=a=b"]) == {"note": "a=b"}


def test_layout_and_wpath(tmp_path):
    paths = pipeline.layout_paths(tmp_path)
    assert paths["ORGANIC_DIR"] == str(tmp_path / "data/organic")
    assert paths["MONITORS_DIR"] == str(tmp_path / "data/monitors")
    settings = {"workdir": str(tmp_path), "paths": {"REPORT_DIR": "/custom/report"}}
    assert pipeline.wpath(settings, "EXPERIMENTS_DIR") == tmp_path / "experiments"
    assert pipeline.wpath(settings, "REPORT_DIR") == Path("/custom/report")


def test_best_experiment_rules():
    ledger = {"experiments": [
        {"id": "exp_001", "status": "done", "feasible": False, "score": 90.0,
         "relative": {"materials_increase": 1.5}},
        {"id": "exp_002", "status": "done", "feasible": True, "score": 105.0,
         "relative": {"materials_increase": 0.1}},
        {"id": "exp_003", "status": "done", "feasible": True, "score": 97.0,
         "relative": {"materials_increase": 0.12}},
        {"id": "exp_004", "status": "failed", "feasible": True, "score": 10.0},
    ]}
    assert pipeline.best_experiment(ledger)["id"] == "exp_003"
    none_feasible = {"experiments": [
        {"id": "a", "status": "done", "feasible": False, "score": 1, "relative": {"materials_increase": 0.9}},
        {"id": "b", "status": "done", "feasible": False, "score": 2, "relative": {"materials_increase": 0.3}}]}
    assert pipeline.best_experiment(none_feasible)["id"] == "b"
    assert pipeline.best_experiment({"experiments": []}) is None


# ---------------------------------------------------------------- config.py
def test_config_layout_defaults():
    p = config.paths()
    w = config.workdir()
    assert p["ORGANIC_DIR"] == Path(config.ORGANIC_DIR).expanduser()
    if not config.PIPELINE_SETTINGS.exists():
        assert w == config.PROJECT_ROOT
        assert p["MONITORS_DIR"] == w / "data/monitors"
        assert p["EXPERIMENTS_DIR"] == w / "experiments"
    assert set(config.monitor_paths()) == {"organic", "materials", "only_adapted"}
    assert config.materials_shard("train0").name == "train0.parquet"


def test_require_output_dirs_not_checked(tmp_path):
    assert config.require(str(tmp_path / "nope"), "EXPERIMENTS_DIR") == tmp_path / "nope"
    with pytest.raises(SystemExit):
        config.require(str(tmp_path / "nope"), "TEACHER_MODEL_DIR")


@pytest.mark.parametrize("text, expected", [
    ("chemical_symbols:\n- Br\n- C\n- H\ncutoff: 6.0\n", ["Br", "C", "H"]),
    ("chemical_symbols: [O, H, 'N']\ncutoff: 6.0\n", ["H", "N", "O"]),
    ("cutoff: 6.0\nchemical_symbols:\n  - S\n  - P\nhas_uq: true\n", ["P", "S"]),
])
def test_model_elements(tmp_path, text, expected):
    (tmp_path / "metadata.yaml").write_text(text)
    assert config.model_elements(tmp_path) == expected
    assert config.teacher_elements(tmp_path) == expected


def test_model_elements_missing(tmp_path):
    assert config.model_elements(tmp_path) == []


# ---------------------------------------------------------------- fetch_materials.py
def _pin_fake_shards(monkeypatch, contents):
    """Pin the fake shard contents, so the real checksum check runs on them."""
    import hashlib
    import checksums
    pins = dict(checksums.FILES)
    for local, remote, data in contents:
        pins[local] = pins[remote] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(checksums, "FILES", pins)


def test_fetch_from_dir_link_copy_and_skip(tmp_path, monkeypatch):
    _pin_fake_shards(monkeypatch, [("val.parquet", "val-00000-of-00001.parquet", b"val"),
                                   ("test.parquet", "test-00000-of-00001.parquet", b"test")])
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "val.parquet").write_bytes(b"val")
    (src / "test-00000-of-00001.parquet").write_bytes(b"test")  # Hugging Face name
    monkeypatch.setattr(fetch_materials, "download",
                        lambda url, dest: pytest.fail(f"unexpected download {url}"))
    res = fetch_materials.fetch(out, ["val", "test"], src, copy=False)
    assert res["val"].is_symlink() and res["val"].read_bytes() == b"val"
    assert res["test"].name == "test.parquet" and res["test"].read_bytes() == b"test"
    # existing files are kept (no relink, no download)
    res2 = fetch_materials.fetch(out, ["val"], None)
    assert res2["val"] == res["val"]
    out2 = tmp_path / "out2"
    res3 = fetch_materials.fetch(out2, ["val"], src, copy=True)
    assert not res3["val"].is_symlink() and res3["val"].read_bytes() == b"val"


def test_fetch_downloads_missing(tmp_path, monkeypatch):
    _pin_fake_shards(monkeypatch, [("train0.parquet", "train-00000-of-00008.parquet", b"x")])
    calls = []
    monkeypatch.setattr(fetch_materials, "download",
                        lambda url, dest: (calls.append(url), dest.write_bytes(b"x")))
    fetch_materials.fetch(tmp_path, ["train0"], tmp_path / "empty")
    assert calls and calls[0].endswith("train-00000-of-00008.parquet")
    assert (tmp_path / "train0.parquet").exists()
    assert "/resolve/main/" not in calls[0]              # pinned revision, not a moving branch


def test_fetch_rejects_wrong_content(tmp_path, monkeypatch):
    import checksums
    monkeypatch.delenv(checksums.SKIP_ENV, raising=False)
    src = tmp_path / "src"
    src.mkdir()
    (src / "val.parquet").write_bytes(b"tampered")
    with pytest.raises(checksums.ChecksumError):
        fetch_materials.fetch(tmp_path / "out", ["val"], src, copy=True)


# ---------------------------------------------------------------- make_replay_dataset.py
ADAPTED = {"Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"}


def fake_materials():
    """Synthetic MPtrj-like rows: (mp_id, atomic numbers)."""
    rows = [("mp-1", [26, 8, 8]), ("mp-1", [26, 8, 8]),     # Fe2O... duplicate frame
            ("mp-2", [14, 14]),                             # Si: no adapted element
            ("mp-3", [6, 6, 6, 6]),                          # C only (only-adapted)
            ("mp-4", [6, 1, 1, 1, 1]),                       # CH4 crystal (only-adapted, H)
            ("mp-5", [3, 9]),                                # LiF (adapted + other, F)
            ("mp-6", [11, 17]),                              # NaCl
            ("mp-7", [12, 8]),                               # MgO
            ("mp-8", [8] * 40),                              # too large
            ("mp-9", [7, 1, 1, 1]),                          # NH3 (only-adapted, H)
            ("mp-10", [29, 1])]                              # CuH (other + H)
    return pd.DataFrame({"mp_id": [r[0] for r in rows], "numbers": [r[1] for r in rows],
                         "num_atoms": [len(r[1]) for r in rows]})


def test_eligible_materials_filters_and_dedupes():
    mats, els = mrd.eligible_materials(fake_materials(), ADAPTED, max_atoms=30,
                                       excluded_ids={"mp-7"}, seed=0)
    ids = set(mats.mp_id)
    assert "mp-2" not in ids          # no adapted element
    assert "mp-8" not in ids          # too many atoms
    assert "mp-7" not in ids          # excluded (validation shard)
    assert list(mats.mp_id).count("mp-1") == 1   # one frame per material
    assert ids == {"mp-1", "mp-3", "mp-4", "mp-5", "mp-6", "mp-9", "mp-10"}
    assert len(els) == len(mats)


def test_select_materials_quota_and_disjoint():
    mats, els = mrd.eligible_materials(fake_materials(), ADAPTED, 30, set(), 0)
    rng = np.random.default_rng(0)
    sel = mrd.select_materials(els, ADAPTED, n_materials=4, n_materials_val=2, n_only_adapted=2,
                               oversample=3.0, rng=rng)
    only = els.map(lambda s: s <= ADAPTED).values
    assert sel["only_adapted_train"] == 2
    assert all(only[i] for i in sel["train"][:2])            # quota first
    assert len(sel["train"]) == 4
    assert not set(sel["train"]) & set(sel["val"])            # train/val disjoint
    assert sel["only_adapted_eligible"] == int(only.sum())


def test_select_materials_handles_shortage_and_zero():
    mats, els = mrd.eligible_materials(fake_materials(), ADAPTED, 30, set(), 0)
    rng = np.random.default_rng(1)
    sel = mrd.select_materials(els, ADAPTED, 100, 50, 100, 3.0, rng)  # asks for more than exist
    assert len(sel["train"]) + len(sel["val"]) <= len(mats)
    assert len(set(sel["train"] + sel["val"])) == len(sel["train"]) + len(sel["val"])
    zero = mrd.select_materials(els, ADAPTED, 0, 0, 0, 3.0, rng)
    assert zero["train"] == [] and zero["val"] == []


def test_draw_oversampling_prefers_emphasis():
    els = pd.Series([{"F"}] * 10 + [{"O"}] * 90)
    counts = 0
    for seed in range(200):
        tr, _ = mrd.draw(els, np.ones(100, bool), 5, 0, {"F"}, 10.0, np.random.default_rng(seed))
        counts += sum(i < 10 for i in tr)
    # without oversampling ~0.5 per draw; with weight 10 it is ~2.6
    assert counts / 200 > 1.5


def _molecules():
    from ase.build import molecule

    return [molecule(n) for n in ("CH4", "H2O", "NH3", "CH3OH", "C2H6", "HCN", "CO2", "H2O2")]


def test_fit_and_apply_energy_offsets_recovers_truth():
    atoms = _molecules()
    truth = {"C": 1.7, "H": 0.9, "N": 3.6, "O": 2.2}
    base = np.linspace(-10, -2, len(atoms))
    teacher = base + np.array([sum(truth[e] for e in a.get_chemical_symbols()) for a in atoms])
    elements, offsets = mrd.fit_energy_offsets(atoms, teacher, base)
    assert dict(zip(elements, np.round(offsets, 6))) == truth
    df = pd.DataFrame({"ase_atoms": atoms, "energy": teacher, "energy_corrected": teacher,
                       "forces": [np.zeros((len(a), 3)) for a in atoms]})
    out = mrd.apply_energy_offsets(df, elements, offsets)
    assert np.allclose(out.energy, base) and np.allclose(out.energy_corrected, base)
    assert all(np.array_equal(f, g) for f, g in zip(out.forces, df.forces))   # forces unchanged
    assert np.allclose(df.energy, teacher)                                   # input untouched


def test_offsets_payload_roundtrip_and_missing():
    payload = mrd.offsets_payload(["C", "H"], [1.5, 0.5], 10)
    els, offs = mrd.offsets_from_payload(json.loads(json.dumps(payload)), {"C", "H"})
    assert els == ["C", "H"] and np.allclose(offs, [1.5, 0.5])
    with pytest.raises(SystemExit):
        mrd.offsets_from_payload(payload, {"C", "N"})


def test_organic_subset():
    df = pd.DataFrame({"x": range(10)})
    assert list(mrd.organic_subset(df, 3).x) == [0, 1, 2]
    assert len(mrd.organic_subset(df, None)) == 10
    with pytest.raises(SystemExit):
        mrd.organic_subset(df, 11)


def _fake_shard():
    rows = []
    for mp, numbers in [("v1", [6, 6]), ("v2", [26, 8]), ("v3", [1, 8, 1]), ("v4", [14, 14]),
                        ("v5", [7, 7]), ("v6", [8] * 50)]:
        n = len(numbers)
        rows.append({"mp_id": mp, "numbers": numbers, "num_atoms": n,
                     "positions": [list(map(float, [i, 0, 0])) for i in range(n)],
                     "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
                     "forces": [[0.1, 0.0, 0.0]] * n, "energy": -5.0 * n})
    return pd.DataFrame(rows)


def test_monitor_frames_and_write(tmp_path):
    atoms = _molecules()
    organic_test = pd.DataFrame({"ase_atoms": atoms, "energy": np.zeros(len(atoms)),
                                 "energy_corrected": np.zeros(len(atoms)),
                                 "forces": [np.zeros((len(a), 3)) for a in atoms],
                                 "dft_energy": 0.0})
    frames = mrd.monitor_frames(organic_test, _fake_shard(), ADAPTED, max_atoms=30, n_monitor=5,
                                seed=0)
    assert len(frames["organic"]) == 5 and list(frames["organic"].columns) == mrd.LABEL_COLUMNS
    mat_ids = {tuple(a.numbers) for a in frames["materials"].ase_atoms}
    assert len(frames["materials"]) == 4                 # v1 v2 v3 v5 (v4 no adapted, v6 too big)
    only = frames["only_adapted"]
    assert len(only) == 3 and all(set(a.get_chemical_symbols()) <= ADAPTED for a in only.ase_atoms)
    assert all(a.pbc.all() for a in frames["materials"].ase_atoms)
    assert mat_ids
    written = mrd.write_monitors(frames, tmp_path)
    assert {p.name for p, _ in written.values()} == {"organic.pkl.gz", "materials.pkl.gz",
                                                    "only_adapted.pkl.gz"}
    back = config.load_frame(tmp_path / "materials.pkl.gz")
    assert np.allclose(back.forces.iloc[0], 0.1 * np.eye(3)[0])


def test_materials_frame_empty():
    df = mrd.materials_frame(_fake_shard().iloc[0:0])
    assert len(df) == 0 and set(mrd.LABEL_COLUMNS) <= set(df.columns)


REAL_VAL = ROOT / "tests/fixtures/mptrj_val_sample.parquet"  # first 2000 rows of the MPtrj val shard


def test_real_val_shard_groups():
    shard = pd.read_parquet(REAL_VAL, columns=["mp_id", "numbers", "num_atoms"])
    mats, els = mrd.eligible_materials(shard, ADAPTED, 30, set(), 0)
    assert len(mats) > 100 and mats.mp_id.is_unique
    only = els.map(lambda s: s <= ADAPTED)
    assert 0 < only.sum() < len(mats)


# ---------------------------------------------------------------- pipeline.py -> B/C commands
def _capture(monkeypatch, tmp_path):
    settings = {"workdir": str(tmp_path), "paths": {"STUDENT_MODEL_DIR": "/m/base"},
                "device": "cpu-jit", "batch_size": 8, "max_buckets": 4, "venv": "/v"}
    calls = []
    monkeypatch.setattr(pipeline, "load_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "keep_awake_prefix", lambda: [])
    monkeypatch.setattr(pipeline, "sh", lambda cmd, **kw: calls.append([str(c) for c in cmd]))
    monkeypatch.delenv("GRACE_PIPELINE_VENV", raising=False)
    return calls


def test_experiment_command(monkeypatch, tmp_path):
    import argparse
    calls = _capture(monkeypatch, tmp_path)
    pipeline.cmd_experiment(argparse.Namespace(set=["rank=16", "batch_size=4"], rationale="why",
                                               parent="exp_001", dry_run=True))
    cmd = calls[0]
    assert cmd[0] == "/v/bin/python" and cmd[1].endswith("scripts/experiment.py")
    assert cmd[cmd.index("--workdir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--python") + 1] == "/v/bin/python"
    assert cmd[cmd.index("--rationale") + 1] == "why"
    assert cmd[cmd.index("--parent") + 1] == "exp_001" and "--dry-run" in cmd
    cfg = json.loads(cmd[cmd.index("--config") + 1])
    # experiment_defaults < pipeline.json device/batch_size/max_buckets < --set
    assert cfg["rank"] == 16 and cfg["batch_size"] == 4
    assert cfg["device"] == "cpu-jit" and cfg["max_buckets"] == 4
    assert cfg["n_organic"] == 1000 and cfg["n_materials"] == 600 and cfg["alpha"] is None


def test_ledger_and_report_commands(monkeypatch, tmp_path):
    import argparse
    calls = _capture(monkeypatch, tmp_path)
    pipeline.cmd_ledger(argparse.Namespace(json=True, best=False))
    pipeline.cmd_ledger(argparse.Namespace(json=False, best=True))
    pipeline.cmd_report(argparse.Namespace(exps=["exp_002"], out=None))
    show, best, report = calls
    assert show[1].endswith("ledger.py") and show[2:] == ["--workdir", str(tmp_path), "show", "--json"]
    assert best[2:] == ["--workdir", str(tmp_path), "best"]
    assert report[1].endswith("report.py") and report[2:] == ["--workdir", str(tmp_path),
                                                              "--exps", "exp_002"]


# ---------------------------------------------------------------- validate-models
def _touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_detect_model_kinds(tmp_path):
    sm = tmp_path / "sm"
    _touch(sm / "saved_model.pb")
    run = tmp_path / "run"
    _touch(run / "saved_model" / "saved_model.pb")
    _touch(run / "adapter.json")
    _touch(run / "best_trainable_state.npz")
    adapter = tmp_path / "adapter"
    _touch(adapter / "adapter.json")
    _touch(adapter / "best_trainable_state.npz")
    legacy = tmp_path / "legacy_adapter"
    _touch(legacy / "args.json")
    _touch(legacy / "best_trainable_state.npz")
    ck_dir = tmp_path / "checkpoint_10"
    _touch(ck_dir / "best_checkpoint.index")
    prefix = tmp_path / "ck" / "model"
    _touch(Path(str(prefix) + ".index"))

    d = pipeline.detect_model(sm)
    assert d["kind"] == "saved_model" and d["model_dir"] == sm
    d = pipeline.detect_model(run)
    assert d["kind"] == "run" and d["model_dir"] == run / "saved_model"
    assert pipeline.detect_model(adapter)["kind"] == "adapter"
    assert pipeline.detect_model(legacy)["kind"] == "adapter"
    d = pipeline.detect_model(ck_dir)
    assert d["kind"] == "checkpoint" and d["source"] == ck_dir / "best_checkpoint"
    d = pipeline.detect_model(prefix)
    assert d["kind"] == "checkpoint" and d["source"] == prefix
    d = pipeline.detect_model(Path(str(prefix) + ".index"))       # the .index file itself
    assert d["kind"] == "checkpoint" and d["source"] == prefix
    with pytest.raises(SystemExit):
        pipeline.detect_model(tmp_path / "nothing")


def test_resolve_models_builds_once(tmp_path):
    adapter = tmp_path / "adapter"
    _touch(adapter / "adapter.json")
    _touch(adapter / "best_trainable_state.npz")
    ck_dir = tmp_path / "checkpoint_11"
    _touch(ck_dir / "best_checkpoint.index")
    sm = tmp_path / "sm"
    _touch(sm / "saved_model.pb")
    out = tmp_path / "out"
    models = pipeline.resolve_models([f"ad={adapter}", f"ck={ck_dir}", f"sm={sm}"], out, "/py")
    ad, ck, plain = models
    assert ad["model_dir"] == out / "models" / "ad"
    assert ad["build"][1].endswith("apply_adapter.py") and ad["build"][2] == adapter
    assert ck["build"][1].endswith("export_checkpoint.py") and ck["build"][2] == ck_dir / "best_checkpoint"
    assert ck["build"][3:] == ["--out", out / "models" / "ck"]
    assert plain["build"] is None and plain["model_dir"] == sm
    # already built -> no build command
    _touch(out / "models" / "ad" / "saved_model.pb")
    again = pipeline.resolve_models([f"ad={adapter}"], out, "/py")
    assert again[0]["build"] is None
    for bad in (["noequals"], [f"base={sm}"], [f"x={sm}", f"x={sm}"]):
        with pytest.raises(SystemExit):
            pipeline.resolve_models(bad, out, "/py")


def _cmds(tmp_path, **kw):
    models = [{"name": "v2", "model_dir": tmp_path / "v2"}, {"name": "ck", "model_dir": tmp_path / "ck"}]
    paths = {"STUDENT_MODEL_DIR": "/cache/base", "TEACHER_MODEL_DIR": "/off/saved_model"}
    cmds = pipeline.validation_commands("/py", models, tmp_path / "org.pkl.gz",
                                        tmp_path / "val.parquet", tmp_path / "out", paths,
                                        tmp_path, **kw)
    return {label: [str(c) for c in cmd] for label, cmd in cmds}


def test_validation_commands_full(tmp_path):
    c = _cmds(tmp_path)
    org, mat, rep = c["organic"], c["materials"], c["report"]
    assert org[1].endswith("validate_model.py")
    assert org[org.index("--trained") + 1: org.index("--trained") + 3] == [
        f"v2={tmp_path / 'v2'}", f"ck={tmp_path / 'ck'}"]
    assert org[org.index("--n") + 1] == "0"
    assert org[org.index("--out") + 1] == str(tmp_path / "out" / "organic")
    models = mat[mat.index("--models") + 1: mat.index("--out")]
    assert models == ["base=/cache/base", f"v2={tmp_path / 'v2'}", f"ck={tmp_path / 'ck'}",
                      "GRACE-OFF=/off/saved_model"]
    assert "--n-a" not in mat
    assert rep[1].endswith("report.py") and rep[rep.index("--validation-only") + 1] == str(tmp_path / "out")


def test_validation_commands_quick_no_base_no_teacher(tmp_path):
    c = _cmds(tmp_path, quick=True, no_base=True, no_teacher=True)
    org, mat = c["organic"], c["materials"]
    assert org[org.index("--n") + 1] == "100"
    models = mat[mat.index("--models") + 1: mat.index("--out")]
    assert models == [f"v2={tmp_path / 'v2'}", f"ck={tmp_path / 'ck'}"]
    assert mat[mat.index("--n-a") + 1] == "50" and mat[mat.index("--n-b") + 1] == "150" \
        and mat[mat.index("--n-c") + 1] == "60"


def test_adhoc_dir_naming(tmp_path):
    import time as _time
    settings = {"workdir": str(tmp_path)}
    t = _time.mktime((2026, 10, 23, 9, 5, 0, 0, 0, -1))
    assert pipeline.adhoc_dir(settings, t) == tmp_path / "validation" / "adhoc_20261023_0905"


def test_validate_models_dry_run(monkeypatch, tmp_path, capsys):
    import argparse
    calls = _capture(monkeypatch, tmp_path)
    org = _touch(tmp_path / "data" / "organic" / "test.pkl.gz")
    mat = _touch(tmp_path / "data" / "materials" / "val.parquet")
    pipeline.load_settings()["paths"].update(TEACHER_MODEL_DIR="/off")
    ck_dir = tmp_path / "checkpoint_10"
    _touch(ck_dir / "best_checkpoint.index")
    args = argparse.Namespace(model=[f"ck10={ck_dir}"], no_base=False, no_teacher=True,
                              quick=True, organic=None, materials=None, out=str(tmp_path / "o"),
                              force=False, dry_run=True)
    pipeline.cmd_validate_models(args)
    text = capsys.readouterr().out
    assert "raw checkpoint" in text and "export_checkpoint.py" in text
    assert str(org) in text and str(mat) in text and "--n 100" in text
    assert "GRACE-OFF=" not in text and calls == []          # dry run: nothing executed


def test_validate_models_missing_data(monkeypatch, tmp_path):
    import argparse
    _capture(monkeypatch, tmp_path)
    sm = tmp_path / "sm"
    _touch(sm / "saved_model.pb")
    args = argparse.Namespace(model=[f"m={sm}"], no_base=False, no_teacher=False, quick=False,
                              organic=None, materials=None, out=None, force=False, dry_run=True)
    with pytest.raises(SystemExit, match="pipeline.py prepare"):
        pipeline.cmd_validate_models(args)


def test_validate_exp_quick_folder_and_sizes(monkeypatch, tmp_path):
    import argparse
    calls = _capture(monkeypatch, tmp_path)
    pipeline.load_settings()["paths"].update(TEACHER_MODEL_DIR="/off")
    _touch(tmp_path / "data" / "organic" / "test.pkl.gz")
    _touch(tmp_path / "data" / "materials" / "val.parquet")
    _touch(tmp_path / "experiments" / "exp_002" / "run" / "saved_model" / "saved_model.pb")
    pipeline.cmd_validate_exp(argparse.Namespace(exp="exp_002", force=False, quick=True))
    organic, materials = calls
    assert organic[organic.index("--n") + 1] == "100"
    assert organic[organic.index("--out") + 1] == str(tmp_path / "validation" / "exp_002_quick" / "organic")
    assert materials[materials.index("--n-c") + 1] == "60"


# ---------------------------------------------------------------- agent options (round 3)
def test_agent_options_defaults_all_off():
    opts = pipeline.get_agent_options({})
    assert opts == {"subagents": False, "critic": False, "self_heal": False, "cockpit": False,
                    "cockpit_publish": False, "predictions": False, "max_subagents": 3}
    partial = pipeline.get_agent_options({"agent_options": {"cockpit": True, "max_subagents": "x"}})
    assert partial["cockpit"] and not partial["critic"] and partial["max_subagents"] == 3


def test_set_agent_options_enable_disable_set():
    settings = {}
    opts, warns = pipeline.set_agent_options(settings, enable=["subagents", "critic"],
                                             sets=["max_subagents=2"])
    assert opts["subagents"] and opts["critic"] and opts["max_subagents"] == 2 and warns == []
    assert settings["agent_options"] == opts                        # stored in pipeline.json
    opts, _ = pipeline.set_agent_options(settings, disable=["critic"])
    assert not opts["critic"] and opts["subagents"] and opts["max_subagents"] == 2


def test_set_agent_options_validation():
    for kw in ({"enable": ["telepathy"]}, {"disable": ["nope"]},
               {"enable": ["critic"], "disable": ["critic"]},
               {"sets": ["max_subagents=-1"]}, {"sets": ["max_subagents=two"]},
               {"sets": ["critic=1"]}, {"sets": ["max_subagents"]}):
        with pytest.raises(SystemExit):
            pipeline.set_agent_options({}, **kw)


def test_critic_without_subagents_warns():
    opts, warns = pipeline.set_agent_options({}, enable=["critic"])
    assert opts["critic"] and not opts["subagents"]
    assert len(warns) == 1 and "--enable subagents" in warns[0]


def test_cmd_options_show_and_save(monkeypatch, capsys):
    import argparse
    store = {}
    monkeypatch.setattr(pipeline, "load_settings", lambda: store)
    monkeypatch.setattr(pipeline, "save_settings", lambda s: store.update(s))
    pipeline.cmd_options(argparse.Namespace(enable=[], disable=[], set=[], json=False))
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "Subagents: OFF (the skill will not launch any agents)"
    assert "max_subagents = 3" in out and "[off] critic" in out
    assert store["agent_options"]["subagents"] is False             # full table written
    pipeline.cmd_options(argparse.Namespace(enable=["subagents"], disable=[],
                                            set=["max_subagents=1"], json=True))
    shown = json.loads(capsys.readouterr().out)
    assert shown["subagents"] is True and shown["max_subagents"] == 1
    assert pipeline.subagents_line(shown) == "Subagents: ON (at most 1 per loop)"


def _exp_args(**kw):
    import argparse
    base = dict(set=[], rationale="r", parent=None, dry_run=False, self_heal=None,
                hypothesis=None, predict=[])
    base.update(kw)
    return argparse.Namespace(**base)


def test_experiment_command_self_heal_and_predictions(tmp_path):
    settings = {"workdir": str(tmp_path)}
    cmd = [str(c) for c in pipeline.experiment_command("/py", settings, _exp_args())]
    assert "--self-heal" not in cmd and "--hypothesis" not in cmd and "--predict" not in cmd
    on = {"workdir": str(tmp_path), "agent_options": {"self_heal": True}}
    assert "--self-heal" in [str(c) for c in pipeline.experiment_command("/py", on, _exp_args())]
    assert "--self-heal" not in [str(c) for c in
                                 pipeline.experiment_command("/py", on, _exp_args(self_heal=False))]
    assert "--self-heal" in [str(c) for c in
                             pipeline.experiment_command("/py", settings, _exp_args(self_heal=True))]
    cmd = [str(c) for c in pipeline.experiment_command(
        "/py", settings, _exp_args(hypothesis="more C data helps C",
                                   predict=["organic_f=95", "only_adapted_f=100.5"]))]
    assert cmd[cmd.index("--hypothesis") + 1] == "more C data helps C"
    preds = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--predict"]
    assert preds == ["organic_f=95", "only_adapted_f=100.5"]
    for bad in (["energy=1"], ["organic_f"], ["organic_f=abc"]):
        with pytest.raises(SystemExit):
            pipeline.experiment_command("/py", settings, _exp_args(predict=bad))


def test_cockpit_command(tmp_path):
    import argparse
    settings = {"workdir": str(tmp_path)}
    cmd = [str(c) for c in pipeline.cockpit_command("/py", settings,
                                                    argparse.Namespace(watch=None, out=None))]
    assert cmd[1].endswith("scripts/cockpit.py")
    assert cmd[cmd.index("--workdir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "report" / "cockpit.html")
    assert "--watch" not in cmd
    cmd = [str(c) for c in pipeline.cockpit_command(
        "/py", settings, argparse.Namespace(watch=30, out=str(tmp_path / "c.html")))]
    assert cmd[cmd.index("--watch") + 1] == "30" and cmd[cmd.index("--out") + 1] == str(tmp_path / "c.html")


# ---------------------------------------------------------------- settings (round 4)
SPECS = pipeline.SETTING_SPECS


def test_settings_registry_complete():
    groups_seen = []
    for key, spec in SPECS.items():
        assert spec["group"] in pipeline.SETTINGS_GROUPS, key
        assert "default" in spec and spec["type"] in ("int", "float", "bool", "choice", "str"), key
        assert spec["affects"] and set(spec["affects"]) <= set(pipeline.AFFECTS), key
        assert spec["short"] and len(spec["short"]) < 80, key
        sentences = [x for x in spec["long"].replace("e.g.", "eg").split(". ") if x.strip()]
        assert 2 <= len(sentences) <= 7, (key, len(sentences))
        if spec["type"] == "choice":
            assert spec["default"] in spec["allowed"], key
        if spec["group"] not in groups_seen:
            groups_seen.append(spec["group"])
    assert tuple(groups_seen) == pipeline.SETTINGS_GROUPS          # registry in display order
    for key in ("experiment_defaults.rank", "device", "batch_size", "max_buckets",
                "objective.max_materials_increase", "budget.time_budget_min",
                "budget.max_experiments", "validation.mid_loop", "validation.final",
                "agent_options.subagents", "agent_options.max_subagents", "agent_options.critic",
                "agent_options.self_heal", "agent_options.predictions", "agent_options.cockpit"):
        assert key in SPECS
    # agent option defaults agree with get_agent_options
    for name, value in pipeline.get_agent_options({}).items():
        assert SPECS[f"agent_options.{name}"]["default"] == value


def test_resolve_setting_dotted_and_short():
    assert pipeline.resolve_setting("rank") == "experiment_defaults.rank"
    assert pipeline.resolve_setting("experiment_defaults.rank") == "experiment_defaults.rank"
    assert pipeline.resolve_setting("critic") == "agent_options.critic"
    assert pipeline.resolve_setting("mid_loop") == "validation.mid_loop"
    assert pipeline.resolve_setting("device") == "device"
    with pytest.raises(SystemExit):
        pipeline.resolve_setting("warp_speed")


@pytest.mark.parametrize("key, raw, ok", [
    ("rank", "16", 16), ("rank", "0", None), ("rank", "65", None), ("rank", "8.5", None),
    ("epochs", "100", 100), ("epochs", "101", None), ("objective.max_materials_increase", "-0.1", None),
    ("objective.max_materials_increase", "0.2", 0.2), ("device", "gpu", "gpu"), ("device", "tpu", None),
    ("validation.mid_loop", "full", "full"), ("validation.final", "medium", None),
    ("critic", "true", True), ("critic", "yes", None), ("alpha", "null", None),
    ("alpha", "4", 4.0), ("batch_size", "null", "ERR"), ("oversample", "0.5", None),
    ("max_subagents", "2", 2), ("max_subagents", "-1", None), ("lr", "1e-3", 0.001),
])
def test_coerce_setting(key, raw, ok):
    full = pipeline.resolve_setting(key)
    if ok is None and not (key == "alpha" and raw == "null"):
        with pytest.raises(SystemExit):
            pipeline.coerce_setting(full, raw)
    elif ok == "ERR":
        with pytest.raises(SystemExit):
            pipeline.coerce_setting(full, raw)
    else:
        assert pipeline.coerce_setting(full, raw) == ok


def test_apply_settings_changes_set_reset_confirm():
    settings = {"settings_confirmed": True, "device": "cpu-jit"}
    changed = pipeline.apply_settings_changes(settings, ["rank=16", "agent_options.critic=true",
                                                         "time_budget_min=240"])
    assert changed == ["experiment_defaults.rank", "agent_options.critic", "budget.time_budget_min"]
    assert settings["experiment_defaults"]["rank"] == 16
    assert settings["agent_options"]["critic"] is True
    assert settings["budget"]["time_budget_min"] == 240
    assert settings["settings_confirmed"] is False                  # any change unconfirms
    assert pipeline.get_setting(settings, "rank") == 16
    settings["settings_confirmed"] = True
    pipeline.apply_settings_changes(settings, reset=["rank"])
    assert pipeline.get_setting(settings, "rank") == 8 and settings["settings_confirmed"] is False
    pipeline.apply_settings_changes(settings, reset=[])               # reset all
    assert all(pipeline.get_setting(settings, k) == v["default"] for k, v in SPECS.items())
    with pytest.raises(SystemExit):
        pipeline.apply_settings_changes(settings, ["student=other"])  # read-only
    with pytest.raises(SystemExit):
        pipeline.apply_settings_changes(settings, ["rank16"])
    unchanged = {"settings_confirmed": True}
    assert pipeline.apply_settings_changes(unchanged) == [] and unchanged["settings_confirmed"]


def test_get_setting_models_and_defaults():
    settings = {"student": {"name": "GRACE-2L-OMAT-large-ft-AM"}, "data": {"source": "local"}}
    assert pipeline.get_setting(settings, "student") == "GRACE-2L-OMAT-large-ft-AM"
    assert pipeline.get_setting(settings, "teacher") == pipeline.DEFAULT_TEACHER
    assert pipeline.get_setting(settings, "data") == "local"
    assert pipeline.get_setting({}, "epochs") == 10


def test_estimate_minutes():
    est = pipeline.estimate_minutes({})
    assert est["s_per_structure_epoch"] == 0.2 and est["structures"] == 1600
    assert abs(est["per_experiment_min"] - 1600 * 10 * 0.2 / 60) < 1e-9
    assert abs(est["total_min"] - 4 * est["per_experiment_min"]) < 1e-9 and est["budget_min"] == 240
    custom = {"benchmark": {"train_s_per_structure_epoch": 0.1},
              "experiment_defaults": {"n_organic": 500, "n_materials": 0, "epochs": 6},
              "budget": {"max_experiments": 2}}
    est = pipeline.estimate_minutes(custom)
    assert est["source"] == "smoke-test benchmark" and abs(est["per_experiment_min"] - 5.0) < 1e-9
    assert abs(est["total_min"] - 10.0) < 1e-9


def test_experiment_config_precedence(tmp_path):
    settings = {"workdir": str(tmp_path), "device": "gpu", "batch_size": 16,
                "experiment_defaults": {"rank": 12, "epochs": 5}}
    cmd = [str(c) for c in pipeline.experiment_command("/py", settings,
                                                       _exp_args(set=["epochs=7", "device=cpu"]))]
    cfg = json.loads(cmd[cmd.index("--config") + 1])
    assert cfg["rank"] == 12                    # stored experiment default
    assert cfg["epochs"] == 7                   # --set wins over the stored default
    assert cfg["batch_size"] == 16              # pipeline.json top-level value
    assert cfg["device"] == "cpu"               # --set wins over pipeline.json
    assert cfg["max_buckets"] == 4 and cfg["n_materials"] == 600   # registry defaults


def test_settings_panel_and_explain(monkeypatch, capsys):
    import argparse
    store = {"experiment_defaults": {"rank": 16}}
    monkeypatch.setattr(pipeline, "load_settings", lambda: store)
    monkeypatch.setattr(pipeline, "save_settings", lambda s: None)
    text = pipeline.settings_panel(store)
    for group in pipeline.SETTINGS_GROUPS:
        assert f"== {group}" in text
    assert "✱" in text and "Subagents: OFF (no extra agents launched)" in text
    assert "≈" in text and "vs time budget 240 min" in text
    assert "references/settings.md" in text and "settings --confirm" in text
    pipeline.cmd_settings(argparse.Namespace(explain="rank", set=[], reset=None, confirm=False,
                                             json=False))
    out = capsys.readouterr().out
    assert out.startswith("experiment_defaults.rank") and "current: 16" in out
    assert SPECS["experiment_defaults.rank"]["long"] in out


def test_cmd_settings_set_then_confirm(monkeypatch, capsys):
    import argparse
    store = {}
    monkeypatch.setattr(pipeline, "load_settings", lambda: store)
    monkeypatch.setattr(pipeline, "save_settings", lambda s: None)
    pipeline.cmd_settings(argparse.Namespace(explain=None, set=["epochs=12"], reset=None,
                                             confirm=False, json=True))
    data = json.loads(capsys.readouterr().out.split("\n", 1)[1])   # skip "Changed:" line
    assert data["settings"]["experiment_defaults.epochs"]["value"] == 12
    assert data["settings"]["experiment_defaults.epochs"]["changed"] is True
    assert data["settings_confirmed"] is False
    pipeline.cmd_settings(argparse.Namespace(explain=None, set=[], reset=None, confirm=True,
                                             json=False))
    assert store["settings_confirmed"] is True and "Settings confirmed." in capsys.readouterr().out


def test_options_change_unconfirms(monkeypatch, capsys):
    import argparse
    store = {"settings_confirmed": True, "agent_options": {"critic": False}}
    monkeypatch.setattr(pipeline, "load_settings", lambda: store)
    monkeypatch.setattr(pipeline, "save_settings", lambda s: None)
    pipeline.cmd_options(argparse.Namespace(enable=["cockpit"], disable=[], set=[], json=True))
    assert store["settings_confirmed"] is False


def test_validation_mode():
    assert pipeline.validation_mode({}) == "quick"                       # mid_loop default
    assert pipeline.validation_mode({}, final=True) == "full"            # final default
    assert pipeline.validation_mode({}, full=True) == "full"
    assert pipeline.validation_mode({"validation": {"final": "quick"}}, final=True) == "quick"
    assert pipeline.validation_mode({"validation": {"mid_loop": "full"}}) == "full"
    assert pipeline.validation_mode({}, quick=True, final=True) == "quick"
    with pytest.raises(SystemExit):
        pipeline.validation_mode({}, quick=True, full=True)


def test_ensure_settings_confirmed_noninteractive(monkeypatch, capsys):
    store = {}
    monkeypatch.setattr(pipeline, "load_settings", lambda: store)
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit, match="settings --confirm"):
        pipeline.ensure_settings_confirmed()
    store["settings_confirmed"] = True
    pipeline.ensure_settings_confirmed()                                  # no exception


def test_set_accepts_several_pairs_per_flag():
    """`--set a=1 b=2 --set c=3` must parse (agents naturally write several pairs per flag)."""
    import subprocess
    out = subprocess.run(
        [sys.executable, str(ROOT / "pipeline.py"), "experiment", "--set", "n_organic=200",
         "n_materials=150", "--set", "epochs=3", "--rationale", "t", "--dry-run", "--help"],
        capture_output=True, text=True)
    assert "unrecognized arguments" not in out.stderr
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="extend", nargs="+", default=[])
    assert p.parse_args(["--set", "a=1", "b=2", "--set", "c=3"]).set == ["a=1", "b=2", "c=3"]
    src = (ROOT / "pipeline.py").read_text()
    assert 'add_argument("--set", action="append"' not in src


# ---------------------------------------------------------------- hardening: checksums + pickles
def test_checksums_pinned_files(tmp_path, monkeypatch):
    import checksums
    monkeypatch.delenv(checksums.SKIP_ENV, raising=False)
    f = tmp_path / "GRACE_test_set.xyz"
    f.write_text("not the real file")
    with pytest.raises(checksums.ChecksumError):
        checksums.verify_file(f)
    monkeypatch.setenv(checksums.SKIP_ENV, "1")
    assert checksums.verify_file(f)                      # warns, does not raise
    other = tmp_path / "my_own.xyz"
    other.write_text("x")
    assert checksums.verify_file(other) is False         # user files are not pinned
    assert checksums.FILES["val.parquet"] == checksums.FILES["val-00000-of-00001.parquet"]
    assert checksums.expected_tree(tmp_path / "x/b_off_medium/seed/1/saved_model") is not None
    assert checksums.expected_tree(tmp_path / "checkpoints/GRACE-2L-OMAT-medium-ft-AM") is \
        checksums.TREES["checkpoints/GRACE-2L-OMAT-medium-ft-AM"]


def test_checksums_cli_on_sample(tmp_path):
    import checksums
    assert checksums.main(["sha256", str(REAL_VAL)]) == 0
    assert checksums.main(["verify", str(REAL_VAL)]) == 0  # not a pinned name -> just reported


def test_pickles_need_a_sidecar(tmp_path, monkeypatch):
    monkeypatch.delenv(config.TRUST_PICKLES_ENV, raising=False)
    df = pd.DataFrame({"a": [1, 2]})
    good = config.save_frame(df, tmp_path / "ok.pkl.gz")
    assert (tmp_path / "ok.pkl.gz.sha256").exists()
    assert config.load_frame(good).equals(df)
    foreign = tmp_path / "foreign.pkl.gz"
    df.to_pickle(foreign)
    with pytest.raises(SystemExit, match="not been recorded"):
        config.load_frame(foreign)
    config.trust_file(foreign)
    assert config.load_frame(foreign).equals(df)
    df.assign(a=[3, 4]).to_pickle(good)                   # changed after it was written
    with pytest.raises(SystemExit, match="changed since"):
        config.load_frame(good)
    monkeypatch.setenv(config.TRUST_PICKLES_ENV, "1")
    assert len(config.load_frame(good)) == 2
