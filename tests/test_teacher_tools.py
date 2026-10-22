"""Tests for scripts/teacher_tools.py: no network, no model loading.
Run with pytest or directly: python tests/test_teacher_tools.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import teacher_tools as T  # noqa: E402


def _tmp():
    return Path(tempfile.mkdtemp(prefix="test_teacher_"))


def fake_model(folder: Path, elements=("C", "H", "O"), cutoff=6.0, var_bytes=1000, meta=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "saved_model.pb").write_bytes(b"\x08\x01fake-protobuf")
    (folder / "variables").mkdir(exist_ok=True)
    (folder / "variables" / "variables.data-00000-of-00001").write_bytes(b"0" * var_bytes)
    (folder / "variables" / "variables.index").write_bytes(b"idx")
    if meta:
        (folder / "metadata.yaml").write_text(
            "chemical_symbols:\n" + "".join(f"- {e}\n" for e in elements) + f"cutoff: {cutoff}\n")
    return folder


def test_classify():
    assert T.classify("GRACE-OFF-2L-medium") == "builtin"
    assert T.classify("grace:GRACE-2L-OMAT-large-ft-AM") == "grace_models"
    assert T.classify("https://github.com/heid-lab/grace-off.git") == "git"
    assert T.classify("https://github.com/heid-lab/grace-off") == "git"
    assert T.classify("https://huggingface.co/org/model") == "git"
    assert T.classify("git@gitlab.com:group/repo.git") == "git"
    assert T.classify("10.1021/acs.jctc.6c01169") == "publication"
    assert T.classify("https://doi.org/10.1021/acs.jctc.6c01169") == "publication"
    assert T.classify("https://arxiv.org/abs/2508.17936") == "publication"
    assert T.classify("/some/local/path") == "local"


def test_resolve_builtin_needs_fetch_then_found():
    wd = _tmp()
    r = T.resolve("GRACE-OFF-2L-medium", wd)
    assert r["kind"] == "builtin" and r["ok"] and r["needs_fetch"]
    assert r["model_dir"].endswith("models/grace-off/models/2l/b_off_medium/seed/1/saved_model")
    assert r["elements"] == T.GRACE_OFF_ELEMENTS
    fake_model(Path(r["model_dir"]), elements=T.GRACE_OFF_ELEMENTS, var_bytes=2000)
    fake_model(Path(r["model_dir"]).parent / "casted_model", elements=T.GRACE_OFF_ELEMENTS,
               var_bytes=1000)
    r = T.resolve("GRACE-OFF-2L-medium", wd)
    assert not r["needs_fetch"] and r["candidates"][0]["precision"] == "fp64"


def test_resolve_grace_models_with_cache():
    cache = _tmp()
    old = os.environ.get("GRACE_CACHE")
    os.environ["GRACE_CACHE"] = str(cache)
    try:
        r = T.resolve("grace:GRACE-2L-OMAT-large-ft-AM", _tmp())
        assert r["kind"] == "grace_models" and r["needs_fetch"] and r["ok"]
        assert len(r["elements"]) == 89
        fake_model(cache / "GRACE-2L-OMAT-large-ft-AM", elements=("Fe", "O"))
        r = T.resolve("grace:GRACE-2L-OMAT-large-ft-AM", _tmp())
        assert not r["needs_fetch"] and r["elements"] == ["Fe", "O"]
    finally:
        if old is None:
            os.environ.pop("GRACE_CACHE")
        else:
            os.environ["GRACE_CACHE"] = old


def test_resolve_local_prefers_saved_model_over_casted():
    root = _tmp()
    fake_model(root / "seed" / "1" / "saved_model", var_bytes=2000)
    fake_model(root / "seed" / "1" / "casted_model", var_bytes=1000)
    fake_model(root / "seed" / "1" / "checkpoints" / "old", var_bytes=10)   # skipped
    r = T.resolve(str(root), _tmp())
    assert r["kind"] == "local" and r["ok"]
    assert r["model_dir"].endswith("seed/1/saved_model")
    assert [Path(c["path"]).name for c in r["candidates"]] == ["saved_model", "casted_model"]
    assert r["candidates"][0]["precision"] == "fp64"
    assert r["candidates"][1]["is_casted"] and r["candidates"][1]["precision"] == "fp32"
    assert r["elements"] == ["C", "H", "O"]


def test_resolve_local_direct_saved_model_and_multiple_candidates():
    root = _tmp()
    direct = fake_model(root / "a" / "saved_model")
    r = T.resolve(str(direct), _tmp())
    assert r["model_dir"] == str(direct) and r["ok"]
    fake_model(root / "b" / "saved_model")
    r = T.resolve(str(root), _tmp())
    assert r["model_dir"] is None and not r["ok"] and len(r["candidates"]) == 2
    assert any("candidate models found" in n for n in r["notes"])


def test_mace_model_rejected():
    root = _tmp()
    (root / "MACE-OFF24_medium.model").write_bytes(b"torch")
    r = T.resolve(str(root), _tmp())
    assert not r["ok"] and any("MACE/PyTorch" in n for n in r["notes"])
    c = T.check_model(root)
    assert not c["ok"] and any("not a GRACE model" in w for w in c["warnings"])
    single = T.resolve(str(root / "MACE-OFF24_medium.model"), _tmp())
    assert not single["ok"] and any("MACE/PyTorch" in n for n in single["notes"])


def test_check_model_warnings():
    good = fake_model(_tmp() / "saved_model")
    c = T.check_model(good)
    assert c["ok"] and c["elements"] == ["C", "H", "O"] and c["cutoff"] == 6.0 and not c["warnings"]
    casted = fake_model(_tmp() / "casted_model")
    c = T.check_model(casted)
    assert not c["ok"] and c["precision"] == "fp32" and any("casted_model" in w for w in c["warnings"])
    nometa = fake_model(_tmp() / "saved_model", meta=False)
    c = T.check_model(nometa)
    assert not c["ok"] and any("metadata.yaml" in w for w in c["warnings"])
    lfs = fake_model(_tmp() / "saved_model")
    (lfs / "saved_model.pb").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:x\n")
    assert any("git-lfs" in w for w in T.check_model(lfs)["warnings"])


def test_publication_returns_note():
    for spec in ("10.1021/acs.jctc.6c01169", "https://doi.org/10.1021/acs.jctc.6c01169"):
        r = T.resolve(spec, _tmp())
        assert r["kind"] == "publication" and not r["ok"] and r["model_dir"] is None
        assert any("Code/Data availability" in n for n in r["notes"])


def test_resolve_git_with_mocked_clone():
    wd = _tmp()
    calls = []

    def fake_clone(url, dest):
        calls.append((url, dest))
        fake_model(dest / "models" / "2l" / "b_off_medium" / "seed" / "1" / "saved_model",
                   elements=T.GRACE_OFF_ELEMENTS, var_bytes=2000)
        fake_model(dest / "models" / "2l" / "b_off_medium" / "seed" / "1" / "casted_model",
                   elements=T.GRACE_OFF_ELEMENTS, var_bytes=1000)
        (dest / "models" / "MACE-OFF24_medium.model").write_bytes(b"torch")
        return ["mock clone"]

    real = T.clone_repo
    T.clone_repo = fake_clone
    try:
        r = T.resolve("https://github.com/heid-lab/grace-off.git", wd)
    finally:
        T.clone_repo = real
    assert calls and calls[0][1] == wd / "models" / "teachers" / "grace-off"
    assert r["kind"] == "git" and r["ok"] and r["model_dir"].endswith("seed/1/saved_model")
    assert len(r["candidates"]) == 2 and r["elements"] == T.GRACE_OFF_ELEMENTS


def test_recommend_ranking():
    wd = _tmp()
    rec = T.recommend(["C", "H", "O", "N"], "organic drug molecules", wd)
    assert rec["teachers"][0]["spec"] == "GRACE-OFF-2L-medium" and rec["teachers"][0]["covers_all"]
    assert not any("no specialist teacher" in n for n in rec["notes"])

    rec = T.recommend(["Fe", "O"], "oxide materials", wd)
    top = rec["teachers"][0]
    assert top["spec"].startswith("grace:") and top["covers_all"]
    off = next(t for t in rec["teachers"] if t["spec"] == "GRACE-OFF-2L-medium")
    assert off["missing"] == ["Fe"] and not off["covers_all"]
    assert any("home domain" in n for n in rec["notes"])
    assert any("no specialist teacher" in n for n in rec["notes"])

    rec = T.recommend(["Li", "C", "O", "F"], "battery electrolyte", wd)
    assert rec["teachers"][0]["spec"].startswith("grace:")
    assert all(t["missing"] == ["Li"] for t in rec["teachers"] if t["spec"].startswith("GRACE-OFF"))
    assert any("no specialist teacher" in n for n in rec["notes"])
    assert rec["replay"]["available"] is False


def test_recommend_includes_local_teachers():
    wd = _tmp()
    fake_model(wd / "models" / "teachers" / "my-li-model" / "saved_model",
               elements=("Li", "C", "O", "F"))
    rec = T.recommend(["Li", "C", "O", "F"], "", wd)
    assert rec["teachers"][0]["covers_all"]
    local = [t for t in rec["teachers"] if t["level_of_theory"].startswith("unknown")]
    assert local and local[0]["covers_all"]
    assert not any("no specialist teacher" in n for n in rec["notes"])


def test_count_materials():
    rows = [([26, 8, 8], "mp-1"),            # Fe O O       -> with
            ([6, 1, 1, 1, 1], "mp-2"),       # C H4         -> with + only
            ([6, 1, 1, 1, 1], "mp-2"),       # duplicate frame of mp-2 -> counted once
            ([14, 14], "mp-3"),              # Si Si        -> neither
            ([8] * 40, "mp-4"),              # too large
            ([6, 8, 8], "mp-5")]             # C O2, excluded (val shard)
    counts = T.count_materials(rows, {"C", "H", "O"}, max_atoms=30, exclude_ids={"mp-5"})
    assert counts == {"materials_with_elements": 2, "only_these_elements": 1}


def test_cli_result_line():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = T.main(["--workdir", str(_tmp()), "resolve", "GRACE-OFF-1L-small"])
    last = out.getvalue().strip().splitlines()[-1]
    assert code == 0 and last.startswith("RESULT:")
    assert json.loads(last[len("RESULT:"):])["needs_fetch"] is True


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
