"""SHA-256 verification of downloaded data and models. Standard library only.

Every file the pipeline downloads has a pinned SHA-256 below, recorded from the published sources
(Zenodo's own MD5s and Hugging Face's LFS SHA-256s were cross-checked against them). A download
whose content differs is rejected, so a corrupted, truncated or swapped file never reaches
training. Models and data the user brings themselves are not in this table and are not checked.

If an upstream source is updated on purpose, set GRACE_LORA_SKIP_CHECKSUMS=1 to bypass the check
(a warning is printed), or update the table after verifying the new files.

    python scripts/checksums.py verify data/biased_test_set.xyz      # one file (by its name)
    python scripts/checksums.py verify-dir ~/.cache/grace/GRACE-2L-OMAT-medium-ft-AM
    python scripts/checksums.py sha256 FILE...                        # print hashes
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

SKIP_ENV = "GRACE_LORA_SKIP_CHECKSUMS"

# Single files, keyed by file name.
FILES = {
    # SPICE v2 subsets, Zenodo record 18662355 (MD5s match the record's published checksums)
    "biased_test_set.xyz": "5765c03b935329bf817a93ba871a39650e24a2b9d456a85aa53f78e897af48ea",
    "GRACE_test_set.xyz": "9f4035bdcb3acff4073d1f8b1f4414ca3e46a65b4aa6199ba0c6e030f8c93c48",
    # MPtrj shards, Hugging Face nimashoghi/mptrj at revision MPTRJ_REVISION (= HF LFS SHA-256)
    "val-00000-of-00001.parquet": "b02fbc68c3f596603973ad078a4202a5f4ce688c17128089a43fb3b66eb2a169",
    "test-00000-of-00001.parquet": "cb77ce289ba73357be0cc375df40631d0df0541e7120913cb2e89a17aae19add",
    "train-00000-of-00008.parquet": "8997efa84cba37495ab57a6fe30ef1727d2136dc403fae8f439d7e3ee3976dea",
}
# Local names of the MPtrj shards (scripts/fetch_materials.py) -> remote names above
FILES["val.parquet"] = FILES["val-00000-of-00001.parquet"]
FILES["test.parquet"] = FILES["test-00000-of-00001.parquet"]
FILES["train0.parquet"] = FILES["train-00000-of-00008.parquet"]

MPTRJ_REVISION = "f88fbe46e16524223210654bad9e1b05a15c2adb"

# Model folders, keyed by a path suffix that identifies them; values: {relative file: sha256}.
# Only the files that define the model are pinned (not download-time metadata).
TREES = {
    # GRACE-OFF 2L medium, github.com/heid-lab/grace-off at commit edab07cc91e9
    "b_off_medium/seed/1/saved_model": {
        "saved_model.pb": "ab671545978be0f4bdc0c8ab9059a4a91735cfabc8f80178594106a79e640594",
        "variables/variables.data-00000-of-00001":
            "f75bef493f26cc11f4e89f1bad94e2d94c88978fe7e7b8a26b7f8ad436743c24",
        "variables/variables.index": "837011db364339fe4930d1ced81da889cb7e734a3a1bc3073232d1e40daf1b65",
    },
    # Default student, from `grace_models download` / `grace_models checkpoint`
    "checkpoints/GRACE-2L-OMAT-medium-ft-AM": {
        "checkpoint.data-00000-of-00001":
            "f72157b35d7a8a7de22b4dbb7fadaf90f2eca94a0498fda951e84f93183652a5",
        "checkpoint.index": "9b6c8532e472bec611c33a5979bd96c031246976b532f12399ed2244f53e6423",
        "model.yaml": "e6938645f8f55816be0ac3b7a07807e3f7a5c78877f5103832f1accdd917181b",
    },
    "GRACE-2L-OMAT-medium-ft-AM": {
        "saved_model.pb": "877664e1c929e975fede6820704d848b4e23c5d490a73d9214e0310313128607",
        "variables/variables.data-00000-of-00001":
            "1747bd85c86a81e7df4b77e8467a5c2878c3b04312cbca3e037718596cdd55bb",
        "variables/variables.index": "434075615c286e1a0cb856f2642d5bd910b50072ce42829e76b2f9573253c541",
    },
}


class ChecksumError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _skip() -> bool:
    return os.environ.get(SKIP_ENV, "").strip() not in ("", "0")


def _fail(msg: str):
    if _skip():
        print(f"  WARNING ({SKIP_ENV} set, continuing): {msg}")
        return
    raise ChecksumError(msg + f"\n  Delete the file and download it again. If the source was updated on "
                              f"purpose, set {SKIP_ENV}=1 after checking the new files.")


def expected_file(path: Path) -> str | None:
    return FILES.get(Path(path).name)


def verify_file(path: Path, expected: str | None = None) -> bool:
    """True if the file matches its pinned hash; False if no hash is pinned for it."""
    path = Path(path)
    expected = expected or expected_file(path)
    if expected is None:
        return False
    got = sha256(path.resolve())
    if got != expected:
        _fail(f"checksum mismatch for {path}\n  expected {expected}\n  got      {got}")
    return True


def expected_tree(folder: Path) -> dict | None:
    """Pinned files for a model folder, matched by the longest path suffix (None if unknown)."""
    parts = Path(folder).resolve().parts
    for key in sorted(TREES, key=len, reverse=True):
        k = tuple(Path(key).parts)
        if parts[-len(k):] == k:
            return TREES[key]
    return None


def verify_tree(folder: Path) -> bool:
    """True if every pinned file of a known model folder matches; False if the folder is unknown."""
    folder = Path(folder)
    pins = expected_tree(folder)
    if pins is None:
        return False
    for rel, expected in pins.items():
        f = folder / rel
        if not f.exists():
            _fail(f"missing model file {f}")
            continue
        got = sha256(f)
        if got != expected:
            _fail(f"checksum mismatch for {f}\n  expected {expected}\n  got      {got}")
    return True


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in ("verify", "verify-dir", "sha256"):
        print(__doc__)
        return 2
    cmd, paths = argv[0], [Path(p).expanduser() for p in argv[1:]]
    try:
        for p in paths:
            if cmd == "sha256":
                print(f"{sha256(p)}  {p}")
            elif cmd == "verify":
                print(f"  {'OK' if verify_file(p) else 'no pinned checksum'}: {p}")
            else:
                print(f"  {'OK' if verify_tree(p) else 'no pinned checksums'}: {p}")
    except ChecksumError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
