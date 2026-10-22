"""Fetch MPtrj (Materials Project trajectory, PBE/PBE+U DFT) parquet shards: the materials
data for replay training, monitors and validation. Standard library only.

Shards (Hugging Face nimashoghi/mptrj, ~11 MB / ~11 MB / ~215 MB):
    val     -> val.parquet      validation + monitor materials (never trained on)
    test    -> test.parquet     replay training materials
    train0  -> train0.parquet   replay training materials (many more crystals)

Existing non-empty files are kept. With --from-dir, shards already on disk (any of the
names above, or the Hugging Face file names) are linked/copied instead of downloaded.

Usage:
    python scripts/fetch_materials.py                         # into <workdir>/data/materials
    python scripts/fetch_materials.py --from-dir ~/my_shards --copy
    python scripts/fetch_materials.py --shards val test
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checksums  # noqa: E402
import config  # noqa: E402

# pinned revision: the shard checksums in scripts/checksums.py belong to this commit
HF_BASE = f"https://huggingface.co/datasets/nimashoghi/mptrj/resolve/{checksums.MPTRJ_REVISION}/data/"
REMOTE = {"val": "val-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet",
          "train0": "train-00000-of-00008.parquet"}


def candidates(name: str, from_dir: Path) -> list:
    """Possible file names of shard `name` in a local folder."""
    return [from_dir / config.MATERIALS_SHARDS[name], from_dir / REMOTE[name]]


def link_or_copy(src: Path, dest: Path, copy: bool):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(src, dest)
    else:
        dest.symlink_to(src.resolve())


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


def fetch(out_dir: Path, shards, from_dir: Path | None = None, copy: bool = False) -> dict:
    """Make every requested shard available in out_dir. Returns {name: path}."""
    result = {}
    for name in shards:
        dest = out_dir / config.MATERIALS_SHARDS[name]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  exists: {dest}")
        elif from_dir and (src := next((c for c in candidates(name, from_dir) if c.exists()), None)):
            link_or_copy(src, dest, copy)
            print(f"  {'copied' if copy else 'linked'}: {src} -> {dest}")
        else:
            print(f"  downloading {name}: {HF_BASE + REMOTE[name]}")
            download(HF_BASE + REMOTE[name], dest)
        checksums.verify_file(dest)  # also covers existing and linked shards
        result[name] = dest
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=config.MATERIALS_DIR, help="target folder")
    p.add_argument("--shards", nargs="+", default=list(REMOTE), choices=list(REMOTE))
    p.add_argument("--from-dir", default=None, help="use shards already on disk")
    p.add_argument("--copy", action="store_true", help="copy instead of symlink (with --from-dir)")
    args = p.parse_args(argv)
    out = Path(args.out).expanduser()
    from_dir = Path(args.from_dir).expanduser() if args.from_dir else None
    try:
        fetch(out, args.shards, from_dir, args.copy)
    except checksums.ChecksumError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
