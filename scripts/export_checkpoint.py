"""Export a raw GRACE training checkpoint (e.g. best_checkpoint.index/.data) to a SavedModel
that TPCalculator, evaluate.py and validate_model.py can load.

The checkpoint must hold the full (merged) model weights of the student architecture; the
architecture comes from the student's model.yaml. Also reports which weight tensors differ
from the base student, i.e. what the training changed.

Usage:
    python scripts/export_checkpoint.py checkpoint_10/best_checkpoint --out checkpoint_10/saved_model
"""

import argparse
import os
import tempfile
from pathlib import Path

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import tensorflow as tf
from tensorpotential.instructions.base import load_instructions
from tensorpotential.tensorpot import TensorPotential

import config


def changed_tensors(checkpoint: str, base_checkpoint: str):
    """(name, n_values, n_changed, max_abs_change) for every tensor that differs."""
    reader, base = tf.train.load_checkpoint(checkpoint), tf.train.load_checkpoint(base_checkpoint)
    base_names = set(base.get_variable_to_shape_map())
    rows = []
    for name in sorted(reader.get_variable_to_shape_map()):
        if name not in base_names or "CHECKPOINTABLE" in name:
            continue
        a, b = reader.get_tensor(name), base.get_tensor(name)
        if a.dtype.kind not in "fc" or a.shape != b.shape:
            continue
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        n_changed = int((diff > 1e-7).sum())
        if n_changed:
            rows.append((name.replace("model/instructions/", "").replace("/.ATTRIBUTES/VARIABLE_VALUE", ""),
                         a.size, n_changed, float(diff.max())))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="checkpoint prefix, e.g. checkpoint_10/best_checkpoint")
    p.add_argument("--student-checkpoint-dir", default=config.STUDENT_CHECKPOINT_DIR,
                   help="base student folder with model.yaml + checkpoint (architecture + diff)")
    p.add_argument("--out", help="SavedModel output folder (default: <checkpoint folder>/saved_model)")
    p.add_argument("--no-jit", action="store_true", help="export without XLA compilation")
    args = p.parse_args()
    config.setup_device()

    ckpt = str(Path(args.checkpoint).expanduser().resolve())
    if not Path(ckpt + ".index").exists():
        raise SystemExit(f"No checkpoint at {ckpt} (expected {ckpt}.index)")
    student = config.require(args.student_checkpoint_dir, "STUDENT_CHECKPOINT_DIR").resolve()
    out = Path(args.out).expanduser().resolve() if args.out else Path(ckpt).parent / "saved_model"

    rows = changed_tensors(ckpt, str(student / "checkpoint"))
    total = sum(r[1] for r in rows)
    print(f"\nTensors changed vs base student: {len(rows)}")
    for name, size, n_changed, max_change in rows:
        print(f"  {name:28s} {n_changed:>10,} of {size:>10,} values changed (max |dw| {max_change:.3g})")
    print(f"  total values in changed tensors: {total:,}")

    instructions = load_instructions(str(student / "model.yaml"))
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # TensorPotential writes scratch folders relative to the cwd
        try:
            tp = TensorPotential(instructions, param_dtype=tf.float32, jit_compile=False)
            tp.load_checkpoint(ckpt, expect_partial=True, model_only=True, verbose=True)
            tp.save_model(exact_path=str(out), jit_compile=not args.no_jit)
        finally:
            os.chdir(cwd)
    print(f"\nSaved: {out}  (load with TPCalculator(model=...))")


if __name__ == "__main__":
    main()
