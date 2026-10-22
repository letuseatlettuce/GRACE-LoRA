"""Build a usable GRACE model from the frozen base plus one or more LoRA adapters.

An adapter is a train_lora.py run folder (adapter.json + best_trainable_state.npz, ~5 MB).
    one adapter:   W = W0 + dW_k                    (the specialised model for domain k)
    several:       W = W0 + sum_k lambda_k * dW_k   (merged model; --weights, default 1 each)
Per-element energy offsets are combined the same way. Adapters trained on overlapping
elements change the same weight rows: that merge is approximate and reported as a warning.

Usage:
    python scripts/apply_adapter.py runs/quick/OFF_lora_r8 --out models/grace_OFF
    python scripts/apply_adapter.py runs/quick/OFF_lora_r8 runs/quick/B_lora_r8 --out models/grace_OFF+B
    python scripts/apply_adapter.py runs/quick/OFF_lora_r8 --info      # describe only
"""

import argparse
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import tensorflow as tf
from tensorpotential.cli.data import inject_or_update_atomic_shift_in_model
from tensorpotential.instructions.base import load_instructions
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import extract_cutoff_and_elements

import config
from lora import LoRAAdapter


def read_adapter(run_dir: Path, element_map: dict) -> dict:
    """adapter.json, or (older runs) the same information rebuilt from the run's files."""
    weights = np.load(run_dir / "best_trainable_state.npz")
    state = {k.replace("__", "/"): weights[k] for k in weights.files}
    if (run_dir / "adapter.json").exists():
        meta = json.loads((run_dir / "adapter.json").read_text())
    else:
        args = json.loads((run_dir / "args.json").read_text())
        if args.get("mode", "lora") != "lora":
            raise SystemExit(f"{run_dir} is a full fine-tuning run, not a LoRA adapter")
        shifts = json.loads((run_dir / "energy_shifts.json").read_text())
        elements = sorted(shifts)  # shifts are fitted for exactly the elements in the data
        meta = {"rank": args["rank"], "alpha": args["alpha"], "elements": elements,
                "element_rows": [element_map[e] for e in elements], "energy_shifts": shifts,
                "targets": sorted({k.rsplit("/", 1)[0] for k in state if k.endswith(("/A", "/B"))}),
                "shift_variables": [k for k in state if not k.endswith(("/A", "/B"))]}
    meta["state"] = state
    meta["dir"] = run_dir
    meta["size_bytes"] = os.path.getsize(run_dir / "best_trainable_state.npz")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapters", nargs="+", help="train_lora.py run folders (LoRA mode)")
    p.add_argument("--weights", nargs="*", type=float, help="merge weight per adapter (default 1)")
    p.add_argument("--base-checkpoint-dir", default=config.STUDENT_CHECKPOINT_DIR)
    p.add_argument("--out", help="SavedModel output folder")
    p.add_argument("--info", action="store_true", help="only describe the adapters")
    args = p.parse_args()
    config.setup_device()

    base_dir = config.require(args.base_checkpoint_dir, "STUDENT_CHECKPOINT_DIR").resolve()
    instructions = load_instructions(str(base_dir / "model.yaml"))
    _, symbols, _ = extract_cutoff_and_elements(instructions)
    element_map = {str(s): i for i, s in enumerate(symbols)}
    adapters = [read_adapter(Path(a).expanduser().resolve(), element_map) for a in args.adapters]
    weights = args.weights or [1.0] * len(adapters)
    if len(weights) != len(adapters):
        raise SystemExit("--weights needs one value per adapter")

    print(f"Base: {base_dir.name}")
    for a, w in zip(adapters, weights):
        print(f"  adapter {a['dir'].name}: rank {a['rank']}, {len(a['targets'])} tensors, "
              f"elements {' '.join(a['elements'])}, {a['size_bytes'] / 1e6:.1f} MB, weight {w}")
    for a, b in combinations(adapters, 2):
        shared = sorted(set(a["elements"]) & set(b["elements"]))
        if shared:
            print(f"  WARNING: {a['dir'].name} and {b['dir'].name} both adapt {' '.join(shared)}: "
                  "merged rows are an approximation (consider one adapter per system instead)")
    if args.info:
        return
    if not args.out:
        raise SystemExit("--out is required to build a model")

    # energy references: injected per-element offsets add up across adapters
    offsets = {el: 0.0 for el in element_map}
    for a, w in zip(adapters, weights):
        for el, v in a["energy_shifts"].items():
            offsets[el] += w * v
    inject_or_update_atomic_shift_in_model(instructions, offsets, element_map)

    out = Path(args.out).expanduser().resolve()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # TensorPotential writes scratch folders relative to the cwd
        try:
            tp = TensorPotential(instructions, param_dtype=tf.float32, jit_compile=False)
            tp.load_checkpoint(str(base_dir / "checkpoint"), expect_partial=True, model_only=True)
            variables = {v.name.split(":")[0]: v for v in tp.model.variables}

            # low-rank updates: W = W0 + sum_k w_k * dW_k
            targets = sorted({t for a in adapters for t in a["targets"]})
            for name in targets:
                var = variables[name]
                total = tf.zeros_like(var)
                for a, w in zip(adapters, weights):
                    if name not in a["targets"]:
                        continue
                    lora = LoRAAdapter(var, a["rank"], a["alpha"], element_rows=a["element_rows"]
                                       if var.shape[0] == len(element_map) else None)
                    lora.load_state(a["state"])
                    total += w * lora.delta()
                var.assign(var + total)

            # trained energy shift: base + sum_k w_k * (adapter_k - base)
            for name in sorted({s for a in adapters for s in a["shift_variables"]}):
                var = variables[name.split(":")[0]]
                base_value = var.numpy()
                delta = sum(w * (a["state"][name] - base_value)
                            for a, w in zip(adapters, weights) if name in a["state"])
                var.assign(base_value + delta)

            tp.save_model(exact_path=str(out), jit_compile=True)
        finally:
            os.chdir(cwd)
    (out / "adapters.json").write_text(json.dumps(
        {"base": str(base_dir), "adapters": [str(a["dir"]) for a in adapters], "weights": weights},
        indent=2))
    print(f"\nSaved: {out}  (load with TPCalculator(model=...))")


if __name__ == "__main__":
    main()
