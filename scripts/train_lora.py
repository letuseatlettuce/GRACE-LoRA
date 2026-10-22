"""LoRA fine-tuning of a GRACE foundation model on GRACE-OFF (teacher) labels.

Pipeline:
  1. Load the student from its checkpoint (model.yaml + checkpoint).
  2. Align energy references: fit per-element shifts between student and teacher
     energies (gracemaker's `shift: auto`) and add them to the model.
  3. Freeze everything; attach LoRA adapters to the target weights
     (default: the `reducing_` ACE coefficients of I1, I2, rho), one per element in the data.
  4. Train the adapters (and the per-element energy shift) on teacher energies + forces.
  5. Merge the best adapters into the weights and export a SavedModel usable by TPCalculator.

`--mode full` trains the same target weights directly (no low-rank constraint): the
frozen-layer fine-tuning baseline from the GRACE paper, for comparison.

Usage:
    python scripts/train_lora.py --name lora_r8 --rank 8 --epochs 30
    python scripts/train_lora.py --name full_baseline --mode full --epochs 30
"""

import argparse
import json
import os
import time
from pathlib import Path

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorpotential import constants as tc
from tensorpotential.cli.data import (
    compute_fm_energy_shift_auto,
    inject_or_update_atomic_shift_in_model,
)
from tensorpotential.cli.prepare import construct_model_functions
from tensorpotential.data.databuilder import (
    GeometricalDataBuilder,
    ReferenceEnergyForcesStressesDataBuilder,
    construct_batches,
)
from tensorpotential.instructions.base import load_instructions, save_instructions_dict
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.tpmodel import extract_cutoff_and_elements, extract_cutoff_dict

import config
from lora import build_adapters

DEFAULT_TARGETS = ["I1/reducing_", "I2/reducing_", "rho/reducing_"]
SHIFT_VARIABLE = "tr_atomic_shift"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", default=config.STUDENT_CHECKPOINT_DIR,
                   help="student folder with model.yaml + checkpoint.index")
    p.add_argument("--data-dir", default=config.DATA_DIR, help="folder with train/val .pkl.gz")
    p.add_argument("--runs-dir", default=config.RUNS_DIR, help="parent folder for run outputs")
    p.add_argument("--name", default="lora", help="run name (sub-folder of runs-dir)")
    p.add_argument("--mode", choices=["lora", "full"], default="lora")
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                   help="substrings of variable names to adapt")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=None, help="LoRA alpha (default: 2*rank)")
    p.add_argument("--no-train-shift", action="store_true",
                   help="keep the per-element energy shift frozen")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None,
                   help="peak learning rate (default: 2e-3 lora, 5e-4 full)")
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--forces-weight", type=float, default=10.0)
    p.add_argument("--adapt-elements", nargs="*", default=None,
                   help="elements that get adapter rows (default: every element in the training "
                        "data); e.g. only the teacher's elements when replaying materials data")
    p.add_argument("--no-energy-shift", action="store_true",
                   help="labels already use the student's energy reference: skip the shift fit")
    p.add_argument("--monitor", nargs="*", default=[],
                   help="name=path.pkl.gz held-out sets evaluated every epoch (columns "
                        "<name>_f_mae_meV_A / <name>_e_mae_meV_atom in history.csv)")
    p.add_argument("--n-shift-structures", type=int, default=200,
                   help="structures used to fit the initial per-element energy shift")
    p.add_argument("--max-buckets", type=int, default=8,
                   help="batch shape buckets (fewer = fewer graph retraces)")
    p.add_argument("--jit", action="store_true", help="XLA-compile the training step (CPU/CUDA; not Metal)")
    p.add_argument("--gpu", action="store_true",
                   help="use the GPU (default: config.USE_GPU; Metal is untested and may fail)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_batches(df, element_map, cutoff, cutoff_dict, batch_size, max_buckets, signatures):
    builders = [
        GeometricalDataBuilder(element_map, cutoff=cutoff, cutoff_dict=cutoff_dict),
        ReferenceEnergyForcesStressesDataBuilder(),
    ]
    raw = construct_batches(df, builders, batch_size=batch_size, max_n_buckets=max_buckets,
                            verbose=False)
    return [{k: tf.convert_to_tensor(b[k], dtype=s.dtype) for k, s in signatures.items()}
            for b in raw]


def batch_errors(batch, pred):
    """Return (sum |dE|/atom, n_structures, sum |dF| per component, n_force_components)."""
    atom_mask = batch[tc.DATA_FORCE_WEIGHTS][:, 0].numpy() > 0
    struct_mask = batch[tc.DATA_ENERGY_WEIGHTS][:, 0].numpy() > 0
    smap = batch[tc.ATOMS_TO_STRUCTURE_MAP].numpy()
    n_struct = len(struct_mask)
    nat = np.bincount(smap[atom_mask], minlength=n_struct)

    de = np.abs(pred[tc.PREDICT_TOTAL_ENERGY].numpy()[:, 0]
                - batch[tc.DATA_REFERENCE_ENERGY].numpy()[:, 0])
    de = de[struct_mask] / nat[struct_mask]
    df = np.abs(pred[tc.PREDICT_FORCES].numpy() - batch[tc.DATA_REFERENCE_FORCES].numpy())
    df = df[atom_mask]
    return de.sum(), de.size, df.sum(), df.size


def evaluate(tp, batches):
    e_sum = e_n = f_sum = f_n = 0
    for b in batches:
        pred = tp.predict(b)
        es, en, fs, fn = batch_errors(b, pred)
        e_sum, e_n, f_sum, f_n = e_sum + es, e_n + en, f_sum + fs, f_n + fn
    return {"e_mae_meV_atom": 1000 * e_sum / e_n, "f_mae_meV_A": 1000 * f_sum / f_n}


def main():
    args = parse_args()
    use_gpu = args.gpu or config.USE_GPU
    if use_gpu and args.jit:
        raise SystemExit("--jit (XLA) does not work on the Apple Metal GPU; use one or the other")
    config.setup_device(use_gpu)
    tf.random.set_seed(args.seed)
    alpha = args.alpha if args.alpha is not None else 2.0 * args.rank
    lr = args.lr if args.lr is not None else (2e-3 if args.mode == "lora" else 5e-4)

    # monitor paths are read after os.chdir(run_dir) below: make them absolute first
    args.monitor = [f"{name}={Path(path).expanduser().resolve()}"
                    for name, _, path in (m.partition("=") for m in args.monitor)]
    ckpt_dir = config.require(args.checkpoint_dir, "STUDENT_CHECKPOINT_DIR").resolve()
    data_dir = config.require(args.data_dir, "DATA_DIR").resolve()
    run_dir = (config.require(args.runs_dir, "RUNS_DIR") / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)  # tensorpotential writes scratch files relative to the cwd
    with open("args.json", "w") as f:
        json.dump({**vars(args), "alpha": alpha, "lr": lr}, f, indent=2)

    train_df = config.load_frame(data_dir / "train.pkl.gz")
    val_df = config.load_frame(data_dir / "val.pkl.gz")
    print(f"Data: {len(train_df)} train / {len(val_df)} val structures")

    # --- 1. student + energy alignment ---------------------------------------------------
    instructions = load_instructions(str(ckpt_dir / "model.yaml"))
    cutoff, symbols, _ = extract_cutoff_and_elements(instructions)
    element_map = {str(s): i for i, s in enumerate(symbols)}
    cutoff_dict = extract_cutoff_dict(instructions)

    if args.no_energy_shift:
        shifts = {}
    else:
        print("Fitting per-element energy shifts (student -> teacher reference)...")
        with tf.device("/CPU:0"):  # this helper XLA-compiles its model; XLA has no Metal backend
            shifts = compute_fm_energy_shift_auto(train_df, str(ckpt_dir),
                                                  max_structures=args.n_shift_structures,
                                                  seed=args.seed)
        # The injected shift layer builds its per-element array from the dict values only,
        # so the dict must cover every element of the model (zeros for elements not in the data).
        full_shifts = {el: shifts.get(el, 0.0) for el in element_map}
        inject_or_update_atomic_shift_in_model(instructions, full_shifts, element_map)
    with open("energy_shifts.json", "w") as f:
        json.dump(shifts, f, indent=2)

    fit_config = {
        "loss": {
            "energy": {"weight": args.energy_weight, "type": "huber", "delta": 0.01},
            "forces": {"weight": args.forces_weight, "type": "huber", "delta": 0.01},
        },
        "optimizer": "Adam",
        "opt_params": {"learning_rate": lr, "amsgrad": True},
    }
    fns = construct_model_functions(fit_config)
    tp = TensorPotential(
        instructions, fit_config=fit_config, global_batch_size=args.batch_size,
        loss_function=fns.loss_fn, compute_metrics=fns.metrics_fn,
        model_train_function=fns.train_fn, model_compute_function=fns.compute_fn,
        param_dtype=tf.float32, jit_compile=args.jit, seed=args.seed,
    )
    tp.load_checkpoint(str(ckpt_dir / "checkpoint"), expect_partial=True, model_only=True)

    # --- 2. choose trainable variables ------------------------------------------------------
    patterns = list(args.targets) + ([] if args.no_train_shift else [SHIFT_VARIABLE])
    tp.model.set_trainable_variables(patterns)
    train_vars = tp.model.variables_to_train
    targets = [v for v in train_vars if SHIFT_VARIABLE not in v.name]
    others = [v for v in train_vars if SHIFT_VARIABLE in v.name]
    if not targets:
        raise SystemExit(f"No variables match --targets {args.targets}")

    data_elements = sorted({e for a in train_df.ase_atoms for e in a.get_chemical_symbols()})
    if args.adapt_elements:
        unknown = sorted(set(args.adapt_elements) - set(element_map))
        if unknown:
            raise SystemExit(f"--adapt-elements not in the model: {unknown}")
        data_elements = sorted(set(args.adapt_elements))
    element_rows = [element_map[e] for e in data_elements]
    n_total = sum(int(tf.size(v)) for v in tp.model.trainable_variables)

    if args.mode == "lora":
        adapters = build_adapters(targets, args.rank, alpha, len(element_map), element_rows,
                                  seed=args.seed)
        opt_vars = [f for a in adapters for f in a.factors] + others
        n_trained = sum(a.n_params for a in adapters)
    else:
        adapters = []
        opt_vars = targets + others
        # only rows of elements present in the data receive non-zero gradients
        n_trained = sum(int(tf.size(v)) // v.shape[0] * len(element_rows)
                        if v.shape[0] == len(element_map) else int(tf.size(v))
                        for v in targets)
    n_trained += sum(int(tf.size(v)) for v in others)
    print(f"Adapted elements: {data_elements}")
    print(f"Target tensors: {[v.name for v in targets]}")
    print(f"Trainable parameters: {n_trained:,} of {n_total:,} "
          f"({100 * n_trained / n_total:.2f}%)  [mode={args.mode}, rank={args.rank}, alpha={alpha}]")

    # --- 3. data ---------------------------------------------------------------------------
    sigs = tp.get_model_grad_signatures()
    train_batches = make_batches(train_df, element_map, cutoff, cutoff_dict, args.batch_size,
                                 args.max_buckets, sigs)
    val_batches = make_batches(val_df, element_map, cutoff, cutoff_dict, args.batch_size,
                               args.max_buckets, sigs)
    monitors = {}
    for spec in args.monitor:
        name, _, path = spec.partition("=")
        monitors[name] = make_batches(config.load_frame(Path(path)), element_map, cutoff,
                                      cutoff_dict, args.batch_size, args.max_buckets, sigs)

    def evaluate_monitors():
        row = {}
        for name, batches in monitors.items():
            m = evaluate(tp, batches)
            row[f"{name}_f_mae_meV_A"] = m["f_mae_meV_A"]
            row[f"{name}_e_mae_meV_atom"] = m["e_mae_meV_atom"]
        return row

    def monitor_text(row):
        return "".join(f"  {k[:-len('_f_mae_meV_A')]} F {v:7.1f}" for k, v in row.items()
                       if k.endswith("_f_mae_meV_A"))

    steps = args.epochs * len(train_batches)
    schedule = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=steps, alpha=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule, amsgrad=True)

    var_index = {v.ref(): i for i, v in enumerate(tp.model.variables_to_train)}
    target_idx = [var_index[v.ref()] for v in targets]
    other_idx = [var_index[v.ref()] for v in others]

    @tf.function(reduce_retracing=True)
    def apply_update(grads):
        if adapters:
            g = []
            for a, i in zip(adapters, target_idx):
                g += a.factor_grads(grads[i])
            g += [grads[i] for i in other_idx]
            optimizer.apply_gradients(zip(g, opt_vars))
            for a in adapters:
                a.merge()
        else:
            g = [grads[i] for i in target_idx + other_idx]
            optimizer.apply_gradients(zip(g, opt_vars))

    def snapshot():
        state = {v.name: v.numpy() for v in others}
        if adapters:
            for a in adapters:
                state.update(a.state())
        else:
            state.update({v.name: v.numpy() for v in targets})
        return state

    def restore(state):
        for v in others:
            v.assign(state[v.name])
        if adapters:
            for a in adapters:
                a.load_state(state)
                a.merge()
        else:
            for v in targets:
                v.assign(state[v.name])

    # --- 4. train --------------------------------------------------------------------------
    history = []
    base = evaluate(tp, val_batches)
    mon = evaluate_monitors()
    print(f"epoch   0  val E {base['e_mae_meV_atom']:8.2f} meV/atom  "
          f"F {base['f_mae_meV_A']:8.2f} meV/A   (shift-aligned base model)")
    history.append({"epoch": 0, "train_loss": np.nan, **base, "time_s": 0.0, **mon})
    if mon:
        print(f"          monitors:{monitor_text(mon)}")
    best, best_state = base["f_mae_meV_A"], snapshot()

    rng = np.random.default_rng(args.seed)
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        losses = []
        for i in rng.permutation(len(train_batches)):
            loss, _, grads, _ = tp.model_grad(train_batches[i])
            # gathered (per-element) weights give sparse IndexedSlices; densify so the
            # update step keeps fixed shapes and is traced only once
            apply_update([tf.convert_to_tensor(g) for g in grads])
            losses.append(float(loss))
        val = evaluate(tp, val_batches)
        dt = time.perf_counter() - t0
        mon = evaluate_monitors()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val, "time_s": dt,
                        **mon})
        flag = ""
        if val["f_mae_meV_A"] < best:
            best, best_state, flag = val["f_mae_meV_A"], snapshot(), "  *best"
        print(f"epoch {epoch:3d}  val E {val['e_mae_meV_atom']:8.2f} meV/atom  "
              f"F {val['f_mae_meV_A']:8.2f} meV/A  loss {np.mean(losses):.3e}  {dt:5.1f}s{flag}")
        if mon:
            print(f"          monitors:{monitor_text(mon)}")
        pd.DataFrame(history).to_csv("history.csv", index=False)

    # --- 5. export best --------------------------------------------------------------------
    restore(best_state)
    np.savez("best_trainable_state.npz", **{k.replace("/", "__"): v for k, v in best_state.items()})
    if adapters:
        # the adapter = best_trainable_state.npz (LoRA factors + trained energy shift) plus this
        # description; scripts/apply_adapter.py rebuilds base + adapter(s) from them
        with open("adapter.json", "w") as f:
            json.dump({
                "format": "grace-lora-adapter/1",
                "base_checkpoint_dir": str(ckpt_dir), "base_model": ckpt_dir.name,
                "weights_file": "best_trainable_state.npz",
                "rank": args.rank, "alpha": alpha,
                "targets": [a.name for a in adapters],
                "elements": data_elements, "element_rows": element_rows,
                "energy_shifts": shifts, "shift_variables": [v.name for v in others],
                "trainable_params": n_trained,
                "size_bytes": os.path.getsize("best_trainable_state.npz"),
            }, f, indent=2)
    save_instructions_dict("model.yaml", tp.model.instructions, param_dtype=tf.float32)
    tf.train.Checkpoint(model=tp.model).write(str(run_dir / "checkpoint"))
    tp.save_model(exact_path=str(run_dir / "saved_model"), jit_compile=args.jit)

    summary = {
        "mode": args.mode, "rank": args.rank, "alpha": alpha,
        "trainable_params": n_trained, "total_params": n_total,
        "val_before": base, "val_best_f_mae_meV_A": best,
        "train_time_s": time.perf_counter() - t_start,
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest val force MAE: {best:.2f} meV/A (start {base['f_mae_meV_A']:.2f})")
    print(f"Saved: {run_dir / 'saved_model'}  (load with TPCalculator(model=...))")


if __name__ == "__main__":
    main()
