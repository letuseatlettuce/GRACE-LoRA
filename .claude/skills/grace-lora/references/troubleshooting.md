# Troubleshooting: every problem met so far, with its fix

Check here before changing any code. If something isn't listed, stop and report the full error
to the user.

| Symptom | Cause | Fix |
|---|---|---|
| `NotFoundError: could not find registered platform with id: 0x…` (macOS) | An XLA-compiled GRACE SavedModel ran on the Apple Metal GPU; XLA has no Metal backend | Keep `USE_GPU = False` (the default). Labelling, validation and inference always run on the CPU. Only `train_lora.py --gpu` may use Metal, and never together with `--jit`. Last resort: `pip uninstall -y tensorflow-metal` |
| `--jit (XLA) does not work on the Apple Metal GPU` | `--gpu` and `--jit` both given | Use one: CPU + `--jit` (default, fast) or `--gpu` |
| `Can not cast TensorSpec(... float64 ...) to TensorSpec(... float32 ...)` | The teacher path points to GRACE-OFF's `casted_model` (fp32), but the calculator feeds fp64 | Use `.../seed/1/saved_model` (fp64). `configure` refuses `casted_model` |
| Training stalls at startup; RAM 100%, swap full, RSS > 11 GB | Batch size 16 + XLA compiling several batch shapes of the second-derivative graph (> 14 GB) | Batch size 8, `--max-buckets 4` (the default on < 24 GB RAM). Peak ≈ 6.3 GB. Kill the stuck job; don't wait for it |
| `NotImplementedError: This functionality is not yet fully implemented` | tensorpotential's built-in LoRA hooks (`TensorPotential.enable_lora_adaptation`) are unfinished | Not used here: LoRA is implemented in `scripts/lora.py` (exact chain-rule LoRA on the merged weights) |
| `indices[0] = 51 is not in [0, 10)` during prediction | The injected energy-shift layer builds its array from the dict *values*, so a dict with only the data's 10 elements gets mis-indexed | Already handled: the shift dict is padded to all 89 elements. If editing training code, keep that |
| `WARNING: N out of the last M calls to apply_update triggered tf.function retracing` | Per-element weights are gathered, so their gradients are sparse `IndexedSlices` with varying shapes | Already handled: gradients are converted to dense tensors before the update step |
| `AssertionError` in `export_to_yaml` (`LinearRadialFunction`) | `TPModel.export_to_yaml` is the old FS-model exporter | Use `save_instructions_dict` (already done in `train_lora.py`) |
| Zenodo `SPICE_V2.0_cleaned.xyz` looks like a training set but equals the test set | On record 18662355 it is byte-identical to `GRACE_test_set.xyz` | Train from `biased_test_set.xyz`, test on `GRACE_test_set.xyz`; `SPICE_V2.0_cleaned.xyz` is never used |
| "Removed 59 pool structures that are also in the test set" | `biased_test_set.xyz` and `GRACE_test_set.xyz` share 59 structures | Expected; `make_dataset.py` removes them so the test set stays unseen |
| MPtrj test and val shards share ~1000 materials (different frames) | Hugging Face splits are per frame, not per material | Training materials exclude every mp_id in the val shard (`--exclude`); one frame per material |
| `GRACE_models: command not found` / `which grace_models` prints nothing | The command is lowercase, and it lives in the venv | `source .venv/bin/activate` then `grace_models list`, or `.venv/bin/grace_models list` |
| `source .venv/bin/activate: No such file or directory` | No venv in this folder (or it was created elsewhere, e.g. `--venv`) | `python3 pipeline.py setup-env`, or use the venv path stored in `pipeline.json` |
| `externally-managed-environment` from pip | pip run against the system Python | Activate the project venv first |
| pip "dependency resolver" ERROR lines about pyiron/atomistics/sphinx-parser | Installed into a foreign environment (conda base or another project's venv) | Deactivate it and use the project `.venv`. Other projects' packages may now need restoring |
| `No module named 'config'` / `'lora'` | A script copied out of `scripts/` | Run `python scripts/<name>.py` from the repository root; the scripts import each other |
| Background job "paused" with `kill -STOP` later found dead (exit 147) | A stopped child is killed when its parent shell exits | Don't pause runs; kill and restart instead. `experiment` and `run` steps are resumable or cheap to redo |
| Python output missing from a log file for minutes | Python buffers stdout when it isn't a terminal | Expected. Watch `history.csv` in the run folder (written every epoch), or set `PYTHONUNBUFFERED=1` (`pipeline.py` already does) |
| `E0000 … cuda_executor … Error loading CUDA libraries. GPU will not be used` | TensorFlow probing for CUDA on a machine without CUDA libraries | Harmless log noise, like the NUMA, absl, oneDNN and `squeeze_dims` messages |
| Validation "E aligned" much larger than raw energy error on diverse materials | Fitting an offset for every element on a small calibration split is under-determined | Fixed: only the adapted elements' offsets are refitted |
| GRACE-OFF materials MAE ~10⁶ meV/Å, R² hugely negative | Real: GRACE-OFF (trained on molecules) explodes on some crystals (solid H, O₄, P₂, …) | Report the **median** per-structure error (~665 meV/Å) and coverage (1.7%) alongside the mean |
| Trained model worse than base on oxides/nitrides/molecular crystals | Forgetting: the adapter changed the shared H/O/C/N… weights | Use replay (the default in `experiment`); raise `n_materials` / `n_only_adapted` |
| Checkpoint folders with only `best_checkpoint.*` + `run.json` (from another training script) | No `model.yaml` / SavedModel | `scripts/export_checkpoint.py <ckpt>/best_checkpoint` rebuilds a SavedModel using the base student's `model.yaml` |
| OneDrive / iCloud / `/mnt/c` project folder: slow, or git and checkpoints misbehave | Sync clients and the Windows file-system bridge | Put the repo and `--workdir` in the Linux / home file system (e.g. `~/grace_work`) |
| The chosen teacher is a MACE / PyTorch model (e.g. `MACE-OFF24_medium.model` from the grace-off repo) | Only GRACE (tensorpotential SavedModel) teachers are supported; labelling uses `TPCalculator` | Choose a GRACE teacher (default GRACE-OFF-2L-medium; `teacher_tools.py recommend`), or use DFT labels |
| Teacher path ends in `casted_model` | GRACE-OFF ships an fp32 `casted_model` next to the fp64 `saved_model`; the fp32 one crashes on fp64 inputs | Point `--teacher` at `.../seed/1/saved_model`; `teacher_tools.py resolve` prefers it |
| `validate` reports teacher or student MISSING elements | The data contains elements the model has no parameters for | Pick a teacher covering all data elements (`teacher_tools.py recommend --elements ...`), filter those structures, or use DFT labels |
| A paper was given as the teacher; `resolve` returns `kind: publication, ok: false` | A publication isn't a model | Find its code/data-availability link (web search, landing page, SI) and resolve that. If none exists, tell the user and use a catalog teacher or DFT labels |

Platform-specific problems (WSL2 memory and OneDrive paths, the CUDA bundle, RTX 50xx, Metal/XLA, conda/venv mix-ups) are collected per platform in [hardware.md](hardware.md).
