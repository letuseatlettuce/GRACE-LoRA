# Hardware notes: known platform issues and fixes

Read the section matching `python3 pipeline.py sysinfo` **before** setting up the environment.
Every issue below actually happened while building this skill, on two machines:
- **Windows 11 + WSL2 (Ubuntu 24.04):** Intel i7-13650HX, 10 threads visible to WSL,
  11.7 GB RAM visible to WSL, NVIDIA RTX 5070 Laptop GPU (8 GB); Python 3.12.
- **MacBook Pro, Apple M4 Pro:** 24 GB unified memory; Python 3.11.

All of this project's results were produced on the WSL2 laptop's **CPU with XLA**
(`device = cpu-jit`).

Contents: [Windows / WSL2](#windows--wsl2) · [NVIDIA GPUs (incl. RTX 50xx)](#nvidia-gpus-incl-rtx-50xx-blackwell) ·
[macOS Apple Silicon](#macos-apple-silicon-m1m4) · [Linux CPU-only](#linux-cpu-only) · [Memory sizing (all platforms)](#memory-sizing-all-platforms)

---

## Windows / WSL2
Native Windows is **not supported**. Use WSL2 with Ubuntu; `sysinfo` reports `[WSL]`.

| Issue | Symptom | Fix |
|---|---|---|
| Project or outputs under `/mnt/c/...` or OneDrive | Slow file access; OneDrive syncing checkpoints mid-write; a folder deleted and recreated straight away can fail (`git clone` failed once right after `rm -rf` in a OneDrive folder) | Keep the repo, `.venv` and the workdir in the Linux file system (`~/…`). Use `configure --workdir ~/grace-lora-work` |
| WSL only sees part of the RAM | WSL2 gets ~50% of Windows RAM by default (11.7 GB here); training at batch size 16 ran out of memory | Use batch size 8 with ≤ 4 buckets (default when RAM < 24 GB). To give WSL more: create `%UserProfile%\.wslconfig` with `[wsl2]` / `memory=16GB` (and optionally `swap=8GB`), then `wsl --shutdown` in PowerShell |
| System Python refuses `pip install` | `externally-managed-environment` (PEP 668) | Never install system-wide. `pipeline.py setup-env` creates `.venv` |
| CUDA log noise | `Could not find cuda drivers`, `Unable to register cuDNN/cuFFT/cuBLAS factory`, `computation placer already registered`, NUMA notes | Harmless with the CPU setup. Ignore them; `pipeline.py` filters most |
| Paths shown to Windows | Windows Explorer can't find `~/.cache/grace` | Use `\\wsl.localhost\Ubuntu-24.04\home\<user>\.cache\grace`; `wslpath -w <path>` prints it |
| Background runs and pausing | A run frozen with SIGSTOP (`kill -STOP`) died when its parent shell exited (exit 147) and lost its results | Never pause; stop with Ctrl+C or kill, and re-run (experiments resume from the ledger) |
| Commands are case-sensitive | `GRACE_models: command not found` | `grace_models` (lowercase), from the activated `.venv` (`source .venv/bin/activate`) |

## NVIDIA GPUs (incl. RTX 50xx Blackwell)
- **Recommendation: use the CPU with XLA (`device = cpu-jit`).** That is the tested path.
- **The RTX 5070 was not used.** TensorFlow 2.19's CUDA build predates Blackwell (sm_120); GPU
  support for the newest cards is uncertain (slow JIT compilation or no detection).
- **CPU+XLA was fast enough:** ~0.2 s per structure per epoch, i.e. ~55 min for the default
  1,600-structure, 10-epoch experiment.
- **The CUDA bundle:** on Linux, `tensorpotential` declares `tensorflow[and-cuda]`, which pulls
  ~4 GB of NVIDIA packages. `pipeline.py setup-env --cpu-only` (automatic when no NVIDIA GPU is
  detected) avoids that by installing `tensorpotential` without its CUDA extra.
- **To try the GPU anyway (untested):**
  1. install with CUDA (`setup-env` without `--cpu-only` on a machine with an NVIDIA GPU);
  2. run `python3 pipeline.py smoke --race`: it times 2 training epochs on the GPU against
     CPU+XLA and picks the faster;
  3. if the GPU run errors out, the race keeps the CPU.

  Older, well-supported NVIDIA GPUs (Ampere/Ada) are the better candidates.
- **GPU memory:** 8 GB of VRAM is tight for batch size 16; start at 8.

## macOS Apple Silicon (M1–M4)
Intel Macs are **not supported** (TensorFlow 2.19 ships Apple Silicon wheels only).

| Issue | Symptom | Fix |
|---|---|---|
| TensorFlow / Metal versions | `tensorflow-metal` 1.2.0 is broken with TF 2.20 | `requirements.txt` pins TF **2.19.1** + `tensorflow-metal` **1.2.0** + `tf-keras` **2.19.0**. Don't upgrade |
| Metal cannot run XLA models | `NotFoundError: could not find registered platform with id: 0x…` when evaluating a model | The downloaded GRACE SavedModels are XLA-compiled and XLA has no Metal backend. Labelling, evaluation and validation **must run on the CPU**: keep `USE_GPU = False`, which is the default and is what `config.setup_device()` enforces. Last resort: `pip uninstall -y tensorflow-metal` |
| No float64 on Metal | Crashes or wrong results with fp64 models on the GPU | fp32 GRACE models only on Metal; on the CPU, use the fp64 teacher `saved_model` |
| `casted_model` vs `saved_model` | `Can not cast TensorSpec(... float64 ...) to ... float32` | Point the teacher at `.../seed/1/saved_model` (fp64), never `casted_model` (fp32) |
| Training on the Metal GPU | Untested | `train_lora.py --gpu` / `device = gpu` may work (the training graph isn't XLA-compiled), but run `smoke --race` first and expect CPU fallbacks. `--gpu` and `--jit` cannot be combined |
| Unified memory | 24 GB is shared between CPU and GPU | Batch size 8 / 4 buckets is the safe default below 24 GB; 16 is possible on 32 GB+ |
| The Mac sleeps mid-run | Long runs stop when the lid closes or the Mac idles | `pipeline.py` wraps long runs in `caffeinate -i`; keep the lid open or use external power + display |
| Wrong Python environment | pip conflicts with `pyiron`, `atomistics`, `sphinx-parser`, `lammpsparser` and `semantikon`; tensorpotential 0.5.9 replaced | The install landed in conda `base` or **another project's venv**. Run `conda deactivate`, then use the project's own `.venv` (`setup-env`). Never reuse another project's environment |
| Python 3.11 missing | `No Python 3.10-3.12 found` | `brew install python@3.11` |
| Hidden cache folder | `~/.cache/grace` not visible in Finder | Press Cmd+Shift+. in Finder, or `open ~/.cache/grace` |
| Stale scripts | An old copy of a script at the repo root was run (e.g. `check_models.py` with the `casted_model` default) | Run `python3 pipeline.py …` or `scripts/<name>.py` from the repo root only |

## Linux CPU-only
- **Install:** `setup-env` installs the CPU wheel automatically when no NVIDIA GPU is found.
- **Device:** use `device = cpu-jit` (XLA), about 2.2× faster than without (`cpu`).
- **Time:** expect ~0.2 s per structure per epoch on a modern laptop CPU. Scale the time budget
  with core count.

## Memory sizing (all platforms)
Measured peak RAM during training, CPU+XLA, 2-layer medium student:

| Batch size | Max buckets | Peak RAM | Result |
|---|---|---|---|
| 16 | 8 | > 14 GB | swapped and stalled on an 11.7 GB machine (killed) |
| 8 | 4 | 6.3 GB | stable (all results in this project) |

- **The default follows RAM:** `sysinfo` / `configure` pick 8 / 4 below 24 GB of RAM and 16 / 8
  above.
- **Measuring:** the smoke test reports peak memory; if it's above ~70% of RAM, lower
  `batch_size` / `max_buckets` in `pipeline.py settings` before the loop.
- **If `self_heal` is on:** an out-of-memory failure automatically halves the batch size and
  retries.
