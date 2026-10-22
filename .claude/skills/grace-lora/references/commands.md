# Command reference

## Contents
1. pipeline.py: setup steps
2. pipeline.py: experiment loop, validation, report
3. Scripts (run with the venv's Python)
4. Runtime layout

**How to run things:** run everything from the repository root.
- **`pipeline.py`** uses only the standard library: `python3 pipeline.py <step>`.
  `--venv <dir>` (or the environment variable `GRACE_PIPELINE_VENV`) uses an existing venv instead
  of `./.venv`.
- **Scripts** need the venv: `.venv/bin/python scripts/<name>.py`.

## 1. pipeline.py: setup
| Command | What it does |
|---|---|
| `sysinfo [--json]` | Detect OS, CPU, RAM, GPU (Apple/NVIDIA), WSL, disk and Python versions; print blockers and recommendations (batch size, device) |
| `setup-env [--python PATH] [--cpu-only]` | Create `.venv` (Python 3.10–3.12) and install `requirements.txt`. On Linux without NVIDIA, installs CPU-only TensorFlow. Checks the pinned versions |
| `catalog` | List students (GRACE foundation models with a checkpoint), teachers (GRACE-OFF-{1L,2L}-{small,medium,large}, `grace:<name>`, a local SavedModel) and data options |
| `configure --student S --teacher T --data spice\|POOL.xyz [--test TEST.xyz] [--student-model DIR] [--device cpu-jit\|cpu\|gpu] [--batch-size N] [--max-buckets N] [--workdir DIR] --yes` | Write `pipeline.json` (paths, device, batch settings, workdir). `--teacher` takes a GRACE-OFF variant name, `grace:<foundation model>`, or a GRACE `saved_model` path (never `casted_model`). Only GRACE teachers are supported; non-default teachers are untested |
| `fetch` | Download what's missing: student model and checkpoint (`grace_models`), teacher (sparse clone of heid-lab/grace-off), SPICE xyz (Zenodo) |
| `validate` | Check files, library versions, element coverage (data ⊆ teacher ∩ student) and dataset sizes |
| `smoke [--race]` | Tiny end-to-end test (label 160 → 3 epochs → evaluate) with PASS/FAIL checks, a timing/memory benchmark and estimates; `--race` also times the GPU |
| `estimate` | Re-print the time estimates from the last benchmark |
| `run --plan pilot\|quick\|solid [...] --yes` | Older fixed plans: dataset → jobs → evaluation → report.md. The autonomous loop replaces this |
| `auto` / `wizard` / no step | `auto`: everything non-interactively from the YOUR CHOICES block at the top of `pipeline.py`. `wizard`: interactive |

## 2. pipeline.py: experiment loop, validation, report
| Command | What it does |
|---|---|
| `prepare [--n-organic-pool 2000] [--n-test 300]` | Idempotent. Builds the teacher-labelled organic pool (`<workdir>/data/organic/{train,val,test}.pkl.gz`), downloads the MPtrj shards (`data/materials/{val,test,train0}.parquet`) and builds the three monitor sets (`data/monitors/{organic,materials,only_adapted}.pkl.gz`) |
| `experiment --set key=value ... --rationale "TEXT" [--parent EXP_ID]` | Run ONE experiment: build its replay data, train, record it in the ledger. The last line of output is `RESULT:{json}` |
| `ledger [--json]` | Show all experiments: config, monitors (organic_f, materials_f, only_adapted_f), relative change vs base, feasible, and the best one |
| `validate-exp --exp <id\|best\|base>` | Full validation: organic (all 300 test molecules, vs DFT and teacher) + materials (MPtrj val shard, groups A/B/C) into `<workdir>/validation/<exp>/` |
| `settings [--json] [--explain KEY] [--set KEY=VALUE ...] [--reset [KEY ...]] [--confirm]` | **Step 0.** Show the settings panel (models & data, training, objective & budget, validation, agent features), explain one setting, change or reset stored defaults, and `--confirm` once the user approves (sets `settings_confirmed`). Loop moves don't change stored settings: they use `experiment --set`. See `settings.md` |
| `validate-exp --exp <id\|best\|base> --full` | Full-size validation (organic 300; materials A 200 / B 500 / C all ≤ 100 atoms), the default for the final best model |
| `options [--enable F ...] [--disable F ...] [--set max_subagents=N] [--json]` | Show or change the optional agent features in `pipeline.json` → `agent_options`: `critic`, `self_heal`, `predictions`, `cockpit`, `subagents` (master switch), `max_subagents` (default 3). **All off by default**; enable only on the user's request. See `agent-options.md` |
| `cockpit [--watch SECONDS] [--out FILE]` | Write `<workdir>/report/cockpit.html` (self-contained, auto-refreshing): ledger, monitor curves, status. Run `--watch 30` in a background shell while the loop runs (needs `cockpit` enabled) |
| `validate-exp --exp <id\|best\|base> --quick` | Quick variant: ~5 min instead of ~15 (≈100 molecules, ≈260 materials). Useful for one-off checks while the loop is stopped |
| `validate-models --model NAME=PATH [--model ...] [--no-base] [--no-teacher] [--quick] [--organic PATH] [--materials PATH] [--out DIR]` | Validation only, no loop. PATH = a SavedModel folder, a `train_lora` run folder, an adapter folder (`adapter.json` + `best_trainable_state.npz`; rebuilt from base + adapter), or a raw checkpoint prefix / a folder with `best_checkpoint.index` (exported automatically). Base and teacher are reference rows unless excluded. Needs `prepare` first, or explicit `--organic` / `--materials`. Output: `<workdir>/validation/adhoc_<timestamp>/{organic/, materials/, summary.md, figures/}` |
| `report [--exps ID ...]` | Write `<workdir>/report/report.md` + figures: summary, experiment timeline, decision log, heatmap, trade-off scatter, monitor curves, parity, Pareto |

**`experiment` config keys (defaults):**
- `n_organic 1000` (taken from the start of the pool's train split), `n_materials 600`,
  `n_materials_val 100`, `n_only_adapted 170`;
- `rank 8`, `alpha` = 2×rank, `epochs 10`, `batch_size 8`, `max_buckets 4`, `lr` (default 2e-3);
- `max_atoms 30`, `oversample 3.0` (weight for H/F-containing materials), `seed 0`;
- `device cpu-jit`, `adapt_elements` = the teacher's elements.

`n_materials=0` gives an organic-only (no-replay) run.

**Optional `experiment` flags:**
- `--self-heal` is added automatically when `self_heal` is on. It retries OOM, Metal/XLA and NaN
  failures at most twice, and records `healing`.
- `--hypothesis "TEXT" --predict organic_f=X --predict materials_f=Y --predict only_adapted_f=Z`
  is used with `predictions` on, only once ≥ 2 experiments have finished.

**Stopping and resuming the loop:** each `experiment` is one process. To stop, kill it
(never `kill -STOP`); its ledger entry becomes `interrupted`. Resume by reading `ledger` and
running the next `experiment`.

**The ledger** (`<workdir>/experiments/ledger.json`):
- **The objective:** `max_materials_increase` (default 0.15) and `max_only_adapted_increase`
  (default 0.75).
- **Base monitor values**, then one entry per experiment: config, rationale, metrics at the
  selected epoch, relative change vs base, feasible, score (organic_f; lower is better).
- **Change the objective:** `.venv/bin/python scripts/ledger.py --workdir <workdir> set-objective key=value`.
- **Other `ledger.py` subcommands** (`--workdir` goes before the subcommand):
  - `show [--json]`, `best`, `get <id>`;
  - `annotate <id> review "TEXT" --verdict accept|doubt|reject`, used by the critic;
  - `predictions`: predicted vs actual, used with the predictions feature.

## 3. Scripts
| Script | Use |
|---|---|
| `teacher_tools.py --workdir W resolve <spec>` | Resolve a teacher spec (builtin name, `grace:<name>`, local path, git URL, DOI/publication URL). `RESULT:{kind, model_dir, candidates:[{path, elements, cutoff, precision, is_casted}], elements, ok, needs_fetch, notes}`. Publications return `ok:false` with a note asking for the code/data link. Prefer the fp64 `saved_model` over `casted_model` |
| `teacher_tools.py check <saved_model_dir>` | Check a GRACE SavedModel: `RESULT:{ok, elements, cutoff, precision, warnings}` |
| `teacher_tools.py --workdir W recommend --elements EL ... [--domain "TEXT"]` | Ranked teachers (`covers_all`, `missing`, `domain`, `level_of_theory`, `why`), a replay suggestion (`materials_with_elements`, `only_these_elements`, suggested `n_materials` / `n_only_adapted`), and notes (e.g. "no catalog teacher covers all elements", "no specialist needed") |
| `domain_distance.py name=PATH ... [--builtin crystals water] [--n 200] [--out FILE.json]` | Feasibility: base GRACE's extrapolation grade γ per dataset and per element, with a verdict (close / moderate / far) |
| `check_models.py [--student DIR] [--teacher DIR]` | Load both models and compare forces on 5 molecules (sanity check) |
| `make_dataset.py --out DIR --n-train N --n-val N --n-test N [--pool XYZ] [--test XYZ]` | Teacher-label structures. Output columns: `ase_atoms, energy, energy_corrected, forces` (teacher) and `dft_energy, dft_forces` |
| `make_replay_dataset.py --organic-dir DIR --materials TEST.parquet TRAIN0.parquet --exclude VAL.parquet --n-organic N --n-materials N --n-materials-val N --n-only-adapted N --out DIR [--no-monitors]` | Mix teacher-labelled organics (energies moved to the base reference) with DFT materials; H/F (and, for only-adapted, O) oversampled; val-shard mp_ids excluded |
| `train_lora.py --data-dir D --runs-dir R --name N [--rank 8] [--epochs 10] [--batch-size 8] [--max-buckets 4] [--jit\|--gpu] [--no-energy-shift] [--no-train-shift] [--adapt-elements ...] [--monitor name=PATH ...] [--mode full]` | Train one LoRA (or full-rows baseline) run. Output: `history.csv` (per-epoch val + monitor errors), `summary.json`, `adapter.json`, `best_trainable_state.npz`, `saved_model/` |
| `validate_model.py --trained name=DIR ... --structures TEST.pkl.gz\|XYZ [--n 0] --out DIR` | Organic validation vs teacher and DFT: MAE/RMSE/R²/cosine/per-element, paired CI, parity plots, `predictions.npz` |
| `validate_materials.py --data VAL.parquet --models base=DIR name=DIR ... --out DIR` | Materials validation by group A/B/C: coverage, force MAE/median/R², energy raw/aligned, stress, paired change vs base, parity grid, `predictions.npz` |
| `apply_adapter.py RUN_DIR [RUN_DIR ...] [--weights ...] --out DIR [--info]` | Rebuild base + adapter(s) as a SavedModel (merging several adapters is approximate if their elements overlap) |
| `export_checkpoint.py CKPT_PREFIX [--out DIR]` | Turn a raw training checkpoint (`best_checkpoint.*`) into a SavedModel using the base student's `model.yaml` |
| `plot_heatmap.py`, `plot_parity.py`, `plot_pareto.py`, `plot_history.py` | Individual figures (also called by `report`) |
| `inspect_setup.py versions\|catalog\|elements` | Machine-readable facts used by `pipeline.py` (prints a `JSON:` line) |

## 4. Runtime layout (`<workdir>`, default = repository root)
```
data/organic/{train,val,test}.pkl.gz        teacher-labelled pool
data/materials/{val,test,train0}.parquet    MPtrj shards (val = held out)
data/monitors/{organic,materials,only_adapted}.pkl.gz
experiments/ledger.json
experiments/<exp_id>/{experiment.json, data/, run/}
validation/<exp_id|base>/{organic,materials}/
validation/adhoc_<timestamp>/{organic,materials,figures}/, summary.md    (validate-models)
report/report.md, report/figures/
```
