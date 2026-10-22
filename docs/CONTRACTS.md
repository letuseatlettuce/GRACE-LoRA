# CONTRACTS.md: interfaces for the parallel build (phase 1)

Four agents build in parallel, each in its own git worktree, then their branches are
merged. **Only edit the files you own.** If you need something from a file you don't own,
code against the interface defined here, and list the request in your final report. Do not
edit the other file.

## Goal of the project
A Claude Code **skill** that, on a laptop CPU, adapts the GRACE foundation interatomic
potential (base: `GRACE-2L-OMAT-medium-ft-AM`) to a new domain with a **LoRA adapter**:
- labels come from a teacher model (default GRACE-OFF, organic molecules);
- the adapter is trained with **replay** (DFT-labelled materials mixed in) so base GRACE does
  not forget materials;
- an **autonomous loop**: the agent runs small experiments, reads results, decides the next
  experiment, and logs its reasoning;
- **validation:** organic test set + materials groups A/B/C;
- **report:** heatmap, parity, Pareto, history plots and a decision log.

Proven results to reproduce (force MAE vs DFT, meV/Å). Organics = 300 held-out SPICE
molecules; materials = MPtrj val shard, groups A/B/C:
- base: 303 organic / 46 all materials;
- L2 (replay; 1000 organic + 600 materials, 170 only-adapted; rank 8, 10 epochs):
  97 organic / 51 all materials;
- smoke test (plain LoRA run without replay): 108 organic / 105 all materials.

## Repository layout and ownership
```
grace-lora/
  README.md                         D
  requirements.txt                  A
  pipeline.py                       A   stdlib-only orchestrator (entry point)
  scripts/
    config.py                       A
    lora.py                         FROZEN (nobody edits)
    train_lora.py                   A
    make_dataset.py                 A
    make_replay_dataset.py          A
    fetch_materials.py              A   (new)
    validate_model.py               A
    validate_materials.py           A
    domain_distance.py              A
    inspect_setup.py, check_models.py, apply_adapter.py, export_checkpoint.py,
    evaluate.py, download_*.sh      A
    experiment.py                   B   (new) run ONE experiment, append to the ledger
    ledger.py                       B   (new) ledger I/O, objective, show/best
    report.py                       C   (new) final report.md + figures
    plot_heatmap.py, plot_parity.py, plot_pareto.py, plot_history.py   C
  .claude/skills/grace-lora/
    SKILL.md                        D
    references/*.md                 D
  docs/CONTRACTS.md                 (integrator only)
  tests/test_core.py A | tests/test_loop.py B | tests/test_report.py C
```

## Runtime layout (under `<workdir>`, from pipeline.json "workdir"; default = repo folder)
```
<workdir>/
  data/organic/{train,val,test}.pkl.gz        teacher-labelled pool (make_dataset.py)
  data/materials/{val,test,train0}.parquet    MPtrj shards (fetch_materials.py)
  data/monitors/{organic,materials,only_adapted}.pkl.gz   fixed held-out monitor sets
  experiments/ledger.json                     the experiment ledger (ledger.py)
  experiments/<exp_id>/experiment.json        the config + rationale used
  experiments/<exp_id>/data/{train,val}.pkl.gz   mixed training data for this experiment
  experiments/<exp_id>/run/                   train_lora.py run folder (see below)
  validation/<exp_id>/organic/                validate_model.py output
  validation/<exp_id>/materials/              validate_materials.py output
  validation/base/...                         base and teacher reference rows (exp_id "base")
  report/report.md + report/figures/*.png     report.py output
```
- **Organic pool:** sized for the largest experiment. Defaults: 2000 train / 100 val /
  300 test, from Zenodo `biased_test_set.xyz` (pool) and `GRACE_test_set.xyz` (test).
- **Monitors:**
  - organic = the first 150 of `data/organic/test.pkl.gz` (teacher labels; forces are used);
  - materials = 150 MPtrj **val-shard** structures with at least one adapted element,
    ≤ 30 atoms (DFT labels);
  - only_adapted = up to 150 val-shard structures made only of adapted elements, ≤ 30 atoms.
  - Monitors are built once and never trained on.
- **Training materials** come from the test + train0 shards, **excluding every mp_id present in
  the val shard** (the val shard is reserved for validation and monitors).

## train_lora.py run folder (exists today; keep compatible)
- `args.json`, `history.csv`, `summary.json`, `energy_shifts.json`;
- `adapter.json`, `best_trainable_state.npz`, `model.yaml`, `checkpoint.*`, `saved_model/`.

`history.csv` columns:
- `epoch, train_loss, e_mae_meV_atom, f_mae_meV_A, time_s` (epoch 0 = untrained base);
- for every `--monitor name=path`: `<name>_f_mae_meV_A, <name>_e_mae_meV_atom`.

Replay flags (exist today): `--no-energy-shift --no-train-shift --adapt-elements <els...>
--monitor organic=... materials=... only_adapted=...`.

## make_replay_dataset.py (A): CLI the loop relies on
```
python scripts/make_replay_dataset.py --organic-dir <workdir>/data/organic \
   --materials <test.parquet> <train0.parquet> --exclude <val.parquet> \
   --n-organic N_ORG --n-materials N_MAT --n-materials-val N_MAT_VAL --n-only-adapted N_ONLY \
   --max-atoms 30 --oversample 3.0 --seed S --out <experiments/<id>/data> [--no-monitors]
```
- **`--n-organic N`** (A adds): use the first N molecules of `organic/train.pkl.gz`, plus all
  of `val.pkl.gz`.
- **`--n-materials 0`** must work: organic-only training, but still with energies converted to
  the base reference.
- **`--no-monitors`** (A adds): skip writing monitor files. Monitors live in
  `data/monitors/`, built by `pipeline.py prepare`.
- **Output:** `train.pkl.gz`, `val.pkl.gz`, `energy_reference.json`.

## pipeline.py subcommands (A implements; thin wrappers, stdlib only)
Existing: `sysinfo setup-env catalog configure fetch validate smoke estimate run auto wizard`.
New:
- `prepare [--n-organic-pool 2000] [--n-test 300]`: idempotent. Builds the teacher-labelled
  organic pool, downloads MPtrj shards (`fetch_materials.py`), builds the 3 monitor sets.
- `experiment --set key=value ... --rationale "TEXT" [--parent EXP_ID]`: runs
  `scripts/experiment.py` (B) with `--workdir` and the venv's Python, and streams its output.
- `ledger [--json]`: runs `scripts/ledger.py show` (B).
- `validate-exp --exp <id|best|base>`: organic (`validate_model.py` on `data/organic/test.pkl.gz`,
  all 300) + materials (`validate_materials.py` on the val shard; models base + that
  experiment; GRACE-OFF only for exp `base`) into `<workdir>/validation/<exp_id>/`.
- `report [--exps ...]`: runs `scripts/report.py` (C).

## experiment.py (B): run ONE experiment
```
python scripts/experiment.py --workdir W --config '<json>' --rationale "TEXT" [--parent ID]
       [--python <venv python>] [--dry-run]
```
Config keys, with defaults:
- `n_organic 1000`, `n_materials 600`, `n_materials_val 100`, `n_only_adapted 170`;
- `rank 8`, `alpha null` (2×rank), `epochs 10`, `batch_size 8`, `max_buckets 4`, `lr null`;
- `max_atoms 30`, `oversample 3.0`, `seed 0`;
- `device "cpu-jit"` (cpu-jit → `--jit`; gpu → `--gpu`; cpu → no flag);
- `adapt_elements null` (= the teacher's elements, from its `metadata.yaml`).

Steps:
1. Allocate `exp_id` (`exp_001`, `exp_002`, …).
2. Write `experiment.json`.
3. Build the data with `make_replay_dataset.py` (CLI above).
4. Run `train_lora.py` with the replay flags and the 3 monitors from `data/monitors/`.
5. Parse `history.csv`: the **selected epoch** = the row with the lowest `f_mae_meV_A`, i.e.
   the epoch train_lora exports. Base reference = epoch 0.
6. Append the ledger entry.
7. Print the last stdout line as `RESULT:{json}` (the ledger entry).

`--dry-run`: print the commands it would run, do nothing else (used in tests).

## Ledger format (B owns `ledger.py`; C reads it)
`<workdir>/experiments/ledger.json`:
```json
{"version": 1,
 "objective": {"primary": "organic_f", "max_materials_increase": 0.15,
               "max_only_adapted_increase": 0.75},
 "base": {"organic_f": 301.5, "materials_f": 50.0, "only_adapted_f": 62.7},
 "experiments": [
   {"id": "exp_001", "parent": null, "time": "2026-10-23 10:00",
    "rationale": "plain LoRA baseline, no replay",
    "config": {...full config incl. defaults...},
    "status": "done|failed|running",
    "run_dir": "experiments/exp_001/run", "duration_s": 1234,
    "selected_epoch": 10,
    "metrics": {"organic_f": 108.1, "materials_f": 125.0, "only_adapted_f": 298.6,
                "organic_e": 9.8, "materials_e": 15.0, "val_f": 113.1},
    "relative": {"organic_gain": 0.64, "materials_increase": 1.50,
                 "only_adapted_increase": 3.76},
    "feasible": false, "score": 108.1, "error": null}
 ]}
```
- **`relative`:** `organic_gain = 1 - organic_f/base.organic_f`;
  `*_increase = x_f/base.x_f - 1`.
- **`feasible`:** `materials_increase ≤ max_materials_increase` AND
  `only_adapted_increase ≤ max_only_adapted_increase`.
- **`score`:** `organic_f` (lower is better). **`best`** = the feasible experiment with the
  lowest score, or, if none is feasible, the one with the lowest `materials_increase`.
- **`base`:** filled from epoch 0 of the first finished experiment's monitors (identical for
  every experiment).

`ledger.py` CLI: `show [--json]`, `best`, `get <id>`, `set-objective key=value`.
Its functions are importable: `load(workdir)`, `save(workdir, ledger)`,
`append(workdir, entry)`, `best(ledger)`, `relative_metrics(...)`.

## report.py (C): final report
```
python scripts/report.py --workdir W [--exps exp_003 exp_005] [--out <workdir>/report]
```
Reads the ledger, `validation/<id>/…`, `validation/base/…` and each run's `history.csv`.

Writes `report.md` with:
1. summary: best experiment and its headline numbers vs base and teacher;
2. the experiment timeline table: config deltas, monitors, feasible;
3. the **decision log** (each experiment's rationale, in order);
4. figures:
   - heatmap (models × A/B/C/all/organic; base, GRACE-OFF, validated experiments);
   - trade-off scatter (organic_f vs materials_f per experiment, base marked,
     feasibility region shaded);
   - monitor curves per epoch (all experiments);
   - parity (best experiment, organic vs DFT and materials A/B/C);
   - Pareto (if run metadata is available);
5. limitations, auto-generated where possible (e.g. "group C still X× base").

**Plot functions must be importable** (`plot_heatmap.make(...)` etc.) as well as usable from
the CLI. Missing inputs → skip that figure with a note, never crash.

## Validation folder formats (exist today, produced by A's scripts)
- `validate_model.py`: `validation.json` → `metrics[name]["vs_dft"|"vs_teacher"]` with
  `force_mae_meV_A, force_r2, energy_mae_meV_atom, ...`; `predictions.npz`; plots.
- `validate_materials.py`: `materials_validation.json` → `results[name]["groups"]["A"|"B"|"C"]`
  with `n, force_mae_meV_A, force_mae_median_structure_meV_A, force_r2, energy_mae_raw_meV_atom,
  energy_mae_aligned_meV_atom, stress_mae_GPa`; `coverage_all`; `group_sizes_available`;
  `predictions.npz` keys `<model>__<group>__fp|dft_f`.

## Skill (D)
- **`.claude/skills/grace-lora/SKILL.md`:** YAML frontmatter `name: grace-lora` and a
  `description` with clear trigger phrases (adapt / fine-tune / specialise GRACE, LoRA,
  interatomic potential, organic molecules, cheap, laptop…).
- **Body:** the end-to-end workflow, using only the commands above:
  1. setup: `pipeline.py auto` steps;
  2. feasibility: `scripts/domain_distance.py`;
  3. `prepare`;
  4. the **autonomous loop** (objective and budget, menu of moves, stopping rules, the
     decision-log requirement);
  5. `validate-exp` for base and best;
  6. `report`.
- **The menu of moves:** baseline; add or scale replay; raise `n_only_adapted`; raise
  `n_organic`; rank 8→16; epochs ±; stop.
- **Guardrails:** RAM-aware batch size; never Metal for XLA models; time budget per experiment
  (~0.17–0.2 s per structure per epoch on a laptop CPU); never train on val-shard materials or
  test molecules.
- **`references/`:** `method.md` (LoRA + replay + why); `interpreting-results.md` (groups
  A/B/C, γ, heatmap); `troubleshooting.md` (all known errors); `commands.md` (CLI reference).
- **Keep SKILL.md concise** (< ~300 lines); push detail into references.
- **README.md:** what it is, results, install, a quick start with Claude Code, licences
  (models and data downloaded, not redistributed).

## Testing
- **Test suite:** `python -m pytest tests -q`. It needs no training and no TensorFlow inference.
- **Quick checks that load no models:** `python3 -m py_compile`, `--help`, `--dry-run`, and the
  stdlib-only `pipeline.py` steps (`sysinfo`, `catalog`).
- **Real runs** (`smoke`, `experiment`, `validate-*`) need several GB of RAM; run them one at a
  time and size `batch_size` / `max_buckets` to the machine (see the skill's hardware notes).
- **Python environment for checks:** the project's `.venv` (numpy, pandas, matplotlib,
  tensorpotential); `pipeline.py` accepts `--venv PATH` for another one.
- **Test data:** `tests/fixtures/` (see its README): small, path-free extracts of real run
  histories, validation outputs and an MPtrj sample. Never modify them in a test; copy to
  `tmp_path` first.
- **Commits:** one logical change per commit, with a message that says what changed and why.
