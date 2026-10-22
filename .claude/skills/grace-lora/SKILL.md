---
name: grace-lora
description: Adapt (fine-tune, specialise, distil) the GRACE foundation interatomic potential to a new chemical domain with a small LoRA adapter, on an ordinary laptop CPU, then validate and report. The default domain is organic molecules with the GRACE-OFF teacher, using replay so that GRACE does not forget inorganic materials. Use this skill whenever the user wants to make GRACE (or a GRACE-2L/OMAT foundation model, an MLIP or machine-learned force field) better on molecules, organics, SPICE-like data, a teacher model's domain or their own DFT data, or mentions LoRA for GRACE, cheap or consumer-hardware fine-tuning of an interatomic potential, catastrophic forgetting in a fine-tuned potential, or running an autonomous training/validation loop for GRACE, even if they don't say "skill" or "LoRA".
---

# GRACE LoRA adaptation, with an autonomous experiment loop

This skill takes a user from "I have a laptop and a domain where GRACE is weak" to a validated,
specialised GRACE model, plus a report explaining what was tried and why. Everything runs through
`pipeline.py` and the scripts in `scripts/` in this repository. Work from the **repository root**
(the folder that contains `pipeline.py`).

What the user gets:
- **A 5 MB LoRA adapter** (rank 8, ~6% of the 21M parameters) on top of the frozen base GRACE
  model.
- **Accuracy:** on the reference task it cuts the organic force error ~3× (303 → 97 meV/Å vs DFT)
  while staying within ~11% of base GRACE on inorganic materials.
- **A ledger** of every experiment, with your reasoning for each one.
- **A report** with a heatmap, parity plots, a trade-off plot and training curves.

Reference numbers and how to read them: `references/interpreting-results.md`.

## Tested scope (tell the user this up front)
- **Tested with exactly one combination:** student `GRACE-2L-OMAT-medium-ft-AM`, teacher
  `GRACE-OFF-2L-medium` (organic molecules), on a CPU-only laptop (WSL2, 11.7 GB RAM). All
  reference numbers come from that setup.
- **Supported by the code but untested:** other GRACE teachers or students (other GRACE-OFF
  variants, other `grace:<foundation models>`, user-provided GRACE SavedModels). Say they're
  untested, **always run the smoke test first**, and present results with caution: compare
  against base, check group A is identical, and don't promise the reference numbers.
- **Not supported:** non-GRACE models (MACE, NequIP, other PyTorch or JAX models), as teacher or
  student. The adapters act on GRACE's `reducing_` tensors and the labelling uses tensorpotential
  SavedModels. Offer the alternatives: a GRACE-format teacher for the same domain, or DFT labels
  (the user's own or a public dataset) instead of a teacher.

## When to use / when not
- **Use** to improve GRACE on organic molecules (the GRACE-OFF teacher), on another GRACE
  teacher's domain (`grace:<foundation-model>` or a local GRACE SavedModel; untested), or on the
  user's own labelled structures.
- **Use** when the user wants a cheap, CPU-only specialisation that keeps base GRACE's materials
  accuracy.
- **Don't use** if they need teacher-level accuracy in one narrow domain. Using the teacher
  directly (e.g. GRACE-OFF, 15 meV/Å on organics) is better. Say so.
- **Don't use** with non-GRACE models (see Tested scope).
- **Don't use** when the target domain is far from what base GRACE learned. Check feasibility
  first (step 4); a LoRA adapter can only re-weight features the frozen base already computes.

## Prerequisites
- **Claude Code on a real machine:** Linux, WSL2 or Apple Silicon macOS. Not Claude.ai: this
  needs multi-GB installs and hour-long runs.
- **Python 3.10–3.12** available; ~8 GB free disk; ≥ 8 GB RAM (12+ GB recommended).
- **Internet access** for downloads: TensorFlow, GRACE models, GRACE-OFF, SPICE (Zenodo), MPtrj
  shards (Hugging Face).

## Ground rules (read once; they prevent the failures we already hit)
- **HARD RULE: no subagents unless switched on.** Do **not** launch subagents, parallel or
  background agents, or multi-agent workflows unless `agent_options.subagents` is true
  (`python3 pipeline.py options`). Even then, launch **at most `max_subagents` per loop**
  (default 3). Many users have limited usage, and agents multiply token cost. The whole default
  workflow (setup → loop → validation → report) runs in you, the main agent, alone. Background
  *shell* jobs such as training runs are fine; they aren't agents.
- **Ask before anything large or long:** multi-GB downloads, environment creation, any run
  expected to take more than ~30 min. Skip asking only if the user pre-approved a budget. Users
  plan around wall-clock time, so state estimates as ranges.
- **Never set a Metal (Apple GPU) device for labelling, validation or the downloaded models.**
  They are XLA-compiled and XLA has no Metal backend ("could not find registered platform"). CPU
  with `--jit` is the default and is fast.
- **Batch size follows RAM.** Under 24 GB use batch size 8 with ≤ 4 batch shapes (the defaults
  `sysinfo` recommends). Batch 16 with XLA needed > 14 GB and thrashed swap on a 12 GB laptop.
- **Keep the held-out data held out.** Never train on MPtrj **val-shard** materials or on the
  organic **test** molecules; validation and the monitors depend on them being unseen.
  `prepare` and `experiment` enforce this. Don't work around them.
- **Don't edit the scripts** to get past an error. Check `references/troubleshooting.md` first,
  then report to the user.
- **Long jobs:** run them in the background and follow the log. To stop one, kill it; don't
  pause it (a suspended job can be killed when its parent shell exits).

## Workflow

### 0. Settings review (always first)
Before any setup work, agree the settings with the user. `sysinfo` may run first, so that the
hardware-dependent defaults (batch size, device) are right.
```bash
python3 pipeline.py sysinfo          # hardware first: stop on BLOCKER lines
python3 pipeline.py settings         # show the panel to the user
```
1. **Summarise the defaults in 3–5 lines**, and what matters most:
   - **Models:** student `GRACE-2L-OMAT-medium-ft-AM`, teacher GRACE-OFF (the tested pair).
   - **Training:** replay on (1000 organic + 600 materials, 170 of them only-adapted), rank 8,
     10 epochs, CPU + XLA.
   - **Objective and budget:** materials ≤ +15% vs base; 240 min or 4 experiments (a default-size experiment takes ~55 min on a laptop CPU).
   - **Validation:** quick checks mid-loop, one full validation at the end.
   - **Agent features:** all OFF, including subagents.
2. **Invite changes,** and offer to explain any setting:
   `python3 pipeline.py settings --explain KEY`, or `references/settings.md` (effects on
   quality, time, tokens and memory, plus quick recipes such as "faster POC" or "low RAM").
3. **Apply changes** with `python3 pipeline.py settings --set KEY=VALUE [--set …]`, then show the
   panel again.
4. **Only when the user is satisfied:** `python3 pipeline.py settings --confirm`, then continue.

**If `settings_confirmed` is already true** (a later session), show a one-line summary and ask
whether to keep it.

**Never change stored settings silently during the loop.** Loop moves go per experiment with
`--set` on `pipeline.py experiment`. The stored defaults change only with the user's approval.

### 1. System check
If `sysinfo` already ran in step 0, just summarise it: OS, CPU, RAM, GPU, disk, the recommended
Python, and any **BLOCKER** lines (Intel Mac, native Windows → use WSL2, no Python 3.10–3.12, too
little disk). Stop on blockers.

Then read the **hardware notes** `sysinfo` points to (its last note lists the matching sections of
[references/hardware.md](references/hardware.md): Windows/WSL2, NVIDIA incl. RTX 50xx, macOS Apple
Silicon/Metal, Linux CPU, memory sizing). These are issues this project actually hit, with their
fixes: WSL memory and OneDrive paths, the 4 GB CUDA bundle, Metal "could not find registered
platform", fp32/fp64 `casted_model`, wrong conda/venv. Warn the user up front about any that
apply, and use the fixes when the same error shows up.

### 2. Environment
`python3 pipeline.py setup-env`: **ask first** (several GB, 5–15 min). It creates `.venv`. Use
`--venv <dir>` only if the user insists on an existing environment.

### 3. Choose the teacher
The teacher decides what the adapter learns, so settle it before downloading data. Tell the user
plainly:
- **The default is GRACE-OFF** (`GRACE-OFF-2L-medium`): organic molecules, 10 elements
  (Br C Cl F H I N O P S), and the **only teacher this workflow has been tested with**.
- **Any other GRACE teacher is supported but untested.** If they pick one, say so, run the smoke
  test before any experiment, and treat the results with caution.
- **Only GRACE teachers work** (a tensorpotential SavedModel). MACE or other PyTorch models
  can't be used: suggest DFT labels or a GRACE alternative instead.

Then offer four ways to pick a different one. Details, the resolver output and the catalog are
in `references/choosing-a-teacher.md`.
1. **An exact name:** another GRACE-OFF variant (`GRACE-OFF-{1L,2L}-{small,medium,large}`) or
   `grace:<GRACE foundation model>` (see `python3 pipeline.py catalog`).
2. **A local path:** a GRACE `saved_model` folder.
3. **A link:** a GitHub or data-repository URL, or a publication (DOI or paper URL). For a paper,
   first find its code/data-availability link (web search is fine), then resolve that link.
4. **Unsure:** ask which **elements**, and optionally what system or application. Then
   **recommend a teacher** first, and **the replay data** after it.

To resolve or check a choice, and for recommendations, use the helper:
```bash
.venv/bin/python scripts/teacher_tools.py --workdir W resolve <name | grace:<name> | path | git URL | DOI/URL>
.venv/bin/python scripts/teacher_tools.py check <saved_model_dir>
.venv/bin/python scripts/teacher_tools.py --workdir W recommend --elements C H O N [--domain "battery electrolytes"]
```
- **Output:** each prints a final `RESULT:{json}` line.
- **Several candidates** (e.g. a repository with many models): show the user each one's elements
  and precision, then pick or ask. Prefer the fp64 `saved_model` over `casted_model`.
- **A publication** returns `ok: false` until you supply its code/data link.
- **`recommend`** also returns a **replay suggestion** (`n_materials`, `n_only_adapted`). Keep
  it: it seeds the experiment loop in step 6.

Then configure, download and validate:
```bash
python3 pipeline.py catalog          # if the user wants to see student/teacher/data options
python3 pipeline.py configure --student GRACE-2L-OMAT-medium-ft-AM --teacher <name | grace:<name> | saved_model path> \
    --data <spice|pool.xyz> [--test test.xyz] [--workdir DIR] --yes
python3 pipeline.py fetch && python3 pipeline.py validate
```
- **Defaults:** data `spice`. If the project sits in OneDrive/iCloud or under `/mnt/c`, suggest
  `--workdir ~/grace_work`.
- **What `validate` checks:** files, versions and that the data's elements are covered by
  **both** teacher and student. Missing elements are a hard stop: explain, then suggest another
  teacher, filtering the data, or DFT labels.

### 4. Feasibility: is the domain reachable from base GRACE?
```bash
.venv/bin/python scripts/domain_distance.py target=<pool .xyz or .pkl.gz> --builtin crystals --n 200
```
Base GRACE's extrapolation grade γ (≤ 1 interpolation, > 2 strong extrapolation) predicts
whether an adapter can work. Verdicts by share of atoms with γ > 2:

| Share with γ > 2 | Verdict | What to tell the user |
|---|---|---|
| < 5% | close | Proceed |
| 5–20% | moderate | Proceed, but expect a larger remaining gap |
| > 20% | far | Recommend against training an adapter |

Also report per-element shares. Elements with high γ (in SPICE, P is 40% and O 17%) will be the
hardest.

### 5. Prepare data and smoke-test
1. `python3 pipeline.py prepare` builds, once:
   - the teacher-labelled organic pool (default 2000 train / 100 val / 300 test);
   - the MPtrj shards (val, test, train0);
   - three fixed monitor sets: organic, materials, only_adapted.

   **Ask first:** about 250 MB of downloads and ~10 min of labelling.
2. `python3 pipeline.py smoke`: about 10–15 min. Report the PASS/FAIL checks, peak memory and the
   timing table; that timing is your per-experiment budget model.

### 6. The autonomous loop
You are now the experimenter: run small experiments, read the ledger, decide the next one,
explain why. The user wants to watch reasoning, not just numbers.

**First, check the optional agent features:** `python3 pipeline.py options`. Do the extra work
only for features that are enabled. See "Optional agent features" below.

**Objective and budget.** Ask the user, or propose these defaults:
- **minimise organic force error** on the organic monitor;
- **keep materials within +15%** of base GRACE (`max_materials_increase 0.15`);
- **keep only-adapted crystals within +75%** (`max_only_adapted_increase 0.75`);
- **a wall-clock budget**, e.g. 3 h or ≤ 5 experiments.

The ledger stores the objective and marks each experiment `feasible` or not. Change it with
`.venv/bin/python scripts/ledger.py --workdir <workdir> set-objective key=value` if the user wants
different limits.

**Before every experiment**, estimate its duration and check it fits the remaining budget:
- **minutes ≈ (n_organic + n_materials) × epochs × 0.2 s / 60 + ~5 min overhead** (dataset
  build, compile, export);
- calibrate it with the smoke-test timing;
- reference points on a laptop CPU: 1,000 structures × 10 epochs ≈ 35 min; 1,600 × 10 ≈ 55 min.

**Run one experiment:**
```bash
python3 pipeline.py experiment --set n_organic=500 --set n_materials=500 --set n_only_adapted=60 --set epochs=6 \
    --rationale "Cheap replay probe: <why, citing numbers>" [--parent exp_00N]
python3 pipeline.py ledger
```
- **Config keys and defaults:** `n_organic 1000, n_materials 600, n_materials_val 100,
  n_only_adapted 170, rank 8, epochs 10, batch_size 8, max_buckets 4, oversample 3.0,
  max_atoms 30, seed 0, device cpu-jit`.
- **The result:** the last line of output is `RESULT:{json}` (the ledger entry). Its metrics are
  the monitor force errors at the selected (exported) epoch: `organic_f`, `materials_f`,
  `only_adapted_f`. `relative` gives the change vs base.

**Every `--rationale` is 1–3 sentences citing the numbers that motivated it.** For example: "exp_002
kept materials at +8% but only-adapted crystals at +92% (infeasible); raising n_only_adapted
60 → 170 to represent C/P/S/CHNO crystals." These rationales become the report's decision log.

**Menu of moves.** Start cheap, then scale what works.

| Situation (from the ledger) | Move |
|---|---|
| No experiments yet | A **cheap replay probe**: n_organic 500, n_materials 500, n_only_adapted ~60, epochs 6 (~20 min). If step 3's `recommend` gave replay sizes, scale those down for the probe and use them at full size once a configuration works |
| User wants to *show* forgetting | One **plain-LoRA contrast**: `n_materials=0`. Expect materials ~2× worse; skip if the budget is tight |
| `materials_increase` over the limit | More replay: raise `n_materials` (and/or `oversample`); or fewer epochs |
| `only_adapted_increase` is the worst constraint | Raise `n_only_adapted` (e.g. 60 → 170). These crystals compete with the molecules for the same element rows |
| Feasible, organic error still falling at the last epoch | More `epochs` (+50%) |
| Feasible, organics plateaued | More data: raise `n_organic` (500 → 1000 → 2000) |
| Feasible, data and epochs already raised, still a gap | Raise `rank` 8 → 16 (more capacity; ~2× adapter size) |
| Budget nearly spent, or < 3% organic gain over the best feasible for 2 experiments | **Stop** |

**Rules of thumb from this project:**
- The data mix matters more than capacity: 7.5× more parameters gave only ~3% better organics.
- Hydrogen-, oxygen- and fluorine-containing crystals are the most damaged; oversampling them
  helps.
- Change one or two things per experiment, so each rationale stays attributable.

**Read the per-epoch curves when deciding.** In `experiments/<id>/run/history.csv`, the
`*_f_mae_meV_A` columns per epoch show whether organics are still improving, and whether
materials are drifting upward late in training (a sign of too many epochs).

**Pausing and one-off checks.** The loop runs one experiment per command and the ledger keeps
everything, so the user can say "stop" at any time:
1. **Stop the running experiment by killing its process.** Never pause or suspend it
   (`SIGSTOP` / `kill -STOP`): that broke a run in this project. The ledger entry becomes
   `interrupted`.
2. **Check whatever they want on any finished experiment:**
   - `python3 pipeline.py validate-exp --exp <id> [--quick]` (`--quick` takes ~5 min instead of
     ~15);
   - or `validate-models` (below) for anything else.
3. **Show the results.**
4. **Resume later** from the ledger: `python3 pipeline.py ledger` shows where things stand. Pick
   the next move as usual, and mention the interruption in the next rationale.

### 7. Validate base and the best experiment
```bash
python3 pipeline.py validate-exp --exp base     # once: base GRACE + GRACE-OFF reference rows
python3 pipeline.py validate-exp --exp best     # the ledger's best experiment
```
- **Organic:** all 300 held-out test molecules vs DFT and vs the teacher.
- **Materials:** MPtrj val shard, groups A/B/C.
- **Runtime:** ~10–15 min each; `--quick` ~5 min. Validate other experiments only if the user
  wants them in the heatmap.

### 8. Report and explain
```bash
python3 pipeline.py report            # → <workdir>/report/report.md + figures
```
Then explain in plain words, using `references/interpreting-results.md`:
- **The headline:** organic error vs base and teacher, and the "gap closed" percentage.
- **Materials kept:** group B (mixed) and C (only adapted elements) vs base. Group A must be
  identical to base; if not, something is wrong.
- **The trade-off:** which experiment won, and why (point to the decision log).
- **Cost:** trainable parameters, adapter size, training time.
- **Limits:** e.g. "group C crystals are still ~1.4× base", "~6× behind the teacher on
  organics".
- **How to use the model:** `<workdir>/experiments/<id>/run/saved_model` loads with
  `TPCalculator(model=...)`. The adapter alone is `adapter.json` + `best_trainable_state.npz`
  (rebuild with `scripts/apply_adapter.py`).

## Optional agent features (all OFF by default)
There are four extra behaviours plus a subagent switch, **switched on only when the user asks**:
`python3 pipeline.py options --enable critic self_heal cockpit predictions` (or `--disable …`;
`options` alone shows the state, which is stored in `pipeline.json` as `agent_options`). You may
mention once that they exist; never enable one yourself. Check the state at the start of the loop.

| Feature | When enabled, you… |
|---|---|
| `critic` | **Requires `subagents` on.** If critic is on but subagents is off, tell the user and skip the critic. Otherwise, after each finished experiment, dispatch one sceptical **reviewer subagent** (template in `references/agent-options.md`; counts toward `max_subagents`). Record its verdict with `ledger.py annotate`. On `reject`, reconsider the next move and say why in the next rationale |
| `self_heal` | Nothing extra to type: `experiment` retries automatic fixes (OOM → smaller batch, Metal/XLA → CPU, NaN → lower lr; at most 2 retries) and logs them as `healing`. Explain any healing to the user. Missing data is never healed: run `prepare` |
| `predictions` | Only once the ledger has **≥ 2 finished experiments**: before each experiment, add `--hypothesis "..." --predict organic_f=… --predict materials_f=… --predict only_adapted_f=…`, grounded in previous results. Afterwards, read `ledger.py predictions` and comment on the miss |
| `cockpit` | Start `python3 pipeline.py cockpit --watch 30` in a background shell, and tell the user to open `<workdir>/report/cockpit.html` (it refreshes itself). Publish it as an artifact only if the user asks |
| `cockpit_publish` | Only if your session has an artifact-publishing tool (Claude Code's Artifact tool); otherwise tell the user once and skip. After each finished experiment, run `python3 pipeline.py cockpit --artifact` and publish `<workdir>/report/cockpit_artifact/cockpit.html` with the PNGs in its `figures/` folder as supporting files, **always to the same artifact** (same file path in this session; pass the artifact URL in a later session). Give the user the link once |


**Subagent switch:** `options --enable subagents` (master switch, default off) and
`options --set max_subagents=N` (default 3). The hard rule above applies.

Details, the reviewer prompt and the exact commands: `references/agent-options.md`.

### Token economy
- **Read summaries, not raw logs or big files:** `pipeline.py ledger`, `summary.md`,
  `RESULT:` lines.
- **Check background jobs with targeted greps** (e.g. `grep -E "^epoch|monitors:|Traceback" log`)
  and filter TensorFlow log noise. Never read a whole log.
- **Don't re-read** large JSON or `predictions.npz` files you've already summarised.
- **Keep rationales to 1–3 sentences.**
- **Mid-loop checks use `--quick` validations;** run one full validation only, for the final
  best model (plus `base` once).

## Validation only (no optimisation loop)
When the user only wants to **evaluate** models (base vs teacher, an existing run or adapter, a
colleague's checkpoint), skip the loop. Do steps 1–3 (system, environment, teacher) and
`prepare`, then:
```bash
python3 pipeline.py validate-models --model NAME=PATH [--model NAME=PATH ...] [--no-base] [--no-teacher] \
    [--quick] [--organic TEST.pkl.gz] [--materials VAL.parquet] [--out DIR]
```
- **What PATH can be:**
  - a SavedModel folder;
  - a `train_lora` run folder;
  - an adapter folder (`adapter.json` + `best_trainable_state.npz`): rebuilt automatically from
    base + adapter;
  - a raw checkpoint prefix, or a folder with `best_checkpoint.index`: exported automatically.
- **Base GRACE and the teacher** are included as reference rows unless `--no-base` /
  `--no-teacher`.
- **Data:** it needs `prepare` first (the organic test set and the MPtrj val shard), or explicit
  `--organic` / `--materials`.
- **Runtime:** `--quick` ≈ 5 min (~100 molecules, ~260 materials); a full run ≈ 15 min.
- **Output:** `<workdir>/validation/adhoc_<timestamp>/`, containing `organic/`, `materials/`,
  `summary.md` (a results table and a verdict per model) and `figures/` (heatmap, parity).
- **Explaining the result:** read `summary.md` and use `references/interpreting-results.md`.
  The same tested-scope caveats apply to models not built with this workflow.

## Reference files (read when needed)
- `references/settings.md`: every setting (default, why, effect on quality / time / tokens /
  memory, safe range, when to change it) and quick recipes. Read it when explaining or changing
  settings.
- `references/agent-options.md`: the optional critic / self-heal / predictions / cockpit
  features: how to use them, the reviewer prompt template and the ledger commands.
- `references/choosing-a-teacher.md`: the teacher options, the resolver/check/recommend outputs,
  the catalog of known teachers, and how replay data follows from the elements.
- `references/method.md`: how the LoRA adapter works, why replay prevents forgetting, the energy
  reference conversion, distillation vs DFT labels. Read to explain the method or debug odd
  results.
- `references/interpreting-results.md`: groups A/B/C, the "all materials" weighting, γ,
  heatmap/parity/trade-off reading, and the reference numbers to compare against.
- `references/hardware.md`: platform-specific issues and fixes met while building this (Windows/WSL2, NVIDIA RTX 50xx, macOS Metal, memory sizing). `sysinfo` names the sections that apply.
- `references/troubleshooting.md`: every error met so far, with its fix. Check it before
  changing anything.
- `references/commands.md`: the full CLI reference for `pipeline.py` and the scripts.
