# PIPELINE.md: agent playbook for the GRACE LoRA distillation pipeline

You are helping a user fine-tune a **GRACE student model** with **LoRA adapters** on labels from a **teacher model**, with `pipeline.py`.
- The default student is the materials model `GRACE-2L-OMAT-medium-ft-AM`.
- The default teacher is GRACE-OFF, a GRACE model for organic molecules.
- `pipeline.py` uses only the Python standard library. Every step is a subcommand that prints human-readable results and saves its state in `pipeline.json`.

**Your role:**
- run the steps in order;
- ask the user the questions marked **ASK** below;
- explain the results in plain words;
- never start long runs without the user's explicit go-ahead.

Humans without an agent can run `python3 pipeline.py wizard`, which does the same thing interactively.

---

## Rules

1. **Work from the project root** (the folder containing `pipeline.py`, `requirements.txt` and `scripts/`). Use `python3 pipeline.py <step>`.
2. **Never edit the scripts** or `scripts/config.py`. Choices go into `pipeline.json` through `configure`.
3. **Ask before anything long or large:** creating the environment (several GB), downloads, the smoke test (~10–15 min), any `run`.
4. **If a step fails, show the user the error.** Check the "Known problems" table below. Do not patch code.
5. **Report every step's result back to the user** in a short summary: what was found, what it means, and what's next.

---

## Step 1: System check (no questions)

```bash
python3 pipeline.py sysinfo
```

**Summarise for the user:**
- OS, CPU, cores, RAM, GPU, free disk space, available Python versions;
- any **BLOCKER** lines (Intel Mac, native Windows, no Python 3.10–3.12, less than 8 GB of disk). If there are any, stop and explain how to fix them;
- the recommendations: training device, batch size, and notes about WSL or the Mac GPU.

**Background you can pass on:**
- **Batch size follows RAM.** Under 24 GB of RAM it is 8, with at most 4 batch shapes. Batch size 16 with XLA needed more than 14 GB and caused swapping on an 11.7 GB machine.
- **Labelling and evaluation always run on the CPU.** The downloaded models are XLA-compiled and XLA has no Apple Metal backend. Training can be raced on the GPU in Step 6.

## Step 2: Python environment

**ASK:** "Create the project environment in `.venv`? It downloads several GB (TensorFlow) and takes 5–15 minutes."

```bash
python3 pipeline.py setup-env              # uses the recommended Python
python3 pipeline.py setup-env --python /path/to/python3.11   # if the user wants a specific one
```
- **Report:** the version table (all pins `OK`) and the `pip check` result.
- **Other environments:** never install into conda `base` or another project's venv. If the user insists on an existing venv, use the global flag `--venv <dir>`, but warn them that installing changes packages there.

## Step 3: Choose models and data

```bash
python3 pipeline.py catalog
```
Show the options, then **ASK** these questions (each has a default the user can just accept):

1. **Student**, the model being improved. Default: `GRACE-2L-OMAT-medium-ft-AM`.
   - Bigger students (`-large`) are slower to train.
   - With their own checkpoint, the user gives the checkpoint folder (`model.yaml` + `checkpoint.index`) **and** its SavedModel folder.
2. **Teacher**, the model whose behaviour is copied. Default: `GRACE-OFF-2L-medium`.
   - Options: the six GRACE-OFF variants, `grace:<foundation-model-name>`, or a local GRACE SavedModel folder.
   - A local path must be the fp64 `saved_model` folder, not `casted_model`.
   - MACE models are **not** supported (they are PyTorch).
3. **Data**, the structures to label. Default: `spice`, the SPICE v2 sets from Zenodo.
   - Or the user's own extended-XYZ file, plus an optional separate test file. Without one, the test set is held out from the pool.
   - Structures without DFT labels are fine; evaluation then reports agreement with the teacher only.
4. **Output location**, only if the project sits in a synced folder (OneDrive or iCloud) or on WSL `/mnt/c`: suggest a folder in the home directory.

```bash
python3 pipeline.py configure --student <...> --teacher <...> --data <spice|pool.xyz> \
    [--test test.xyz] [--student-model <SavedModel dir>] [--workdir ~/grace_runs] --yes
```
Leave out any option the user accepted as default. Show the "Saved pipeline.json" summary.

## Step 4: Download

**ASK:** "Download the selected models and data now?" Sizes: student ~325 MB, GRACE-OFF teacher ~25 MB, SPICE data ~76 MB.

```bash
python3 pipeline.py fetch
```
- **Downloads only what's missing.** Models already in `~/.cache/grace` are reused.
- **User-supplied paths are checked, not downloaded.**

## Step 5: Validate (no questions)

```bash
python3 pipeline.py validate
```
**Report:**
- file checks;
- library versions;
- element coverage: which elements are in the data, and whether the teacher and the student cover them all;
- dataset sizes;
- warnings.

**Missing elements are a hard failure.** Explain that the teacher cannot label, or the student cannot represent, structures containing those elements. Suggest another teacher or student, or filtering the data.

## Step 6: Smoke test and timing benchmark

**ASK:** "Run the smoke test (~10–15 minutes)? It checks the pipeline end to end and measures speed for the time estimates." If `sysinfo` found a GPU, also **ASK:** "Also race the GPU against the CPU for training (+2–5 minutes)?"

```bash
python3 pipeline.py smoke            # add --race if the user agreed
```

**It runs:**
1. both models on 5 molecules;
2. labelling of 160 molecules;
3. 3 LoRA epochs on 96 molecules;
4. an evaluation.

**Report:**
- the PASS/FAIL checks;
- peak training memory against the machine's RAM;
- the GPU race result, if run;
- **the time-estimate table.**

**Estimate ranges:** estimates are accurate to roughly −25% / +40%. Present them as ranges (for example "about 1–1½ hours").

## Step 7: Choose and run a plan

| Plan | Molecules (train / val / test) | Epochs | Training jobs | Purpose |
|---|---|---|---|---|
| `pilot` | 500 / 100 / 300 | 10 | LoRA rank 8 | First real result, short |
| `quick` | 1,000 / 200 / 1,000 | 15 | LoRA rank 8 + full baseline | Standard POC |
| `solid` | 2,000 / 200 / 1,000 | 30 | LoRA ranks 4, 8, 16 + full baseline | Publication-style comparison, overnight |

The "full baseline" trains the same weights without the low-rank limit, to show how much LoRA gives up. Sizes can be overridden with `--n-train`, `--n-val`, `--n-test` and `--epochs`.

**ASK:** "Which plan, and should it start now?" Use the estimates from Step 6. If the user has a deadline, pick the plan or overrides that fit, with a margin.

```bash
python3 pipeline.py run --plan <pilot|quick|solid> [--epochs N] --yes
```

**While it runs:**
- **It's resumable.** Re-running the same command skips finished steps: a labelled dataset is reused and finished training jobs are skipped. `--force` retrains.
- **On macOS** the run is wrapped in `caffeinate -i`, so the Mac doesn't sleep. The lid must stay open, or the Mac must be on power with an external display.
- **For progress,** the per-epoch lines show validation force error and epoch time. The log is `<workdir>/runs/<plan>/pipeline.log`.

## Step 8: Results

At the end, `run` writes `<workdir>/runs/<plan>/report.md`, plus `eval.json`, and `history.csv` and `summary.json` for each run.

**Explain to the user:**
- **The main result:** force error against DFT for the base student, each LoRA run and the teacher, and the **"gap to teacher closed"** percentage.
- **Parameter efficiency:** the number and percentage of trainable parameters.
- **Forgetting:** materials drift. Energy drift should be about 0, because only the trained elements changed; force drift should be small.
- **Training cost:** train time per run.
- **Using the result:** the trained model is `<workdir>/runs/<plan>/<job>/saved_model`, usable with `TPCalculator(model=...)`. Like all GRACE SavedModels, it runs on the CPU, or on an NVIDIA GPU.

---

## Known problems

| Symptom | Cause | Fix |
|---|---|---|
| `could not find registered platform` | An XLA model ran on the Apple Metal GPU | Labelling and evaluation must use the CPU (already the default); only `--race` and `device=gpu` use Metal, for training |
| Training very slow, swap full, RAM at 100% | Batch size or batch shapes too large for the RAM | `configure --batch-size 8 --max-buckets 4` (or lower), then rerun `smoke` |
| `Can not cast TensorSpec ... float64 ... float32` | Teacher is a `casted_model` | Point the teacher at `.../saved_model` |
| `validate`: teacher or student MISSING elements | Data contains elements the model doesn't know | Different teacher or student, or filter the data |
| `No Python 3.10-3.12 found` | No suitable interpreter | macOS `brew install python@3.11`; Ubuntu `sudo apt install python3.11-venv` |
| `pip check` lists pyiron, atomistics, … | A foreign environment is active | Deactivate it; use the project `.venv` |
| `git clone` / `sparse-checkout` fails | Network problem, or a half-finished clone | Move `models/grace-off` aside, then `fetch` again |
| TensorFlow log noise (NUMA, absl, oneDNN, deprecations) | Harmless | Ignore; `pipeline.py` already hides most of it |
