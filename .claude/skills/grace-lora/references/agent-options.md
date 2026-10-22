# Optional agent features (all OFF by default)

## Contents
1. Switching features on and off
2. The subagent rule (applies to everything)
3. critic: a sceptical reviewer after each experiment
4. self_heal: automatic retry of known failures
5. predictions: hypothesis + predicted metrics before each experiment
6. cockpit: a live HTML dashboard (6b. cockpit_publish: the same page as a web artifact)
7. Token economy

## 1. Switching features on and off
```bash
python3 pipeline.py options                                   # show the current state
python3 pipeline.py options --enable critic self_heal cockpit predictions subagents
python3 pipeline.py options --disable critic
python3 pipeline.py options --set max_subagents=N             # default 3
```
- **Storage:** `pipeline.json` → `agent_options` (`critic`, `self_heal`, `predictions`,
  `cockpit`, `cockpit_publish`, `subagents`, `max_subagents`).
- **Defaults:** every flag is `false`; `max_subagents` is 3.
- **Rules for you:**
  - Check the state at the start of the loop, and do extra work only for enabled features.
  - **Never enable a feature yourself.** You may mention once that they exist: "Optional extras
    are available: a reviewer agent, automatic retries, predictions, a live dashboard. Say if you
    want any."

## 2. The subagent rule
- **Off by default:** no subagents, parallel agents, background agents or multi-agent workflows
  unless `subagents` is true.
- **Even when it's on:** at most `max_subagents` per loop. Count every Agent-tool launch.
- **Why:** many users have limited usage, and each agent carries its own context and cost. The
  default workflow (setup → loop → validation → report) runs entirely in the main agent.
- **Not agents:** background **shell** jobs (training, validation, the cockpit watcher) don't
  count, and are fine.
- **critic requires `subagents`.** If critic is on and subagents is off, tell the user ("the
  critic needs subagents enabled: `options --enable subagents`") and skip it.

## 3. critic
**When:** after each **finished** experiment (`status: done`), if `critic` and `subagents` are
both on and the per-loop budget allows. Skip failed or interrupted experiments.

**What to hand the reviewer** (keep it small; don't paste whole files):
- the ledger entry (`ledger.py --workdir W get <id>`);
- the parent experiment's metrics (or base, if there's no parent);
- the per-epoch monitor curves: the `epoch`, `organic_f_mae_meV_A`, `materials_f_mae_meV_A` and
  `only_adapted_f_mae_meV_A` columns of `experiments/<id>/run/history.csv` (11 rows);
- the move you're planning next, with its rationale.

**Reviewer prompt template** (fill in the braces; send through the Agent tool):
```
You are a sceptical reviewer of one machine-learning experiment in an autonomous loop that adapts
the GRACE interatomic potential to organic molecules with a LoRA adapter plus replay of materials
data. Judge the evidence, not the narrative.

Objective: minimise organic_f (force MAE, meV/Å, 150-molecule monitor) subject to
materials_f <= {1+max_materials_increase} x base and only_adapted_f <= {1+max_only_adapted_increase} x base.
Base: organic {b_org}, materials {b_mat}, only_adapted {b_only}.

Parent ({parent_id}): {parent_metrics}
This experiment ({id}): config changes vs parent: {config_delta}
  metrics at selected epoch {epoch}: {metrics}; feasible: {feasible}
Per-epoch monitors (epoch, organic, materials, only_adapted):
{curve_rows}
Proposed next move: {next_move} because {next_rationale}

Check, briefly:
1. Noise: monitor sets have ~150 structures, so differences under ~2–3 meV/Å are noise. Is the
   claimed improvement over the parent real?
2. Overfitting/drift: does materials or only_adapted rise in late epochs while organics fall?
   Was the selected epoch sensible?
3. Attribution: does the config change plausibly explain the effect?
4. Is the proposed next move justified by these numbers, or is there a better move (e.g. more
   replay, more only-adapted data, more organics, more epochs, higher rank, stop)?

Answer in exactly this form:
VERDICT: accept | doubt | reject
REASON: 2–4 sentences citing numbers.
```

**Recording the verdict:**
```bash
.venv/bin/python scripts/ledger.py --workdir W annotate <id> review "<2–4 sentence reason>" --verdict accept|doubt|reject
```
- **`reject`:** reconsider the planned move, and explain in the next experiment's `--rationale`
  why you changed it, or why you kept it despite the review.
- **`doubt`:** mention the doubt in the next rationale. Consider repeating with a different seed
  if the difference is within noise.
- **Reporting:** the report shows review verdicts next to each experiment.

## 4. self_heal
- **When it applies:** with `self_heal` on, `pipeline.py experiment` passes `--self-heal`
  automatically. On failure, the runner classifies the error and retries at most **2** times:

  | Error class | Automatic fix |
  |---|---|
  | Out of memory / killed | Smaller `batch_size` (and fewer buckets) |
  | Metal / XLA platform error | Switch to CPU (`device cpu-jit`) |
  | NaN loss | Lower `lr` |

- **Recording:** each attempt goes in the ledger entry as `healing` (the error class, what was
  changed, the outcome).
- **Your job:** tell the user what failed and what was changed. Note that the healed run's
  config differs from the requested one, and say so in the next rationale.
- **Never healed:** missing data or files. Run `python3 pipeline.py prepare` (or `fetch`) instead
  of retrying. Errors outside the classes above stop the experiment; consult
  `troubleshooting.md`.

## 5. predictions
- **Only once the ledger has ≥ 2 finished experiments.** Before that there's nothing to ground a
  prediction in, and it wastes tokens.
- **Before each experiment**, add a one-sentence hypothesis and numeric predictions, extrapolated
  from previous results (trends with data size, epochs, replay):
  ```bash
  python3 pipeline.py experiment --set ... --rationale "..." \
      --hypothesis "Raising n_only_adapted 60→170 cuts only_adapted error by ~20% at ≤3 meV/Å organic cost" \
      --predict organic_f=99 --predict materials_f=54 --predict only_adapted_f=105
  ```
- **After it finishes:** `.venv/bin/python scripts/ledger.py --workdir W predictions` shows
  predicted vs actual. Comment on the miss in one sentence ("only_adapted came in 8% better than
  predicted: the extra crystals helped more than linearly") and use it to calibrate the next
  prediction.
- **The report** gets a predicted-vs-actual section.

## 6. cockpit
- **Start it** in a background shell (it's not an agent):
  ```bash
  python3 pipeline.py cockpit --watch 30
  ```
- **What it does:** every 30 s it writes `<workdir>/report/cockpit.html`, a self-contained page
  that refreshes itself. It shows the ledger, monitor curves and status.
- **Tell the user** to open that file in a browser.
- **Publishing:** if the Claude Code session has an artifact-publishing tool, publish the page
  **only if the user asks.**
- **Stopping:** stop the watcher (kill it) when the loop ends.

### 6b. cockpit_publish (Claude Code only)
A private claude.ai page that others can open from any browser or phone. It is a **snapshot that
the agent republishes**, not a live connection to this machine.
- **Requires:** an artifact-publishing tool in the session (Claude Code's `Artifact` tool). Other
  agents: tell the user once that this option is unavailable, and use `cockpit` instead.
- **After each finished experiment** (and once when the loop ends):
  ```bash
  python3 pipeline.py cockpit --artifact
  # -> <workdir>/report/cockpit_artifact/cockpit.html + figures/*.png
  ```
  Publish `cockpit.html` with `files` = `{"figures/<name>.png": "<path>", ...}` for every PNG the
  command lists. If the tool only publishes from the working directory or a scratchpad, copy
  the whole `cockpit_artifact/` folder there first, always to the same place. **Reuse the same artifact:** the same `file_path` within one session; in a later
  session, pass the artifact's URL (read it first). Never create a new link per experiment.
- **The first publish:** give the user the link once. Later republishes reach open viewers
  automatically; mention them only in your normal per-experiment summary.
- **Privacy:** artifacts are private until the user shares them, but publishing does send the
  results to claude.ai. The page contains no local paths.
- **Cost:** one command and one publish per experiment (a few tokens), no agents.

## 7. Token economy
- **Read summaries, not raw outputs:** `python3 pipeline.py ledger`, `summary.md`, `RESULT:`
  lines, `ledger.py get <id>`.
- **Background jobs:** check them with targeted greps (`grep -E "^epoch|monitors:|Traceback"`),
  filtering TensorFlow noise (cuda, absl, oneDNN, XLA, deprecation lines). Never read a whole log.
- **Large files:** don't re-read large JSON or `predictions.npz` files once summarised.
- **Rationales:** 1–3 sentences.
- **Validations:** use `--quick` mid-loop. Run one full validation for the final best model, plus
  `base` once.
- **Subagents:** none unless enabled; at most `max_subagents` per loop.
