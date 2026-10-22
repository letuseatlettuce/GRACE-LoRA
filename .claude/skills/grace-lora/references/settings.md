# Settings reference

## Contents
1. How settings work
2. Models & data
3. Training & quality
4. Objective & budget
5. Validation
6. Usage & agent features
7. Quick recipes

## 1. How settings work
```bash
python3 pipeline.py settings                    # show the panel
python3 pipeline.py settings --json             # machine-readable
python3 pipeline.py settings --explain KEY      # what one setting does
python3 pipeline.py settings --set KEY=VALUE [--set KEY=VALUE ...]
python3 pipeline.py settings --reset [KEY ...]  # back to the defaults
python3 pipeline.py settings --confirm          # the user approved: proceed
```
- **Storage:** settings live in `pipeline.json`. **The key names are the ones the panel
  shows**; if a name below differs, trust the panel and `--explain`.
- **Stored defaults vs per-experiment changes:**
  - the stored defaults are the starting configuration of every experiment;
  - the loop changes things **per experiment** with `pipeline.py experiment --set …`, which never
    rewrites the stored defaults;
  - stored defaults change only with the user's approval.

**Reference results quoted below** (tested setup: student `GRACE-2L-OMAT-medium-ft-AM`, teacher
GRACE-OFF-2L-medium, i7-13650HX, 11.7 GB RAM, WSL2, CPU + XLA). Organic / all-materials force
MAE vs DFT in meV/Å:

| Run | Organic | All materials | Training time |
|---|---|---|---|
| base | 303 | 46 | — |
| smoke test (plain LoRA, 500 molecules, no replay) | 108 | 105 | — |
| L1 (replay, 500 + 500) | 106 | 54 | ~35 min |
| L2 (replay, 1000 + 600 / 170) | **97** | **51** | ~55 min |

## 2. Models & data
| Setting | Default | What it does / why the default | Effect | Safe range / when to change | Related problem |
|---|---|---|---|---|---|
| `student` | `GRACE-2L-OMAT-medium-ft-AM` | The foundation model being adapted. **Tested** | `-large` students: slower (~2×), more memory | Any GRACE foundation model with a checkpoint (`catalog`); untested | Missing checkpoint → `fetch` |
| `teacher` | `GRACE-OFF-2L-medium` | Labels the target domain. **Tested**; organic molecules, 10 elements | Bigger teacher = slower labelling (~0.1–0.2 s per structure for 2L-medium on CPU) | Other GRACE teachers: untested. Non-GRACE: not supported. See `choosing-a-teacher.md` | `casted_model` crash; missing elements |
| `data` | `spice` (Zenodo SPICE v2 subsets) | Structures the teacher labels | — | A user `.xyz` when the domain differs | 59 overlapping structures (removed automatically) |
| `workdir` | Repository folder | Where data, experiments, validation and the report go | Synced folders slow everything | Use a local home folder if the repo is in OneDrive, iCloud or `/mnt/c` | OneDrive / WSL slowness |
| `n_organic_pool` | 2000 (train) + 100 val + 300 test | Size of the labelled organic pool `prepare` builds; experiments draw from it | Labelling ~0.2 s per molecule (~8 min for 2400) | Must be ≥ the largest `n_organic` you plan to use | — |
| materials shards | MPtrj val + test + train0 (Hugging Face) | Val = held-out validation and monitors; test + train0 = replay pool | ~240 MB download | Don't change: the val shard must stay unseen | Train/val material overlap is excluded automatically |

## 3. Training & quality
| Setting | Default | What it does / why the default | Effect (quality · time · memory) | Safe range / when to change | Related problem |
|---|---|---|---|---|---|
| `replay` (on when `n_materials` > 0) | **on** | Mixes DFT materials into training so base GRACE doesn't forget. Without it, all-materials error went 46 → 105 | Big win for materials (105 → 51); organics unchanged or better; ~1.6× training time | Turn off (`n_materials=0`) only to *demonstrate* forgetting | Forgetting on oxides etc. |
| `n_organic` | 1000 | Teacher-labelled molecules used for training. 500 → 1000 took organics 106 → 97 | More = better organics; time ∝ size | 300–2000. Raise when organics plateau | — |
| `n_materials` | 600 | Replay materials (contain adapted elements) | More = materials closer to base; slightly slower | 300–1000. Raise when `materials_increase` exceeds the limit | — |
| `n_only_adapted` | 170 (of `n_materials`) | Crystals made only of adapted elements (molecular crystals, C/P/S solids): the hardest group (C) | ~10 → 170 took group C 148 → 124 | 50–300 (limited by availability: ~490 small ones). Raise when only-adapted is the worst constraint | Group C still above base |
| `n_materials_val` | 100 | Replay materials in the validation split (selects the epoch) | — | 50–200 | — |
| `rank` | 8 | LoRA rank: adapter capacity (~1.36M parameters, 6.4%) | rank 16 ≈ 2× parameters, slightly slower. Data mattered more than capacity here (7.5× parameters gave ~3%) | 4–16. Raise after data and epochs are exhausted | — |
| `alpha` | 2 × rank | LoRA scaling | — | Leave at 2 × rank | — |
| `epochs` | 10 | Training passes; cosine learning-rate decay over all epochs | Time ∝ epochs. Organics still improving ~1 meV/Å per epoch at 10 | 6 (probe) to 20. Raise if organics are still falling; lower if materials drift up late | — |
| `lr` | 2e-3 | Peak learning rate (Adam, amsgrad) | Too high → NaN or instability | 5e-4 to 3e-3. Lower on NaN | NaN loss (self-heal lowers it) |
| `batch_size` | 8 (< 24 GB RAM) | Structures per step | 16 needs > 14 GB with XLA; 8 peaks ≈ 6.3 GB | 4–8 on ≤ 16 GB; 16 only with ≥ 32 GB | OOM / swap thrash |
| `max_buckets` | 4 | Distinct batch shapes XLA compiles | More shapes = more memory and compile time | 2–4 | OOM at startup |
| `max_atoms` | 30 | Largest replay material (atoms) | Larger = more memory per batch | 20–40 | — |
| `oversample` | 3.0 | Sampling weight for H/F-containing materials (O for only-adapted): the most damaged elements | Better protection of H, O, F | 1–5 | — |
| `device` | `cpu-jit` | CPU with XLA compilation: ~2.2× faster than plain CPU | — | `gpu` (training only; Metal/CUDA untested); `cpu` if XLA fails | XLA/Metal "registered platform" error |
| `seed` | 0 | Data sampling and initialisation | — | Change to test whether a gain is noise | — |
| `adapt_elements` | The teacher's elements | Which elements get adapter rows | Fewer elements = smaller adapter; others stay identical to base | Keep the default. Listing extra elements breaks "group A identical" | — |

## 4. Objective & budget
| Setting | Default | What it does / why | Effect | When to change |
|---|---|---|---|---|
| `max_materials_increase` | 0.15 (+15%) | Feasibility limit on the materials monitor vs base. L2 achieved ~+9% | Stricter → safer materials, possibly worse organics | 0.05 if materials matter most; 0.3 if organics matter most |
| `max_only_adapted_increase` | 0.75 (+75%) | Limit for group-C-like crystals (hardest) | Stricter → more `n_only_adapted` needed | 0.5 for molecular-crystal work; 1.0 if irrelevant |
| `time_budget_min` | 240 | Wall-clock budget for the loop (4 default-size experiments ≈ 213 min on the test laptop) | More time = more experiments | 60 (demo) to 480 (overnight) |
| `max_experiments` | 4 | Cap on experiments per loop | Each costs 20–60 min plus tokens | 2–8 |

Experiment duration ≈ (n_organic + n_materials) × epochs × 0.2 s + ~5 min. For example,
1600 × 10 ≈ 55 min, and the 500 + 500 × 6-epoch probe ≈ 20 min.

## 5. Validation
| Setting | Default | What it does | Effect | When to change |
|---|---|---|---|---|
| Mid-loop validation | `--quick` | Organic 100 molecules; materials A 50 / B 150 / C 60 | ~5 min, cheap in tokens | Default for one-off checks |
| Final validation | full, once, for the best model (plus `base` once) | Organic 300; materials A 200 / B 500 / C all ≤ 100 atoms (`validate-exp --full` or the default without `--quick`) | ~15 min | Validate other experiments only if they should appear in the heatmap |
| Monitors | Organic 150, materials 150, only-adapted ≤ 150 | Evaluated every epoch during training; drive the ledger | Differences < ~2–3 meV/Å are noise | Fixed by `prepare` |

## 6. Usage & agent features (all OFF by default)
| Setting | Default | What it does | Cost | When to turn on |
|---|---|---|---|---|
| `subagents` | **off** | Master switch: may the skill launch any agent? | Each agent multiplies tokens | Only for users with ample usage |
| `max_subagents` | 3 | Cap on agents per loop | — | 1–3 |
| `critic` | off | A sceptical reviewer agent per experiment (needs `subagents`) | ~1 agent per experiment | When rigour or a demo of agent oversight matters |
| `self_heal` | off | Auto-retry OOM / XLA / NaN failures (≤ 2 retries) | A retry costs time | Unattended or overnight runs |
| `predictions` | off | Hypothesis + predicted metrics before each experiment (after ≥ 2 done) | A few extra lines per experiment | To show calibrated, scientific reasoning |
| `cockpit` | off | Live HTML dashboard (background shell, not an agent) | Negligible | When someone watches the run |
| `cockpit_publish` | off | Republish the cockpit as a private claude.ai artifact after each experiment (Claude Code only) | One publish per experiment | When others follow the run from a browser or phone |

Details: `agent-options.md`.

## 7. Quick recipes
| Goal | Settings (per-experiment `--set` or stored defaults) | Expectation (tested hardware) |
|---|---|---|
| **Faster POC** | `n_organic=500 n_materials=500 n_only_adapted=60 epochs=6`, `max_experiments=2`, `--quick` validations | ~20 min per experiment; organics ~110–120, materials near base |
| **Best quality overnight** | `n_organic=2000 n_materials=1000 n_only_adapted=250 epochs=15` (then `rank=16`), `time_budget_min=480`, `max_experiments=6`, `self_heal` on | Several hours; expect organics < 97 and group C closer to base (untested beyond L2) |
| **Low RAM (8 GB)** | `batch_size=4 max_buckets=2 max_atoms=20`, close other apps | Slower; peak memory roughly halved |
| **Minimal token usage** | All agent features off (default), `max_experiments=2`, `--quick` mid-loop, one full final validation | Fewest agent turns; the agent reads only `ledger` and summaries |
| **Materials-safe** | `max_materials_increase=0.05`, `n_materials=1000`, `n_only_adapted=250`, `oversample=4` | Materials ≈ base; organics slightly worse than L2 |
| **Organics-max** | `n_organic=2000 epochs=15 rank=16`, `max_materials_increase=0.3` | Lowest organic error; more materials drift accepted |
