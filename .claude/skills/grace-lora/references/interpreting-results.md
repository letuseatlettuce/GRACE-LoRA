# Interpreting results

## Contents
1. Reference numbers (what "good" looks like)
2. The organic validation
3. The materials validation: groups A, B, C and "all materials"
4. Feasibility: γ (extrapolation grade)
5. Reading each figure
6. Per-element patterns
7. Phrases for explaining results to a user

## 1. Reference numbers
Proof-of-concept results on a laptop: Intel i7-13650HX under WSL2, 11.7 GB RAM, CPU only with XLA.
Force MAE vs DFT in meV/Å:

| Model | A: no organic elements | B: organic + other elements | C: only organic elements | All materials* | Organic molecules (SPICE, 300 held out) |
|---|---|---|---|---|---|
| base (GRACE-2L-OMAT-medium-ft-AM) | 29 | 50 | 86 | **46** | 303 |
| smoke test: plain LoRA, no replay (500 molecules, 10 ep) | 29 | 125 | 302 | 105 | 108 |
| L1: replay (500 + 500, 10 ep) | 29 | 59 | 148 | 54 | 106 |
| **L2: replay** (1000 + 600 incl. 170 only-adapted, 10 ep) | **29** | **56** | **124** | **51** | **97** |
| GRACE-OFF (teacher) | | | | | **15** |

\* "All materials" weights A, B and C by their share of the MPtrj val shard: ~24% / 74% / 1.7%
by structure count.

- **L2 cost:** 1.36M trainable parameters (6.4%), a 5.4 MB adapter, ~55 min of training.
- **L2 organics:** R² vs DFT 0.974; 72% of the base → teacher gap closed; organic energy
  9.1 meV/atom (base 12.8).

## 2. Organic validation (`validation/<exp>/organic/`)
- **Force MAE/RMSE vs DFT and vs the teacher:** nearly identical, because the teacher is ~15
  meV/Å from DFT.
- **Force R²:** 1 is perfect; base ≈ 0.82, a good adapter ≈ 0.97.
- **Force direction cosine:** whether forces point the right way.
- **Energy MAE:** per-element offsets are aligned on a 20% calibration split, so models with
  different energy references compare fairly.
- **Per-element MAE** shows which elements still lag.
- **Paired test vs base:** the mean per-molecule improvement with a bootstrap 95% CI, and the
  number of molecules where the adapter wins. PASS = the whole CI is above 0.
- **Gap closed** = (base − model) / (base − teacher) on force error vs DFT.

## 3. Materials validation (`validation/<exp>/materials/`)
MPtrj val-shard DFT, at the same PBE/PBE+U level base GRACE was trained on. Groups are defined by
the **adapted elements** (by default the teacher's 10):
- **A: no adapted element** (metals, alloys, Si, GaAs, …). Must be **identical to base**: the
  largest change is 0.0. Any difference means the adapter touched non-adapted rows, which is a
  bug.
- **B: adapted + other elements** (oxides, nitrides, halides, sulfides, phosphates, hydrides).
  This is **the forgetting test**: most of inorganic chemistry is here because of oxides.
- **C: only adapted elements** (diamond, graphite, black P, S/I crystals, CHNO molecular
  crystals). The hardest group: the same elements as the molecules, but periodic and at PBE. It
  is also the only group GRACE-OFF can run on, and it does badly there.
- **Columns:** force MAE and **median** (the median is robust to blow-ups), R², energy raw and
  aligned (only the adapted elements' offsets are refitted), stress MAE (GPa).
- **Paired change vs base** with a 95% CI; a positive value means worse.

## 4. Feasibility: γ (`scripts/domain_distance.py`)
GRACE foundation models report a per-atom extrapolation grade γ: a GMM distance in the model's
latent space.
- **Thresholds:** γ ≤ 1 interpolation; 1–2 mild extrapolation; > 2 strong.
- **Verdicts** by share of atoms with γ > 2:

  | Share | Verdict |
  |---|---|
  | < 5% | close |
  | 5–20% | moderate |
  | > 20% | far: an adapter is unlikely to suffice |

- **Measured here:** inorganic crystals 0%; liquid water 0%; SPICE organics 1% overall, but
  **P 40%** and **O 17%**.
- **It matches the outcomes:** P has the largest remaining organic error. γ does not catch every
  difficulty, though: N has low γ but still a large error, because the teacher's and student's
  physics disagree.

## 5. Figures
- **Heatmap** (models × A / B / C / all materials / organics; log colour scale, green is good).
  Each model is green in its home domain. The goal is a row that is green-ish on *all materials*
  **and** on *organics*. Grey = the model cannot run (elements not supported).
- **Trade-off scatter** (organic_f vs materials_f per experiment):
  - lower-left is better;
  - the shaded region satisfies the materials constraint;
  - base sits at the bottom-right (good materials, poor organics);
  - smoke test, C10 and C11 (plain LoRA) top-left (forgetting); replay moves down toward the bottom-left.
- **Monitor curves per epoch:** organics should fall steeply and then flatten. Materials should
  stay flat near base. A late upward drift in materials means too many epochs or too little
  replay.
- **Parity plots** (DFT on x, model on y, one point per force component): tighter around the
  diagonal is better.
  - GRACE-OFF on crystals shows a vertical streak at DFT ≈ 0: huge spurious forces on
    near-equilibrium solids.
  - Panels note how many points are clipped beyond the axes.
- **Pareto** (force error vs trainable parameters / training effort / energy error): which models
  are not beaten on both axes at once.

## 6. Per-element patterns (only-adapted crystals; smoke test vs base)
| Element | smoke test vs base | After L1 |
|---|---|---|
| O | ~6× worse | ~1.9× |
| H | ~6× | ~2.2× |
| F | ~5× | ~2.5× |
| N | ~3× | — |
| S, Cl, P, C | ~2× | — |
| Br, I | ~1.5× | — |

- **The damage tracks how heavily each element was retrained:** elements frequent in SPICE (H, O,
  F) are hit most; rare ones (Br, I) least.
- **By material family:** carbon compounds without H (CN, CO) and P/S/N oxides were hit hardest;
  elemental solids least.
- **Even organic molecular crystals** (the closest to SPICE) were ~4× worse without replay:
  periodic packing at PBE is not what the adapter learned.

## 7. Explaining results to a user (plain-language templates)
- "On organic molecules the adapted model's force error dropped from X to Y meV/Å (Z× better than
  base GRACE). It closes N% of the gap to the specialist GRACE-OFF, which remains ~K× more
  accurate there."
- "On inorganic materials it stays within P% of base GRACE. Materials without any of the adapted
  elements are predicted exactly as before."
- "GRACE-OFF can't be used for materials: it supports only 10 elements (1.7% of materials) and
  produces huge errors on the crystals it can run on. The adapted model covers both."
- "The remaining weak spot is crystals made only of the adapted elements (molecular crystals,
  carbon, P/S solids): they're ~Q× base, because the same element weights must serve two DFT
  methods."
