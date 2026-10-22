# Method: LoRA adapters for GRACE, with replay

## Contents
1. The setting: student, teacher, data
2. What is frozen, what is adapted
3. The LoRA update and how it is trained
4. Why plain adaptation forgets, and why replay fixes it
5. Energy references
6. Distillation vs DFT labels
7. What an adapter is on disk

## 1. The setting
- **Student:** a GRACE foundation model, by default `GRACE-2L-OMAT-medium-ft-AM`: 21.1M parameters,
  89 elements, trained on PBE/PBE+U VASP data (OMat24, then MPtrj and sAlex). It is good on
  inorganic materials and poor on organic molecules (~303 meV/Å force error vs DFT on SPICE).
- **Teacher:** a model that is good in the target domain. The default is **GRACE-OFF 2L-medium**
  (heid-lab/grace-off; trained from scratch on ~2M SPICE structures at ωB97M-D3(BJ); 10
  elements: Br C Cl F H I N O P S). It is a specialist for molecules made of these elements.
- **Labels:** the teacher's energies and forces on sampled structures (distillation), plus DFT
  labels for the replay materials (MPtrj, PBE/PBE+U).

## 2. What is frozen, what is adapted
GRACE computes atomic features from:
- element embeddings;
- radial functions (6 Å cutoff);
- angular basis functions;
- products of those.

It then combines them with **`reducing_` coefficient tensors** of shape
(89 elements, out channels, in channels, coupling paths), in three layers: I1, I2 and rho. That
makes 12 tensors.

- **Frozen:** everything, including the feature construction. The adapter therefore cannot invent
  new descriptors; it can only re-weight what base GRACE already computes. That is why the γ
  feasibility check matters.
- **Adapted:** for each **adapted element** (by default the teacher's elements), the slice
  W[z] of every `reducing_` tensor. Other elements' slices stay exactly as in base. So a structure
  containing none of the adapted elements is predicted **identically** to base GRACE (group A in
  the validation). This is guaranteed by construction, not a learned outcome.
- **Symmetry:** only weight values change, never the architecture, so the model stays rotationally
  equivariant.

## 3. The LoRA update
For each adapted slice: **W = W₀ + (α/r)·B·A**.
- **Factors:** B has shape (n_out × r) and starts at zero; A has shape
  (r × n_in·n_paths) and starts random. So W = W₀ at step 0.
- **Defaults:** rank r = 8, α = 2r. That gives ~1.36M trainable numbers (~6.4%) for the 10
  GRACE-OFF elements.
- **Training is custom code** (`scripts/lora.py`, `scripts/train_lora.py`). tensorpotential's own
  LoRA hooks raise `NotImplementedError`. Each step:
  1. tensorpotential computes dL/dW for the merged weight;
  2. the chain rule maps it to dL/dA and dL/dB;
  3. Adam updates A and B;
  4. W is rewritten as W₀ + (α/r)·B·A.

  This is mathematically exact LoRA, with no changes inside GRACE.
- **Loss:** Huber on energies (weight 1) and forces (weight 10); cosine learning-rate decay;
  amsgrad.
- **Selected epoch:** the one with the lowest validation force error is exported.

## 4. Why plain adaptation forgets, and why replay fixes it
- **The problem:** there is one set of weights per element. Teaching O, H, C, N… how they behave
  in molecules, at the teacher's DFT level, also changes how they behave in oxides, nitrides,
  hydrides and molecular crystals. The plain adapter (organics only) got organics to 108 meV/Å
  but made "all materials" 105 meV/Å vs base 46. It was worst on H, O and F (~6× in
  only-adapted crystals).
- **Training longer does not fix it.** Nothing in the loss asks the model to stay good on
  materials, so more epochs move the weights further from base.
- **Replay** mixes DFT-labelled materials containing the adapted elements into every training
  run. The adapter must then keep base-like behaviour on those environments while learning
  organics:
  - L1 (500 + 500): organics 106, all materials 54;
  - L2 (1000 molecules + 600 materials, 170 of them made only of adapted elements): organics
    **97**, all materials **51** (base 46).
- **The remaining tension:** crystals made only of adapted elements (group C: molecular crystals,
  C/P/S solids) share elements with the molecules, but are labelled with a different DFT method
  (PBE vs ωB97M). One adapter must compromise there. Raising `n_only_adapted` helps; it does not
  vanish.

## 5. Energy references
Teacher and student use different per-element energy references. SPICE/GRACE-OFF energies are
relative to isolated atoms at ωB97M; MPtrj/base GRACE are raw PBE VASP totals.
- **Replay:** the organic labels are moved onto the **base reference** before mixing:
  E′ = E − Σ_el n_el·δ_el. Here δ is fitted by least squares on teacher − base predictions over
  the training molecules.
  - This is exact for energies; forces are untouched.
  - Training then uses `--no-energy-shift --no-train-shift`, so non-adapted elements keep base
    energies exactly.
- **Teacher-only runs** (no replay) instead fit and inject a per-element shift (gracemaker's
  `shift: auto`), and train it. That shifts the energies of O, H and so on in materials by
  constants. This is harmless for forces, but raw materials energies then look wrong; compare
  them per-element aligned.

## 6. Distillation vs DFT labels
- **Distillation (a teacher model):** labels are cheap (~0.1–0.2 s per structure) for *any*
  geometry: MD frames, rattled structures, new molecules. Accuracy is capped at the teacher's.
- **DFT labels:** no teacher ceiling and no teacher needed, but limited to existing datasets.
  New DFT costs minutes to hours per structure.
- **Labels are rarely the bottleneck here.** The student sits ~100 meV/Å from DFT, while
  GRACE-OFF is 15 meV/Å from DFT; the student's capacity and data mix limit it.
- **What matters more is matching the DFT method of what you want to keep.** Replay materials are
  PBE, the same as base GRACE.

## 7. On disk
- **The adapter, in each run folder:**
  - `best_trainable_state.npz`: the A/B factors (+ energy shift), ~5 MB;
  - `adapter.json`: rank, α, targets, element rows, energy shifts.
- **The full model:** `saved_model/` (~84 MB) is the merged model.
- **Rebuilding:** `scripts/apply_adapter.py <run_dir> --out <dir>` rebuilds base + adapter
  (bit-identical predictions). Several adapters can be merged with `--weights`. That is valid
  for adapters on disjoint elements and approximate otherwise.
