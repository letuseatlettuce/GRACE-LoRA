# Choosing a teacher (and the replay data that follows from it)

## Contents
1. Tested scope: what is supported and what is tested
2. The default, and the four ways to choose another teacher
3. The helper: resolve / check / recommend (output formats)
4. Known teachers (catalog)
5. From elements to replay data
6. Configure, fetch, validate
7. Common problems

## 1. Tested scope
| Combination | Status |
|---|---|
| Student `GRACE-2L-OMAT-medium-ft-AM` + teacher `GRACE-OFF-2L-medium`, CPU-only laptop (WSL2, 11.7 GB RAM) | **Tested.** All reference numbers come from this |
| Other GRACE teachers or students: other GRACE-OFF variants, `grace:<foundation model>`, user GRACE SavedModels | **Supported by the code, untested.** Tell the user, run `smoke` first, and present results with caution (compare with base; group A must be identical; don't promise the reference numbers) |
| Non-GRACE models (MACE, NequIP, other PyTorch or JAX models) as teacher or student | **Not supported.** Offer a GRACE-format teacher for the same domain, or DFT labels instead of a teacher |

Why non-GRACE models can't be used:
- **Labelling** loads the teacher with tensorpotential's `TPCalculator`, which needs a GRACE
  SavedModel.
- **The student's adapters** live on GRACE's `reducing_` tensors.

A MACE-OFF `.model` file sitting next to GRACE-OFF in the same repository is a common trap: it
won't load.

## 2. Default and options
Tell the user: **the default teacher is GRACE-OFF-2L-medium** (heid-lab/grace-off; organic
molecules; trained on SPICE v2 at ωB97M-D3(BJ); 10 elements: Br C Cl F H I N O P S). It is the
one this project was tested with. Alternatives:
1. **An exact name:**
   - `GRACE-OFF-{1L,2L}-{small,medium,large}` (same domain, different size and speed);
   - `grace:<name>` for a GRACE foundation model from `python3 pipeline.py catalog` (e.g. a larger
     materials model teaching the medium one).
2. **A local path** to a GRACE `saved_model` folder: it contains `saved_model.pb`, `variables/`
   and usually `metadata.yaml` with `chemical_symbols` and `cutoff`.
3. **A link:**
   - a git repository or data record (GitHub, Zenodo, Hugging Face, …);
   - a **publication** (DOI or article URL). Papers aren't models: find the paper's
     code/data-availability statement first (search the web, read the landing page or SI), then
     resolve that link. If there's no code/data link, tell the user and fall back to options 1, 2
     or 4.
4. **Unsure:** ask which **elements** they care about, and optionally the system or application
   (e.g. "Li-ion electrolytes", "MOFs for CO₂", "H in WO₃"). Run `recommend`, present the ranked
   teachers and explain the best, then present the replay suggestion.

## 3. The helper: `scripts/teacher_tools.py`
Each command prints a final line `RESULT:{json}`; parse that.

**Resolve a teacher spec:**
```bash
.venv/bin/python scripts/teacher_tools.py --workdir W resolve <spec>
```
- **What `<spec>` can be:** a builtin name | `grace:<name>` | a local path | a git URL | a
  DOI/publication URL.
- **Result fields:**

| Field | Meaning |
|---|---|
| `kind` | `builtin`, `grace`, `local`, `git` or `publication` |
| `model_dir` | the chosen SavedModel, if determined |
| `candidates` | `[{path, elements, cutoff, precision, is_casted}]`: every SavedModel found (git repos may hold several) |
| `elements` | the chosen candidate's elements |
| `ok` | whether a usable GRACE teacher was resolved |
| `needs_fetch` | the model must still be downloaded (`pipeline.py fetch`) |
| `notes` | human-readable remarks |

- **A publication** returns `kind: "publication"`, `ok: false` and a note asking for the
  code/data link.
- **Several candidates:** show elements and precision, and pick the one whose elements cover the
  user's data. **Prefer the fp64 `saved_model` over `casted_model`** (fp32): `casted_model`
  crashes on the fp64 inputs used for labelling. If still ambiguous, ask.

**Check a SavedModel folder:**
```bash
.venv/bin/python scripts/teacher_tools.py check <saved_model_dir>
```
Returns `{ok, elements, cutoff, precision, warnings}`. Warnings flag things like `casted_model`,
missing `metadata.yaml`, or a cutoff different from the student's 6 Å.

**Recommend a teacher:**
```bash
.venv/bin/python scripts/teacher_tools.py --workdir W recommend --elements C H O N [--domain "free text"]
```
Returns three parts:
- **A ranked teacher table:** `covers_all`, `missing` elements, `domain`, `level_of_theory`,
  `why`.
- **A replay suggestion:** `materials_with_elements`, `only_these_elements` (how many MPtrj
  materials contain them, or consist only of them), and suggested `n_materials` and
  `n_only_adapted`.
- **Notes**, for example:
  - "no catalog teacher covers all elements → provide a link or use DFT labels";
  - "elements all well covered by base GRACE → no specialist needed". Say this honestly. An
    adapter may not be worth training if base GRACE is already good there; check with
    `domain_distance.py`.

## 4. Known teachers
| Teacher | Elements | Domain / data | Level of theory | Status |
|---|---|---|---|---|
| `GRACE-OFF-2L-medium` (default) | Br C Cl F H I N O P S | Organic molecules (SPICE v2, ~2M structures) | ωB97M-D3(BJ)/def2-TZVPPD | **Tested** |
| `GRACE-OFF-{1L,2L}-{small,large}`, `GRACE-OFF-1L-medium` | same | same | same | Untested (larger = more accurate, slower labelling) |
| heid-lab `a_wpS` models (same repo) | same | 14k-structure dataset | — | Untested; weaker version of the same domain |
| `grace:<GRACE foundation model>` (e.g. `GRACE-2L-OMAT-large-ft-AM`) | 89 elements | Inorganic materials (OMat24 / MPtrj / sAlex) | PBE / PBE+U | Untested; in-domain distillation (large → medium), no new domain |
| Public specialist GRACE models for water, molecular crystals, MOFs, electrolytes | — | — | — | **None found as of 2026-10**; use DFT labels (e.g. OC22, OMC25, ODAC23 subsets) or a user link |

## 5. From elements to replay data
Replay protects base GRACE wherever the adapted elements appear in materials. The adapted
elements are, by default, the teacher's elements; replay should cover the same elements:
- **`n_materials`:** MPtrj materials that contain at least one adapted element, mixed with other
  elements (oxides, nitrides, …). These protect group B.
- **`n_only_adapted`:** materials made **only** of adapted elements (molecular crystals, elemental
  solids). These protect group C, the hardest group: the same elements as the teacher's domain
  but a different DFT method.
- **Reference sizes** (tested; the 10 GRACE-OFF elements; 1,000 molecules):
  - 600 / 170 gave all materials within 11% of base and group C 1.4× base;
  - 500 / ~10 gave group C 1.7× base.
- **More `n_only_adapted` helps group C** until it starts pulling the organics.
- **If few materials contain only the adapted elements**, use what exists and note the limitation.

Pass the suggestion into the loop:
```bash
python3 pipeline.py experiment --set n_materials=<N> --set n_only_adapted=<M> ... --rationale "..."
```
(For the first cheap probe, scale these down; see SKILL.md step 6.)

## 6. Configure, fetch, validate
```bash
python3 pipeline.py configure --teacher <name | grace:<name> | /path/to/saved_model> [--data ...] --yes
python3 pipeline.py fetch       # downloads a named teacher; checks a local path
python3 pipeline.py validate    # data elements must be covered by BOTH teacher and student
```
For a git or link-resolved teacher, use the `model_dir` from `resolve` as the `--teacher` path
(after `fetch` if `needs_fetch`).

## 7. Common problems
| Problem | What to do |
|---|---|
| The user names a MACE / PyTorch model | Not supported. Offer a GRACE-format teacher for the domain, or DFT labels |
| `casted_model` chosen | Use the sibling `saved_model` (fp64). `configure` refuses `casted_model` |
| `validate`: teacher (or student) MISSING elements | The data contains elements the model doesn't know. Pick a teacher that covers them (run `recommend` with the data's elements), filter those structures out, or use DFT labels |
| A publication with no code/data link | Tell the user it can't be used directly. Ask for the model files or a repository, or choose a catalog teacher / DFT labels |
| `recommend` says no specialist is needed | Base GRACE already covers the elements well. Measure base error first; an adapter may not pay off |
