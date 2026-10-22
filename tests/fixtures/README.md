# Test fixtures

Small, path-free extracts of the reference runs, so every test runs on any machine:

- `runs/exp_001..003/`: `history.csv`, `summary.json`, `args.json` (and `adapter.json`) of the
  smoke test, L1 and L2 runs. Local paths are replaced by `<local>/...`.
- `validation/{base,exp_002,exp_003}/{organic,materials}/`: `validation.json` /
  `materials_validation.json` (full metrics) and `predictions.npz`, cut down to the first 40
  organic structures and 400 force rows per material group (float32).
- `mptrj_val_sample.parquet`: first 2000 rows (`mp_id`, `numbers`, `num_atoms`) of the MPtrj
  validation shard.

Sources and licences: SPICE v2 subsets from Zenodo record 18662355 (CC BY 4.0) and MPtrj
(Materials Project, CC BY 4.0, via Hugging Face `nimashoghi/mptrj`). No pickles are stored here.
