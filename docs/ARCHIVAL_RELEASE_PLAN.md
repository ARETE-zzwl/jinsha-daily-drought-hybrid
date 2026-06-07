# Archival Release Plan

This GitHub repository contains code, processed inputs, compact result tables,
and figure-source CSVs. Larger generated artifacts have been packaged and
uploaded to an unpublished Zenodo draft with reserved DOI
`10.5281/zenodo.20583059`.

## Recommended Zenodo Software Record

Archive the full GitHub release:

- `drought_hybrid/`
- `scripts/`
- `data/processed/`
- `data/derived/`
- `results/example_run/`
- documentation and metadata files

Suggested tag: `v1.0.0-wrr-submission`

## Recommended Zenodo Data/Output Record

The Zenodo draft contains these large reproducibility artifacts:

- Full one-step prediction table from the manuscript run:
  `output/hybrid_modal_physics_joint/daily_multitask_joint_8stations_journal_tier1_leakfree/predictions_daily_all_models.csv`
- Full recursive prediction table from the manuscript run:
  `output/hybrid_modal_physics_joint/daily_multitask_joint_8stations_journal_tier1_leakfree/predictions_daily_recursive_all_models.csv`
- Model checkpoints:
  `best_checkpoint_tcn_daily_hybrid.pt`,
  `best_checkpoint_gru_daily_hybrid.pt`
- Optional full per-model prediction CSVs from `per_model_results/`, SHAP detail arrays, and bias-corrected CMIP station daily files can be added later if reviewers need exact regeneration beyond the compact CSVs already included in GitHub.

## GitHub Exclusions

These objects are excluded from GitHub because they are large, generated, or
better treated as archived research data rather than source code.

If any additional excluded file is cited in the manuscript, add it to the
Zenodo draft and list the exact filename in `docs/OPEN_RESEARCH_STATEMENT.md`.
