# Archival Release Status

This GitHub repository contains code, processed inputs, compact result tables,
and figure-source CSVs. The revision source archive, normalized Upper Yellow
River inputs, and larger generated artifacts are preserved in the Zenodo version
at https://doi.org/10.5281/zenodo.22232487.

## Recommended Zenodo Software Record

Archive the full GitHub release:

- `drought_hybrid/`
- `scripts/`
- `data/processed/`
- `data/derived/`
- `results/example_run/`
- documentation and metadata files

Archived source file in the Zenodo record:

- `jinsha-daily-drought-hybrid-v1.1.0-wrr-revision-source.zip`

Release tag: `v1.1.0-wrr-revision`

## Recommended Zenodo Data/Output Record

The Zenodo record also contains these large reproducibility artifacts:

- Full one-step prediction table from the manuscript run:
  `output/hybrid_modal_physics_joint/daily_multitask_joint_8stations_journal_tier1_leakfree/predictions_daily_all_models.csv`
- Full recursive prediction table from the manuscript run:
  `output/hybrid_modal_physics_joint/daily_multitask_joint_8stations_journal_tier1_leakfree/predictions_daily_recursive_all_models.csv`
- Model checkpoints:
  `best_checkpoint_tcn_daily_hybrid.pt`,
  `best_checkpoint_gru_daily_hybrid.pt`
- Station-matched CMIP6 auxiliary training contexts:
  `jinsha-daily-drought-hybrid-v1.0.1-wrr-submission-cmip6-station-contexts.zip`
- Normalized 16-station Upper Yellow River inputs:
  `jinsha-daily-drought-hybrid-v1.1.0-wrr-revision-upper-yellow-data.zip`
- Upper Yellow River checkpoints, metrics, model-selection files, logs, and
  zero-shot transfer assets:
  `jinsha-daily-drought-hybrid-v1.1.0-wrr-revision-upper-yellow-support.zip`
- Complete Upper Yellow River zero-shot and local-recalibration predictions,
  partitioned by station without sampling or rounding:
  `jinsha-daily-drought-hybrid-v1.1.0-wrr-revision-upper-yellow-predictions-part-01.zip`
  through `...part-04.zip`
- Optional full per-model prediction CSVs from `per_model_results/` and SHAP detail arrays can be added later if reviewers need exact regeneration beyond the compact CSVs already included in GitHub.

## GitHub Exclusions

These objects are excluded from GitHub because they are large, generated, or
better treated as archived research data rather than source code.

If any additional excluded file is cited in the manuscript after publication,
create a new Zenodo version and list the exact filename in
`docs/OPEN_RESEARCH_STATEMENT.md`.
