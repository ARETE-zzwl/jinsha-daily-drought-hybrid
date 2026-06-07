# Archival Release Plan

This GitHub-ready repository contains code, processed inputs, compact result
tables, and figure-source CSVs. The following larger or optional objects from
the working project should be considered for Zenodo or another DOI-based archive
instead of GitHub.

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

Add large reproducibility artifacts that are useful but not GitHub-friendly:

- Full one-step prediction table from the manuscript run:
  `output/hybrid_modal_physics_joint/daily_multitask_joint_8stations_journal_tier1_leakfree/predictions_daily_all_models.csv`
- Full recursive prediction table from the manuscript run:
  `output/hybrid_modal_physics_joint/daily_multitask_joint_8stations_journal_tier1_leakfree/predictions_daily_recursive_all_models.csv`
- Optional model checkpoints:
  `best_checkpoint_tcn_daily_hybrid.pt`,
  `best_checkpoint_gru_daily_hybrid.pt`
- Optional full per-model prediction CSVs from `per_model_results/`
- Optional full SHAP detail arrays if reviewers need exact interpretability regeneration
- Optional bias-corrected CMIP station daily files if future-scenario analyses are included in the final manuscript claims

## GitHub Exclusions

These objects are excluded from GitHub because they are large, generated, or
better treated as archived research data rather than source code.

If any excluded file is cited in the manuscript, add the Zenodo DOI and exact
filename to `docs/OPEN_RESEARCH_STATEMENT.md`.
