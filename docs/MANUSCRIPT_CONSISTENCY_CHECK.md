# Manuscript Consistency Check

Date checked: 2026-06-08

Source manuscript: `Manuscript(1).docx`

## Overall Assessment

The open package is consistent with the manuscript's main daily-scale experiment after the corrections listed below. It contains the daily physics-deep learning workflow, eight processed Jinsha River Basin station files, manuscript-derived table/figure CSVs, and compact main-run metrics.

## Items Checked

| Manuscript item | Open package status |
| --- | --- |
| Eight stations: Huatan, Pingshan, Gangtuo, Batang, Panzhihua, Shigu, Jinjiangjie, Ahai | Present in `data/processed/station_daily/` with ASCII slugs and English labels |
| Daily station meteorological variables: precipitation, temperature, ET0, wind, net radiation, downward radiation | Present as `pr`, `tmean`, `tmax`, `tmin`, `et0`, `wind`, `rad_net`, `rad_down` |
| Static attributes: elevation, latitude, longitude | Present in `data/processed/station_metadata.csv` |
| Targets: custom `idx_30`, `idx_90`, and `flash_label` | Constructed at runtime in `drought_hybrid/daily_data.py` |
| Leakage control: train-period target scaling and modal-feature scoring | Implemented in `drought_hybrid/daily_data.py` and `drought_hybrid/daily_trainer.py` |
| 30-day sliding input window | Reproduction command and architecture spec use `--seq-len 30` |
| 24 dynamic features: 8 raw + 16 selected modal components | `results/example_run/selected_feature_list.csv` contains 24 features |
| TCN-Transformer and GRU-Transformer branches | Implemented in `drought_hybrid/daily_models.py` |
| Differentiable water-balance physics branch and adaptive gate | Implemented in `drought_hybrid/daily_models.py` |
| Main one-step RMSE 0.0381 | `results/example_run/metrics_daily_model_comparison.csv` best test `reg_macro` is 0.038105 for `fusion_daily_stacking` |
| Main recursive RMSE 0.278 | `results/example_run/metrics_daily_model_comparison_recursive.csv` best test recursive `reg_macro` is 0.278227 for `fusion_targetwise_blend_recursive` |
| Flash drought output treated as probabilistic risk signal | Reflected in README, data dictionary, and open-research statement |

## Corrections Made During This Check

- Updated the main reproduction command to include `--model-dim 96`, matching the manuscript architecture metadata.
- Replaced the public `drought_hybrid/data.py` with a clean station-slug/English-column implementation and removed legacy raw-column handling.
- Cleaned `data/derived/figure_data/model_architecture_spec.json` so it no longer contains encoding artifacts.
- Updated data licensing text to state that processed station observations may be publicly redistributed and are released under CC BY 4.0.
- Recorded the suggested public repository name as `jinsha-daily-drought-hybrid`.

## Naming Assessment

The local folder name `wrr_submission_open_project` is useful as a working directory but is not ideal as a public repository name. The recommended public repository name is:

```text
jinsha-daily-drought-hybrid
```

This name is concise, avoids journal-specific wording, and captures the study area, temporal scale, target domain, and method family.

## Remaining Publication Metadata

The public GitHub URL and Zenodo DOI have been inserted. The manuscript DOI in `.zenodo.json` remains pending until journal acceptance. The Zenodo record includes the tagged source archive, station-matched CMIP6 auxiliary contexts, full prediction tables, and model checkpoints at https://doi.org/10.5281/zenodo.20705450.
