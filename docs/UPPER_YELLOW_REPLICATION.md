# Upper Yellow River External Validation

## Purpose

This revision adds a climatically contrasting external-basin assessment at 16
Upper Yellow River stations. It contains two distinct protocols and does not
present the Yellow River analysis as a second full main experiment.

1. **Zero-shot transfer:** Jinsha-trained TCN and GRU checkpoints are evaluated
   without updating weights on Yellow River data. Station embeddings and static
   inputs are disabled. The target-basin drought indices use a local training-era
   climatology, but the model and classification threshold are not tuned locally.
2. **Local recalibration:** The complete framework is retrained from scratch on
   the 16 Yellow River stations using the reported hyperparameters. CMIP6
   auxiliary contexts and static DEM inputs are disabled because matched
   external-basin inputs were not available.

These protocols answer different questions. Zero-shot transfer tests portability
of learned weights, whereas local recalibration tests whether the released
workflow can be reproduced in a contrasting basin.

## Stations and Time Split

The normalized archive contains 16 stations with complete daily records from
2010-01-01 through 2024-03-31. Xiaheyan was excluded before analysis because the
meteorological and runoff coordinates differed by approximately 2.73 degrees.

| Role | Period |
|---|---|
| Training and local index climatology | 2010-01-01 to 2019-12-21 |
| Validation | 2019-12-22 to 2022-02-08 |
| Independent test | 2022-02-09 to 2024-03-31 |

## Archived Files

Extract the Zenodo data archive into the repository root. It creates:

```text
data/external/upper_yellow_station_daily/
data/external/upper_yellow_station_metadata.csv
```

Extract the external-validation support archive into the repository root. It
creates:

```text
results/archived/jinsha_no_station_embedding/
results/archived/upper_yellow_local_recalibration/  # metrics/checkpoints/logs
results/archived/upper_yellow_zero_shot/             # metrics/protocol
```

The support archive includes pooled and per-station metrics, selected features,
stacking weights, archived zero-shot standardization parameters, training logs,
and TCN/GRU checkpoints. Full one-step and recursive predictions are partitioned
by station across `upper-yellow-predictions-part-01.zip` through `part-04.zip`.
Together the four files contain every original prediction row without sampling
or numeric rounding; `SPLIT_ARCHIVE_README.md` describes reconstruction. GitHub
contains only code and compact main-study outputs, while large generated files
remain in Zenodo.

## Reported Results

For local recalibration, the prespecified one-step stacking output achieved RMSE
0.0480, MAE 0.0314, and NSE 0.9980. Its RMSE was 48.2% lower than persistence and
18.2% lower than linear least squares. At the validation-selected threshold
0.250, flash-drought test F1 was 0.640, MCC was 0.635, AUC was 0.988, and Brier
score was 0.0091.

The validation-selected target-wise recursive output achieved RMSE 0.4742 and
NSE 0.8002, but KNN was selected for both regression targets; therefore this
result must not be attributed to the hybrid branches. The best calibrated hybrid
recursive component was GRU (RMSE 0.5268; NSE 0.7336), which improved over
recursive gradient boosting but did not outperform KNN.

In the zero-shot test, the transferred TCN improved one-step RMSE over persistence
by 14.1%, whereas the transferred GRU and archived fusion did not. All transferred
hybrid branches had negative NSE over the full 782-day recursive horizon. These
mixed results support local recalibration and expose a limit of direct long-horizon
weight transfer.

## Interpretation Limits

- The formal local-recalibration run uses one random seed (42).
- The Yellow River experiment excludes matched DEM and CMIP6 auxiliary inputs.
- The study supports cross-basin reproducibility under recalibration, not universal
  zero-shot generalization.
- The flash-drought output remains a probabilistic early-risk signal requiring
  local threshold selection before operational deployment.
