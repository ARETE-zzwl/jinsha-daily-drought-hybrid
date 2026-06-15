# Data Dictionary

## Station Metadata

File: `data/processed/station_metadata.csv`

| Column | Description | Unit |
| --- | --- | --- |
| `station_slug` | ASCII station identifier used by the code | unitless |
| `station_name` | Original station name | unitless |
| `station_en` | English station label | unitless |
| `longitude` | Station longitude | decimal degrees |
| `latitude` | Station latitude | decimal degrees |
| `elevation_m` | Station elevation | m |

## Daily Station Files

Files: `data/processed/station_daily/*_daily.csv`

| Column | Description | Unit |
| --- | --- | --- |
| `date` | Observation date in UTC | YYYY-MM-DD |
| `station_slug` | ASCII station identifier | unitless |
| `station_name` | Original station name | unitless |
| `station_en` | English station label | unitless |
| `pr` | Daily precipitation | mm day-1 |
| `tmean` | Mean 2 m air temperature | deg C |
| `tmax` | Maximum 2 m air temperature | deg C |
| `tmin` | Minimum 2 m air temperature | deg C |
| `et0` | Reference evapotranspiration | mm day-1 |
| `wind` | Mean wind speed | m s-1 |
| `rad_net` | Net radiation | J m-2 |
| `rad_down` | Downward shortwave radiation | J m-2 |
| `runoff` | Streamflow/runoff discharge | m3 s-1 |
| `longitude` | Station longitude | decimal degrees |
| `latitude` | Station latitude | decimal degrees |
| `year` | Calendar year | year |
| `surface_pressure_hpa` | Surface pressure | hPa |
| `dewpoint` | Dew point temperature | deg C |
| `wind_v` | Meridional wind component | m s-1 |
| `wind_u` | Zonal wind component | m s-1 |

## Model-Derived Targets

The training pipeline derives targets from the daily station files at runtime.

| Name | Description |
| --- | --- |
| `idx_30` | Custom standardized 30-day rolling water-balance anomaly. Lower values indicate drier-than-normal conditions. |
| `idx_90` | Custom standardized 90-day rolling water-balance anomaly. Lower values indicate longer-term moisture deficit. |
| `flash_label` | Operational flash-drought early-risk label derived from rapid `idx_30` decline and persistent dry conditions. |

For each station, the daily water balance is:

```text
WB_t = pr_t - et0_t
```

For window `W` in `{30, 90}`, the rolling accumulated water balance is standardized using training-period statistics only:

```text
idx_W,t = (sum(WB over the previous W days) - mean_train_W) / std_train_W
```

Flash-drought labels use the manuscript thresholds:

```text
rapid_t = idx_30,t - idx_30,t-14 <= -0.8
persist_t = max(idx_30,t-4 ... idx_30,t) <= -1.0
flash_label_t = rapid_t and persist_t
```

## CMIP6 Auxiliary Station Contexts

Files are archived in Zenodo under:

```text
jinsha-daily-drought-hybrid-v1.0.1-wrr-submission-cmip6-station-contexts.zip
```

After extraction, the expected repository path is:

```text
data/external/cmip_station_daily_extract/*_cmip_daily_bias_corrected.csv
```

These files are derived from NEX-GDDP-CMIP6 daily downscaled climate-model
series and are used only for training-stage auxiliary regularization. They are
not used as validation/test predictors.

| Column | Description | Unit |
| --- | --- | --- |
| `station_slug` | ASCII station identifier used by the code | unitless |
| `station` | Original station name | unitless |
| `date` | CMIP6 daily timestamp | YYYY-MM-DD |
| `scenario` | `historical`, `ssp126`, `ssp245`, or `ssp585` | unitless |
| `pr` | Station-extracted daily precipitation before local bias correction | mm day-1 |
| `tasmax` | Daily maximum near-surface air temperature before local bias correction | deg C |
| `tasmin` | Daily minimum near-surface air temperature before local bias correction | deg C |
| `tas` | Daily mean near-surface air temperature before local bias correction | deg C |
| `pr_bc` | Locally bias-corrected daily precipitation used by the training code | mm day-1 |
| `tasmax_bc` | Locally bias-corrected daily maximum temperature used by the training code | deg C |
| `tasmin_bc` | Locally bias-corrected daily minimum temperature used by the training code | deg C |
| `tas_bc` | Locally bias-corrected daily mean temperature | deg C |

## Derived Manuscript Data

`data/derived/paper_tables/` contains compact CSVs supporting manuscript result tables, pooled metrics, station-level metrics, ablation summaries, gate weights, and observed-vs-predicted slices.

`data/derived/figure_data/` contains compact CSVs used for figure rendering, including study-area summaries, modal feature scores, one-step and recursive metric summaries, flash-drought classification curves, and SHAP/feature-importance summaries.

Files ending in `_en.csv` are English companion versions of selected summary
tables/manifests for international review. They do not replace the original
manuscript-facing CSVs.
