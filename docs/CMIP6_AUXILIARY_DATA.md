# CMIP6 Auxiliary Data

The manuscript uses station-matched NEX-GDDP-CMIP6 daily sequences only as
training-stage auxiliary regularization. These data are not used as predictors
during validation or independent testing.

The public GitHub repository does not commit the CMIP6 station files because the
processed files are larger than the compact observation-driven inputs. They are
archived in the Zenodo reproducibility package:

```text
jinsha-daily-drought-hybrid-v1.0.1-wrr-submission-cmip6-station-contexts.zip
```

To reproduce the manuscript protocol with CMIP6 auxiliary regularization,
download and extract that archive into the repository root so that the files are
available under:

```text
data/external/cmip_station_daily_extract/
```

Expected files:

```text
ahai_cmip_daily_bias_corrected.csv
batang_cmip_daily_bias_corrected.csv
gangtuo_cmip_daily_bias_corrected.csv
huatan_cmip_daily_bias_corrected.csv
jinjiangjie_cmip_daily_bias_corrected.csv
panzhihua_cmip_daily_bias_corrected.csv
pingshan_cmip_daily_bias_corrected.csv
shigu_cmip_daily_bias_corrected.csv
all_station_grid_match_info.csv
```

Each station CSV contains daily rows from 1970-01-01 to 2100-12-31 for
`historical`, `ssp126`, `ssp245`, and `ssp585`. The training code reads the
bias-corrected precipitation and temperature columns (`pr_bc`, `tasmax_bc`, and
`tasmin_bc`) and internally derives the lightweight context variables required
by the auxiliary regularization terms.

If these files are absent, the code still runs, but the CMIP6 auxiliary context
count is zero and the run is no longer the exact manuscript main-training
protocol.
