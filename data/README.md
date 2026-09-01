# Data Layout

```text
processed/
  station_daily/        Clean daily station input files used by the model
  station_metadata.csv  Coordinates and elevation for the eight stations
derived/
  paper_tables/         Compact result tables supporting manuscript claims
  figure_data/          CSV data used to render manuscript figures
external/
  cmip_station_daily_extract/  CMIP6 auxiliary files extracted from the Zenodo archive
  upper_yellow_station_daily/  Sixteen normalized external-validation station files
  upper_yellow_station_metadata.csv
```

The default training workflow uses `processed/station_daily/` and
`processed/station_metadata.csv`.

The exact manuscript training protocol also uses station-matched CMIP6
auxiliary sequences for training-stage regularization. These files are archived
in Zenodo, not committed to GitHub:

```text
jinsha-daily-drought-hybrid-v1.0.1-wrr-submission-cmip6-station-contexts.zip
```

Download and extract that archive into the repository root before running the
main reproduction command if exact CMIP6-assisted training is required.

The Upper Yellow River inputs are distributed in
`jinsha-daily-drought-hybrid-v1.1.0-wrr-revision-upper-yellow-data.zip` on
Zenodo. Extract the archive into the repository root. The formal data include
16 stations from 2010-01-01 through 2024-03-31, matching the exact rows used in
the reported external-basin split.
