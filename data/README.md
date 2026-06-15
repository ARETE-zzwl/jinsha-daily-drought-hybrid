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
