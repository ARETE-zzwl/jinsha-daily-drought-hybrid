# Data Layout

```text
processed/
  station_daily/        Clean daily station input files used by the model
  station_metadata.csv  Coordinates and elevation for the eight stations
derived/
  paper_tables/         Compact result tables supporting manuscript claims
  figure_data/          CSV data used to render manuscript figures
external/
  cmip_station_daily_extract/  Optional external CMIP files, not included by default
```

The default training workflow uses `processed/station_daily/` and
`processed/station_metadata.csv`.

Large external climate products and full prediction tables should be archived
with DOI-based records when needed for publication review. Do not commit files
larger than normal GitHub limits to the development repository.
