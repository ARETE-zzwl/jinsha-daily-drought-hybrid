# Physics-Deep Learning Hybrid Framework for Daily Drought Forecasting

Public repository: https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid

This repository contains the code, processed station data, and publication table/figure data for the manuscript:

**Physics-Deep Learning Hybrid Framework for Multi-Station Daily Drought Indices and Flash Drought Forecasting in the Jinsha River Basin**

Release `v1.1.0-wrr-revision` adds an external-basin assessment at 16 Upper
Yellow River stations. The repository name is retained because the Jinsha River
Basin remains the primary experiment; the Yellow River analysis is a targeted
cross-basin robustness test requested during revision.

The package is organized for submission to *Water Resources Research* (WRR). It follows the AGU expectation that analysis code be openly developed on a platform such as GitHub and preserved in an archival repository such as Zenodo with a DOI.

## Repository Contents

```text
drought_hybrid/                 Core Python package for data processing, models, and training
scripts/                        Data preparation and package-check helpers
data/processed/station_daily/   Eight processed daily station input files
data/processed/station_metadata.csv
data/derived/paper_tables/      Curated CSVs supporting manuscript tables and result summaries
data/derived/figure_data/       Curated CSVs used to recreate manuscript figures
results/example_run/            Main-run metrics and model-selection outputs from the manuscript run
scripts/*upper_yellow*          External-basin checks, training, evaluation, and summaries
docs/                           Data dictionary, reproducibility notes, and WRR open-research statement
```

Large full prediction tables, model checkpoints, the normalized Upper Yellow
River inputs, and station-matched CMIP6 auxiliary training sequences are not
committed to GitHub. The versioned Zenodo record at
https://doi.org/10.5281/zenodo.22232487 contains those files plus a source archive
for the tagged GitHub release.

## Main Data

The processed station files contain daily meteorological, radiation, ET0, runoff, and site metadata for eight representative stations from 2010 to 2024. The model constructs two custom standardized water-balance drought indices:

- `idx_30`: 30-day standardized rolling water-balance anomaly
- `idx_90`: 90-day standardized rolling water-balance anomaly

Flash drought labels are derived from rapid decline and persistent dryness in `idx_30`; they are used as an operational early-risk signal, not as a universal flash-drought definition.

See [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for variable definitions.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`shap`, `PyEMD`, and `vmdpy` are optional. The default manuscript configuration uses moving-average decomposition and can run without `PyEMD` or `vmdpy`.

## Quick Check

Run a lightweight import and data check:

```bash
python scripts/check_package.py
```

Expected result: the script prints the eight station slugs, row counts, static metadata dimensions, and selected target columns produced from the processed data.

## Reproduce the Main Experiment

For the exact manuscript protocol, first download
`jinsha-daily-drought-hybrid-v1.0.1-wrr-submission-cmip6-station-contexts.zip`
from Zenodo and extract it into the repository root. The expected path is:

```text
data/external/cmip_station_daily_extract/
```

If these CMIP6 files are absent, the code still runs, but the CMIP6 auxiliary
regularization context is skipped and the run is not the exact manuscript
training protocol. See [docs/CMIP6_AUXILIARY_DATA.md](docs/CMIP6_AUXILIARY_DATA.md).

The main training entry point is:

```powershell
python run_daily_drought_model.py `
  --stations all `
  --seq-len 30 `
  --epochs 30 `
  --batch-size 128 `
  --modal-method moving `
  --top-k-per-base 2 `
  --model-dim 96 `
  --run-tag wrr_reproduce
```

For a quick smoke run, reduce epochs and skip heavier baselines:

```bash
python run_daily_drought_model.py --stations all --epochs 1 --skip-baselines --run-tag smoke
```

Outputs are written to `results/runs/`.

## Upper Yellow River External Validation

Download the data and support archives from Zenodo and extract both into the
repository root:

```text
jinsha-daily-drought-hybrid-v1.1.0-wrr-revision-upper-yellow-data.zip
jinsha-daily-drought-hybrid-v1.1.0-wrr-revision-upper-yellow-support.zip
```

The full prediction tables are partitioned by station across four additional
Zenodo files named `...upper-yellow-predictions-part-01.zip` through
`...part-04.zip`. The parts jointly contain every original row without sampling
or numeric rounding. The support archive includes `SPLIT_ARCHIVE_README.md` with
the exact reconstruction instructions.

Check the external data before running any model:

```bash
python scripts/check_upper_yellow_package.py
```

The local-recalibration protocol retrains the framework from scratch in the
Upper Yellow River Basin. It uses the same hyperparameters as the reported run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_upper_yellow_replication.ps1
```

The zero-shot protocol transfers Jinsha-trained TCN and GRU weights without any
Upper Yellow River weight update, station embedding, static input, or local
threshold tuning:

```bash
python scripts/evaluate_jinsha_to_upper_yellow_zero_shot.py
```

See [docs/UPPER_YELLOW_REPLICATION.md](docs/UPPER_YELLOW_REPLICATION.md) for
the split dates, archived file map, reported metrics, and interpretation limits.

## Manuscript Results Already Included

`results/example_run/` stores compact outputs from the manuscript's main run, including:

- `metrics_daily_model_comparison.csv`
- `metrics_daily_model_comparison_recursive.csv`
- `selected_feature_list.csv`
- `split_time_ranges_daily.csv`
- `stacking_weights_daily.csv`
- `training_log_daily.csv`

`data/derived/` stores curated table and figure data used in the manuscript. These files are small enough for a GitHub repository and are suitable for reviewer inspection.

## Archival Release Plan

For submission and archival review:

1. Public GitHub repository: https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid
2. Versioned release: https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid/releases/tag/v1.1.0-wrr-revision
3. Zenodo archival record: source archive, processed cross-basin data, full prediction tables, model checkpoints, and CMIP6 auxiliary station contexts at https://doi.org/10.5281/zenodo.22232487.

## License

Code is provided under the MIT License. Processed data and derived manuscript data are released under CC BY 4.0 based on the author's confirmation that the station observations may be publicly redistributed; see [DATA_LICENSE.md](DATA_LICENSE.md).
