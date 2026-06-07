# Physics-Deep Learning Hybrid Framework for Daily Drought Forecasting

Public repository: https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid

This repository contains the code, processed station data, and publication table/figure data for the manuscript:

**Physics-Deep Learning Hybrid Framework for Multi-Station Daily Drought Indices and Flash Drought Forecasting in the Jinsha River Basin**

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
docs/                           Data dictionary, reproducibility notes, and WRR statement draft
```

Large full prediction tables and model checkpoints from the working project are intentionally not placed in the GitHub-ready package because some files exceed typical GitHub file-size limits. They are uploaded to the Zenodo draft deposition with reserved DOI `10.5281/zenodo.20583059`; publish the Zenodo draft to register and activate the DOI.

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

Before submission or first public release:

1. Publish this repository on GitHub or another public development platform.
2. Create a versioned release, for example `v1.0-wrr-submission`.
3. Archive that release on Zenodo and obtain a DOI.
4. Deposit large prediction tables and checkpoints in Zenodo if they are needed for complete reproduction.
5. Publish the Zenodo draft deposition for `10.5281/zenodo.20583059` after final review, so the DOI becomes registered and public.

## License

Code is provided under the MIT License. Processed data and derived manuscript data are released under CC BY 4.0 based on the author's confirmation that the station observations may be publicly redistributed; see [DATA_LICENSE.md](DATA_LICENSE.md).
