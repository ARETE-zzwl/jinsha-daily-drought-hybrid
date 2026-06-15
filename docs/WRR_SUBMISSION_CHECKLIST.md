# WRR Open Science Submission Checklist

Use this checklist before submitting the manuscript.

## Repository

- [x] Public GitHub repository created from this directory: https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid
- [x] Release tag created: https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid/releases/tag/v1.0.0-wrr-submission
- [x] Repository README includes installation, quick check, and reproduction commands.
- [x] `LICENSE`, `DATA_LICENSE.md`, `CITATION.cff`, and `.zenodo.json` are complete.
- [x] All GitHub and Zenodo DOI placeholders are replaced; manuscript DOI remains pending post-acceptance metadata.
- [x] No local absolute paths are required by default runtime code.
- [x] No private files, credentials, editor folders, or oversized prediction tables are committed.

## Data

- [x] Processed station data are present in `data/processed/station_daily/`.
- [x] Station metadata are present in `data/processed/station_metadata.csv`.
- [x] Derived table/figure data are present in `data/derived/`.
- [x] Data dictionary explains all variables and units.
- [x] Redistribution permission has been confirmed for the processed observational station data.
- [x] Processed data and derived manuscript data are marked as CC BY 4.0.

## Software Preservation

- [x] GitHub repository URL is recorded in Zenodo metadata.
- [x] Zenodo DOI is registered: `10.5281/zenodo.20583059`.
- [x] The Zenodo record includes the same authors as the manuscript.
- [x] The Zenodo record includes license, keywords, and description.
- [x] Source archive for the tagged GitHub release is uploaded to Zenodo.
- [x] Large prediction tables/checkpoints are uploaded to Zenodo, not GitHub.
- [x] Zenodo record is published and publicly resolvable.

## Manuscript

- [ ] Open Research section contains a Data Availability Statement.
- [ ] Open Research section contains a Software Availability Statement.
- [ ] Statements include direct DOI links, license/access conditions, and the GitHub development link.
- [ ] Data/software records are cited in the References section, not only linked in the text.
- [ ] The Methods section describes how released data/software map to the analysis in the paper.

## Basic Verification

- [x] `python scripts/check_package.py` passes.
- [ ] Smoke run completes:

```bash
python run_daily_drought_model.py --stations all --epochs 1 --skip-baselines --run-tag smoke
```

- [ ] Main reproduction command is documented and has been run or marked as computationally expensive.
