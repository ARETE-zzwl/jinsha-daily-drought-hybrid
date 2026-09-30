# Physics-Deep Learning Hybrid Framework for Daily Drought Forecasting

[中文](#中文) · [English](#english)

## 中文

这份仓库提供日尺度干旱预测研究的代码、处理后的站点数据，以及论文图表所用的数据。主要实验覆盖金沙江流域 8 个站点；`v1.1.0-wrr-revision` 版本增加了黄河上游 16 个站点的外部流域评估。

对应稿件：*Physics-Deep Learning Hybrid Framework for Multi-Station Daily Drought Indices and Flash Drought Forecasting in the Jinsha River Basin*。仓库按 *Water Resources Research* 投稿复现材料组织。

### 从哪里开始

需要 Python 3.10 或更新版本。

```bash
git clone https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid.git
cd jinsha-daily-drought-hybrid
python -m venv .venv
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活；macOS / Linux 使用 `source .venv/bin/activate`。然后安装依赖并检查数据：

```bash
python -m pip install -r requirements.txt
python scripts/check_package.py
```

快速试跑：

```bash
python run_daily_drought_model.py --stations all --epochs 1 --skip-baselines --run-tag smoke
```

这条命令用于检查运行流程，不等同于论文实验。

### 数据与复现范围

- `data/processed/`：金沙江流域 8 个站点的日尺度输入和元数据。
- `data/derived/`：论文图表和汇总表所用的 CSV。
- `results/example_run/`：主实验的指标、特征、时间划分和训练记录。
- [Zenodo 归档](https://doi.org/10.5281/zenodo.22232487)：大型预测表、模型权重、黄河上游输入和 CMIP6 辅助训练数据。

目标 `idx_30` 和 `idx_90` 是自定义的 30 日与 90 日标准化水量平衡异常指标。骤旱标签由 `idx_30` 的快速下降和持续干旱状态构造，用于早期风险识别。变量定义见 [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)。

严格复现主实验时，需要从 Zenodo 下载 CMIP6 站点辅助数据。缺少这些文件时，代码会跳过该部分，结果不能视为完全复现论文训练协议。文件名、解压位置和完整参数见下方英文说明及 [CMIP6_AUXILIARY_DATA.md](docs/CMIP6_AUXILIARY_DATA.md)。

### 黄河上游验证

这里包含两个不同协议：

1. **零样本迁移**：使用金沙江训练的 TCN / GRU 权重，不在黄河上游更新权重，也不作本地分类阈值调优。
2. **本地重训**：在黄河上游 16 个站点重新训练整个流程，不使用该流域未提供的匹配 DEM 和 CMIP6 辅助输入。

两者分别检验权重可迁移性和流程的跨流域复现能力。研究结果不支持把本地重训效果解释成普适的零样本泛化；长时段递推迁移也有明显限制。数据包、时间划分、运行命令和结果解释见 [UPPER_YELLOW_REPLICATION.md](docs/UPPER_YELLOW_REPLICATION.md)。

### 引用与许可

使用代码或数据时，请参考 [CITATION.cff](CITATION.cff) 引用归档复现包和对应稿件。代码使用 [MIT](LICENSE)，处理数据与论文派生数据使用 [CC BY 4.0](DATA_LICENSE.md)。

## English

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
