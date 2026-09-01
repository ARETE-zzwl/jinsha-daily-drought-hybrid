"""Create a concise, claim-bounded summary of the formal replication run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ONE_STEP_PRIMARY = "fusion_daily_stacking"
RECURSIVE_PRIMARY = "fusion_targetwise_blend_recursive"


def metric_row(metrics: pd.DataFrame, model: str, phase: str, target: str) -> pd.Series:
    rows = metrics[
        (metrics["model"] == model)
        & (metrics["phase"] == phase)
        & (metrics["target"] == target)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for model={model}, phase={phase}, target={target}; "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def reduction(candidate: float, baseline: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    one = pd.read_csv(run_dir / "metrics_daily_model_comparison.csv")
    recursive = pd.read_csv(run_dir / "metrics_daily_model_comparison_recursive.csv")
    one_station = pd.read_csv(run_dir / "metrics_daily_model_comparison_per_station.csv")
    recursive_station = pd.read_csv(
        run_dir / "metrics_daily_model_comparison_recursive_per_station.csv"
    )
    stations = sorted(
        set(one_station["station"].dropna().astype(str))
        | set(recursive_station["station"].dropna().astype(str))
    )
    if len(stations) != 16:
        raise ValueError(f"Expected 16 stations, found {len(stations)}")

    one_primary = metric_row(one, ONE_STEP_PRIMARY, "test", "reg_macro")
    one_persistence = metric_row(one, "persistence", "test", "reg_macro")
    one_linear = metric_row(one, "linear_lstsq", "test", "reg_macro")
    rec_primary = metric_row(recursive, RECURSIVE_PRIMARY, "test_recursive", "reg_macro")
    rec_gbrt = metric_row(recursive, "gbrt_multi_recursive", "test_recursive", "reg_macro")
    rec_hybrid = metric_row(
        recursive, "gru_daily_hybrid_recursive_regcal", "test_recursive", "reg_macro"
    )

    sidecar = run_dir / "metrics_daily_model_comparison_cls_thr_on_valid_sidecar_mcc.csv"
    flash = metric_row(pd.read_csv(sidecar), ONE_STEP_PRIMARY, "test", "flash_label")
    key = pd.DataFrame(
        [
            {
                "forecast_mode": "one_step",
                "model": ONE_STEP_PRIMARY,
                **{key: float(one_primary[key]) for key in ("rmse", "mae", "nse", "kge", "pearson_r")},
            },
            {
                "forecast_mode": "recursive_selected",
                "model": RECURSIVE_PRIMARY,
                **{key: float(rec_primary[key]) for key in ("rmse", "mae", "nse", "kge", "pearson_r")},
            },
            {
                "forecast_mode": "recursive_hybrid",
                "model": "gru_daily_hybrid_recursive_regcal",
                **{key: float(rec_hybrid[key]) for key in ("rmse", "mae", "nse", "kge", "pearson_r")},
            },
        ]
    )
    key.to_csv(run_dir / "formal_replication_key_metrics.csv", index=False)

    lines = [
        "# Formal Upper Yellow River replication summary",
        "",
        f"The formal external-basin replication used {len(stations)} stations and one random seed (42).",
        "",
        (
            f"The prespecified one-step stacking output achieved RMSE = {float(one_primary['rmse']):.5f}, "
            f"MAE = {float(one_primary['mae']):.5f}, and NSE = {float(one_primary['nse']):.5f}. "
            f"Its RMSE was {reduction(float(one_primary['rmse']), float(one_persistence['rmse'])):.1f}% "
            f"lower than persistence and {reduction(float(one_primary['rmse']), float(one_linear['rmse'])):.1f}% "
            "lower than linear least squares."
        ),
        "",
        (
            f"At the validation-selected classification threshold ({float(flash['cls_threshold']):.3f}), "
            f"test F1 = {float(flash['f1']):.3f}, MCC = {float(flash['mcc']):.3f}, "
            f"AUC = {float(flash['auc']):.3f}, and Brier score = {float(flash['brier']):.4f}."
        ),
        "",
        (
            f"The validation-selected target-wise recursive output achieved RMSE = {float(rec_primary['rmse']):.5f} "
            f"and NSE = {float(rec_primary['nse']):.5f}. Validation selected KNN for both regression "
            "targets, so this value cannot be attributed to the hybrid branches. The best calibrated "
            f"hybrid component was GRU (RMSE = {float(rec_hybrid['rmse']):.5f}, NSE = "
            f"{float(rec_hybrid['nse']):.5f}), with RMSE "
            f"{reduction(float(rec_hybrid['rmse']), float(rec_gbrt['rmse'])):.1f}% lower than recursive "
            "gradient boosting but higher than the selected KNN output."
        ),
        "",
        (
            "This run demonstrates reproducibility under local recalibration, not zero-shot transfer. "
            "Optional CMIP6 contexts and static DEM inputs were disabled because matched external-basin "
            "inputs were unavailable."
        ),
    ]
    summary = "\n".join(lines) + "\n"
    (run_dir / "formal_replication_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
