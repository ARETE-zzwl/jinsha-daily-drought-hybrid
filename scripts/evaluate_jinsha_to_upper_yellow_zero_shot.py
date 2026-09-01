"""Evaluate Jinsha-trained no-station-embedding weights in the Upper Yellow Basin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import drought_hybrid.data as data_module
from drought_hybrid.daily_data import (
    CLS_TARGET_COL,
    REG_TARGET_COLS,
    DailyMultiTaskSeqDataset,
    Standardizer,
    build_daily_station_frames,
    build_feature_frame_daily,
    make_daily_samples,
)
from drought_hybrid.daily_models import DailyHybridModel
from drought_hybrid.daily_trainer import (
    apply_stack,
    build_rollout_inputs,
    evaluate_daily_long,
    evaluate_daily_long_per_station,
    recursive_rollout_hybrid,
    run_inference_hybrid,
)
from drought_hybrid.data import split_by_time


SEQ_LEN = 30


def enrich_frames(
    frames: dict[str, pd.DataFrame], feature_cols: list[str]
) -> dict[str, pd.DataFrame]:
    enriched: dict[str, pd.DataFrame] = {}
    for station, frame in frames.items():
        work = frame.sort_values("date").reset_index(drop=True).copy()
        features = build_feature_frame_daily(
            work, feature_cols=feature_cols, modal_method="moving"
        )
        for column in feature_cols:
            work[column] = features[column].values
        enriched[station] = work
    return enriched


def fit_jinsha_preprocessors(
    data_dir: Path, feature_cols: list[str]
) -> tuple[Standardizer, Standardizer, Standardizer]:
    data_module.DATA_DIR = data_dir
    stations = data_module.available_stations()
    if len(stations) != 8:
        raise ValueError(f"Expected 8 Jinsha stations, found {len(stations)}")
    frames = enrich_frames(
        build_daily_station_frames(stations, fold_idx=0, total_folds=1),
        feature_cols,
    )


def load_preprocessors(
    archived_run: Path,
    jinsha_data_dir: Path,
    feature_cols: list[str],
) -> tuple[Standardizer, Standardizer, Standardizer, str]:
    archive = archived_run / "jinsha_zero_shot_preprocessors.npz"
    if archive.is_file():
        arrays = np.load(archive)
        return (
            Standardizer(mean=arrays["x_mean"], std=arrays["x_std"]),
            Standardizer(mean=arrays["y_mean"], std=arrays["y_std"]),
            Standardizer(mean=arrays["static_mean"], std=arrays["static_std"]),
            archive.name,
        )
    print(
        "[ZERO-SHOT] archived preprocessors not found; reconstructing from public Jinsha CSVs",
        flush=True,
    )
    x_scaler, y_scaler, static_scaler = fit_jinsha_preprocessors(
        jinsha_data_dir, feature_cols
    )
    return x_scaler, y_scaler, static_scaler, "reconstructed_from_public_jinsha_data"
    train_parts = []
    for station in stations:
        train, _, _ = split_by_time(frames[station])
        train_parts.append(train)
    train_df = pd.concat(train_parts, ignore_index=True)
    return (
        Standardizer.fit(train_df[feature_cols].values.astype(np.float32)),
        Standardizer.fit(train_df[REG_TARGET_COLS].values.astype(np.float32)),
        Standardizer.fit(np.zeros((len(stations), 3), dtype=np.float32)),
    )


def load_model(
    checkpoint: Path,
    *,
    deep_type: str,
    input_dim: int,
    device: torch.device,
) -> DailyHybridModel:
    model = DailyHybridModel(
        input_dim=input_dim,
        num_stations=1,
        static_dim=3,
        deep_type=deep_type,
        station_emb_dim=0,
        model_dim=96,
        n_heads=4,
        tcn_layers=2,
        gru_layers=1,
        transformer_layers=1,
        dropout=0.05,
        gate_hidden_dim=32,
        fusion_mode="full",
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def one_step_persistence(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, frame in test_df.groupby("station"):
        work = frame.sort_values("date").reset_index(drop=True)
        for index in range(SEQ_LEN, len(work)):
            current = work.iloc[index]
            previous = work.iloc[index - 1]
            for target in REG_TARGET_COLS:
                rows.append(
                    {
                        "station": station,
                        "phase": "test_zero_shot",
                        "date": str(current["date"]),
                        "target": target,
                        "y_true": float(current[target]),
                        "y_pred": float(previous[target]),
                        "model": "persistence",
                    }
                )
            rows.append(
                {
                    "station": station,
                    "phase": "test_zero_shot",
                    "date": str(current["date"]),
                    "target": CLS_TARGET_COL,
                    "y_true": float(current[CLS_TARGET_COL]),
                    "y_pred": float(previous["idx_30"] < -1.0),
                    "model": "persistence",
                }
            )
    return pd.DataFrame(rows)


def recursive_persistence(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, frame in test_df.groupby("station"):
        work = frame.sort_values("date").reset_index(drop=True)
        initial = work.iloc[0]
        for index in range(1, len(work)):
            current = work.iloc[index]
            for target in REG_TARGET_COLS:
                rows.append(
                    {
                        "station": station,
                        "phase": "test_zero_shot_recursive",
                        "date": str(current["date"]),
                        "target": target,
                        "y_true": float(current[target]),
                        "y_pred": float(initial[target]),
                        "model": "persistence_recursive",
                    }
                )
            rows.append(
                {
                    "station": station,
                    "phase": "test_zero_shot_recursive",
                    "date": str(current["date"]),
                    "target": CLS_TARGET_COL,
                    "y_true": float(current[CLS_TARGET_COL]),
                    "y_pred": float(initial["idx_30"] < -1.0),
                    "model": "persistence_recursive",
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jinsha-data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "station_daily",
    )
    parser.add_argument(
        "--upper-yellow-data-dir",
        type=Path,
        default=ROOT / "data" / "external" / "upper_yellow_station_daily",
    )
    parser.add_argument(
        "--archived-run",
        type=Path,
        default=ROOT / "results" / "archived" / "jinsha_no_station_embedding",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "runs" / "upper_yellow_zero_shot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = pd.read_csv(
        args.archived_run / "selected_feature_list.csv", encoding="utf-8-sig"
    )["selected_features"].astype(str).tolist()
    x_scaler, y_scaler, static_scaler, preprocessor_source = load_preprocessors(
        args.archived_run, args.jinsha_data_dir, feature_cols
    )

    data_module.DATA_DIR = args.upper_yellow_data_dir
    yellow_stations = data_module.available_stations()
    if len(yellow_stations) != 16:
        raise ValueError(f"Expected 16 Upper Yellow stations, found {len(yellow_stations)}")
    yellow_frames = enrich_frames(
        build_daily_station_frames(yellow_stations, fold_idx=0, total_folds=1),
        feature_cols,
    )
    test_parts = []
    for station in yellow_stations:
        _, _, test = split_by_time(yellow_frames[station])
        test_parts.append(test)
    test_df = pd.concat(test_parts, ignore_index=True)

    station_to_id = {station: 0 for station in yellow_stations}
    station_static_map = {
        station: np.zeros(3, dtype=np.float32) for station in yellow_stations
    }
    samples = make_daily_samples(
        test_df,
        x_scaler,
        y_scaler,
        static_scaler,
        feature_cols,
        station_to_id,
        station_static_map,
        SEQ_LEN,
        "test_zero_shot",
    )
    loader = DataLoader(
        DailyMultiTaskSeqDataset(samples), batch_size=512, shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[ZERO-SHOT] device={device} stations={len(yellow_stations)} samples={len(samples)}",
        flush=True,
    )

    tcn = load_model(
        args.archived_run / "best_checkpoint_tcn_daily_hybrid.pt",
        deep_type="tcn_transformer",
        input_dim=len(feature_cols),
        device=device,
    )
    gru = load_model(
        args.archived_run / "best_checkpoint_gru_daily_hybrid.pt",
        deep_type="gru_transformer",
        input_dim=len(feature_cols),
        device=device,
    )
    pred_tcn = run_inference_hybrid(tcn, loader, y_scaler, device, "tcn_jinsha_zero_shot")
    pred_gru = run_inference_hybrid(gru, loader, y_scaler, device, "gru_jinsha_zero_shot")
    coefficients = np.loadtxt(
        args.archived_run / "stacking_weights_daily.csv",
        delimiter=",",
        skiprows=1,
        dtype=np.float32,
        encoding="utf-8-sig",
    )
    pred_fusion = apply_stack(
        pred_tcn, pred_gru, coefficients, "fusion_jinsha_zero_shot"
    )
    pred_one = pd.concat(
        [pred_tcn, pred_gru, pred_fusion, one_step_persistence(test_df)],
        ignore_index=True,
    )

    rollout = build_rollout_inputs(
        test_df,
        feature_cols,
        x_scaler,
        y_scaler,
        station_to_id,
        station_static_map,
        static_scaler,
        SEQ_LEN,
    )
    rec_tcn = recursive_rollout_hybrid(
        tcn,
        rollout,
        y_scaler,
        device,
        "tcn_jinsha_zero_shot_recursive",
        "test_zero_shot_recursive",
        recursive_prev_blend=0.55,
    )
    rec_gru = recursive_rollout_hybrid(
        gru,
        rollout,
        y_scaler,
        device,
        "gru_jinsha_zero_shot_recursive",
        "test_zero_shot_recursive",
        recursive_prev_blend=0.55,
    )
    rec_fusion = apply_stack(
        rec_tcn, rec_gru, coefficients, "fusion_jinsha_zero_shot_recursive"
    )
    pred_recursive = pd.concat(
        [rec_tcn, rec_gru, rec_fusion, recursive_persistence(test_df)],
        ignore_index=True,
    )

    outputs = {
        "predictions_zero_shot_one_step.csv": pred_one,
        "predictions_zero_shot_recursive.csv": pred_recursive,
        "metrics_zero_shot_one_step.csv": evaluate_daily_long(pred_one),
        "metrics_zero_shot_recursive.csv": evaluate_daily_long(pred_recursive),
        "metrics_zero_shot_one_step_per_station.csv": evaluate_daily_long_per_station(pred_one),
        "metrics_zero_shot_recursive_per_station.csv": evaluate_daily_long_per_station(pred_recursive),
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    protocol = {
        "source_basin": "Jinsha River Basin",
        "target_basin": "Upper Yellow River Basin",
        "weight_updates_on_target_basin": 0,
        "checkpoint": args.archived_run.name,
        "preprocessor_source": preprocessor_source,
        "station_embedding_dimension": 0,
        "static_features_enabled": False,
        "target_basin_stations": yellow_stations,
        "target_basin_station_count": len(yellow_stations),
        "target_index_climatology_period": ["2010-01-01", "2019-12-21"],
        "evaluation_period": ["2022-02-09", "2024-03-31"],
        "local_threshold_or_model_tuning_on_target_basin": False,
        "recursive_previous_state_blend": 0.55,
        "excluded_station": "Xiaheyan (meteorology/runoff coordinate mismatch)",
    }
    (args.output_dir / "zero_shot_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ZERO-SHOT] output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
