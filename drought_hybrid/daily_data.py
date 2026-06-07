from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import BASE_FEATURES, CMIP_DIR, MODE_NAMES
from .data import decompose, read_station_data, safe_series, split_by_time

REG_TARGET_COLS = ["idx_30", "idx_90"]
CLS_TARGET_COL = "flash_label"


def _static_row_for_daily(station_static_map: Dict[str, np.ndarray], station: str) -> np.ndarray:
    """(1, D) static row; D matches map (supports extended 5-D static). Defined here to avoid import cycles."""
    if station in station_static_map:
        return station_static_map[station].astype(np.float32).reshape(1, -1)
    proto = next(iter(station_static_map.values()))
    return np.zeros_like(proto, dtype=np.float32).reshape(1, -1)


def _zscore_with_train_ref(series: pd.Series, train_len: int) -> pd.Series:
    ref = series.iloc[: max(train_len, 1)].dropna()
    if len(ref) == 0:
        ref = series.dropna()
    mu = float(ref.mean()) if len(ref) > 0 else 0.0
    sd = float(ref.std()) if len(ref) > 0 else 1.0
    sd = 1.0 if abs(sd) < 1e-6 else sd
    return ((series - mu) / sd).fillna(0.0).astype(np.float32)


def add_daily_targets(
    df_daily: pd.DataFrame,
    train_len: int,
    windows: Tuple[int, int] = (30, 90),
    flash_threshold: float = -1.0,
    drop_threshold: float = -0.8,
    lookback_days: int = 14,
    min_duration: int = 5,
) -> pd.DataFrame:
    out = df_daily.copy()
    wb = safe_series(out["pr"]) - safe_series(out["et0"])
    for w in windows:
        roll = wb.rolling(w, min_periods=w).sum()
        out[f"idx_{w}"] = _zscore_with_train_ref(roll, train_len=train_len)
    idx30 = out["idx_30"].astype(np.float32)
    rapid_drop = (idx30 - idx30.shift(max(int(lookback_days), 1))) <= float(drop_threshold)
    persistent_dry = idx30.rolling(max(int(min_duration), 1), min_periods=1).max() <= float(flash_threshold)
    out[CLS_TARGET_COL] = (rapid_drop & persistent_dry).fillna(False).astype(np.float32)
    return out


def build_daily_station_frames(stations: List[str], fold_idx: int, total_folds: int) -> Dict[str, pd.DataFrame]:
    frames = {}
    for st in stations:
        daily = read_station_data(st).sort_values("date").reset_index(drop=True)
        tr, va, te = split_by_time(daily, fold_idx=fold_idx, total_folds=total_folds)
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            continue
        full = pd.concat([tr, va, te], ignore_index=True)
        full = add_daily_targets(full, train_len=len(tr))
        full["station"] = st
        frames[st] = full
    return frames


def build_modal_selection_global_daily(
    station_frames: Dict[str, pd.DataFrame],
    modal_method: str,
    top_k_per_base: int,
    target_col: str = "idx_30",
    fold_idx: int = 0,
    total_folds: int = 1,
):
    """Modal scores / global top-k use **training rows only** (same `split_by_time` as modeling) to avoid target leakage."""
    enriched = {}
    rows = []
    for station, df in station_frames.items():
        work = df.sort_values("date").reset_index(drop=True).copy()
        tr, _, _ = split_by_time(work, fold_idx=fold_idx, total_folds=total_folds)
        n_tr = int(len(tr))
        for base in BASE_FEATURES:
            modes = decompose(work[base], modal_method)
            for mode_name in MODE_NAMES:
                col = f"{base}_mode_{mode_name}"
                work[col] = safe_series(modes[mode_name])
                tr_col = work[col].iloc[:n_tr]
                tr_tgt = work[target_col].iloc[:n_tr]
                corr = abs(tr_col.corr(tr_tgt))
                var = float(tr_col.var())
                score = 0.8 * (0.0 if pd.isna(corr) else corr) + 0.2 * np.log1p(max(var, 0.0))
                rows.append(
                    {
                        "station": station,
                        "base_feature": base,
                        "mode": mode_name,
                        "feature": col,
                        "score": score,
                        "abs_corr_with_target": corr,
                        "variance": var,
                    }
                )
        enriched[station] = work
    score_df = pd.DataFrame(rows)
    selected = []
    if len(score_df) > 0:
        global_scores = score_df.groupby(["base_feature", "mode"], as_index=False)["score"].mean().rename(columns={"score": "global_score"})
        for base in BASE_FEATURES:
            sub = global_scores[global_scores["base_feature"] == base].sort_values("global_score", ascending=False).head(top_k_per_base)
            selected.extend([f"{base}_mode_{m}" for m in sub["mode"].tolist()])
        score_df["selected"] = score_df["feature"].isin(selected)
    return enriched, score_df, selected


def build_feature_frame_daily(df: pd.DataFrame, feature_cols: List[str], modal_method: str) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    cache: Dict[str, Dict[str, pd.Series]] = {}
    for col in feature_cols:
        if "_mode_" in col:
            base = col.split("_mode_")[0]
            mode_name = col.split("_mode_")[1]
            if base not in cache:
                if base in df.columns:
                    cache[base] = decompose(df[base], modal_method)
                else:
                    cache[base] = {m: pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index) for m in MODE_NAMES}
            feat[col] = safe_series(cache[base].get(mode_name, pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)))
        else:
            feat[col] = safe_series(df[col]) if col in df.columns else 0.0
    return feat.ffill().bfill().fillna(0.0).reindex(columns=feature_cols, fill_value=0.0)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        std = np.where(x.std(axis=0) < 1e-6, 1.0, x.std(axis=0))
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class DailyMultiTaskSeqDataset(Dataset):
    def __init__(self, samples: List[Tuple]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_seq, y_reg, y_cls, pr_t, et0_t, prev_reg, sid, static_vec, station_name, phase, date = self.samples[idx]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_reg, dtype=torch.float32),
            torch.tensor(y_cls, dtype=torch.float32),
            torch.tensor(pr_t, dtype=torch.float32),
            torch.tensor(et0_t, dtype=torch.float32),
            torch.tensor(prev_reg, dtype=torch.float32),
            torch.tensor(sid, dtype=torch.long),
            torch.tensor(static_vec, dtype=torch.float32),
            station_name,
            phase,
            date,
        )


def make_daily_samples(
    split_df: pd.DataFrame,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    static_scaler: Standardizer,
    feature_cols: List[str],
    station_to_id: Dict[str, int],
    station_static_map: Dict[str, np.ndarray],
    seq_len: int,
    phase: str,
):
    samples = []
    for station in sorted(split_df["station"].unique()):
        df = split_df[split_df["station"] == station].sort_values("date").reset_index(drop=True)
        x = x_scaler.transform(df[feature_cols].values.astype(np.float32))
        y_reg = y_scaler.transform(df[REG_TARGET_COLS].values.astype(np.float32))
        y_cls = df[[CLS_TARGET_COL]].values.astype(np.float32)
        pr = df["pr"].values.astype(np.float32).reshape(-1, 1)
        et0 = df["et0"].values.astype(np.float32).reshape(-1, 1)
        sid = station_to_id[station]
        static_raw = _static_row_for_daily(station_static_map, station).astype(np.float32)
        static_std = static_scaler.transform(static_raw)[0]
        for i in range(seq_len, len(df)):
            prev_reg = np.array([y_reg[i - 1, 0], y_reg[i - 1, 1]], dtype=np.float32)
            samples.append((x[i - seq_len : i], y_reg[i], y_cls[i], pr[i], et0[i], prev_reg, sid, static_std, station, phase, str(df.loc[i, "date"])))
    return samples


def read_cmip_daily_station(station: str) -> pd.DataFrame:
    fp = CMIP_DIR / f"{station}_cmip_daily_bias_corrected.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    if "date" not in df.columns or "scenario" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    pr_col = "pr_bc" if "pr_bc" in df.columns else ("pr" if "pr" in df.columns else None)
    tx_col = "tasmax_bc" if "tasmax_bc" in df.columns else ("tasmax" if "tasmax" in df.columns else None)
    tn_col = "tasmin_bc" if "tasmin_bc" in df.columns else ("tasmin" if "tasmin" in df.columns else None)
    if pr_col is None or tx_col is None or tn_col is None:
        return pd.DataFrame()
    cm = pd.DataFrame(
        {
            "date": df["date"],
            "scenario": df["scenario"].astype(str),
            "pr": safe_series(df[pr_col]),
            "tmax": safe_series(df[tx_col]),
            "tmin": safe_series(df[tn_col]),
        }
    )
    cm["tmean"] = (cm["tmax"] + cm["tmin"]) / 2.0
    td = np.maximum(cm["tmax"] - cm["tmin"], 0.0)
    cm["et0"] = np.maximum(0.0023 * (cm["tmean"] + 17.8) * np.sqrt(td), 0.0)
    cm["wind"] = 0.0
    cm["rad_net"] = 0.0
    cm["rad_down"] = 0.0
    cm["runoff"] = 0.0
    return cm.sort_values(["scenario", "date"]).reset_index(drop=True)


def build_cmip_daily_contexts(
    stations: List[str],
    feature_cols: List[str],
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    static_scaler: Standardizer,
    station_to_id: Dict[str, int],
    station_static_map: Dict[str, np.ndarray],
    seq_len: int,
    modal_method: str,
    device: torch.device,
):
    ctx_hist = []
    ctx_scen = []
    for st in stations:
        cm = read_cmip_daily_station(st)
        if len(cm) < seq_len + 2:
            continue
        for scn, sub in cm.groupby("scenario"):
            sub = sub.sort_values("date").reset_index(drop=True)
            sub = add_daily_targets(sub, train_len=max(int(len(sub) * 0.6), 1))
            feat = build_feature_frame_daily(sub, feature_cols=feature_cols, modal_method=modal_method)
            x_std = x_scaler.transform(feat.values.astype(np.float32))
            y_std = y_scaler.transform(sub[REG_TARGET_COLS].values.astype(np.float32))
            static_raw = _static_row_for_daily(station_static_map, st).astype(np.float32)
            static_std = static_scaler.transform(static_raw).reshape(-1)
            sid = station_to_id[st]
            for i in range(seq_len, len(sub)):
                item = {
                    "x_seq": torch.tensor(x_std[i - seq_len : i], dtype=torch.float32, device=device).unsqueeze(0),
                    "pr_t": torch.tensor([[float(sub.loc[i, "pr"])]], dtype=torch.float32, device=device),
                    "et0_t": torch.tensor([[float(sub.loc[i, "et0"])]], dtype=torch.float32, device=device),
                    "prev_reg": torch.tensor([[float(y_std[i - 1, 0]), float(y_std[i - 1, 1])]], dtype=torch.float32, device=device),
                    "target_reg_std": torch.tensor(y_std[i : i + 1], dtype=torch.float32, device=device),
                    "station_id": torch.tensor([sid], dtype=torch.long, device=device),
                    "static_vec": torch.tensor(static_std.reshape(1, -1), dtype=torch.float32, device=device),
                }
                if str(scn).lower() == "historical":
                    ctx_hist.append(item)
                else:
                    ctx_scen.append(item)
    return {"historical": ctx_hist, "scenario": ctx_scen}

