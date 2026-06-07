import argparse
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, MultiTaskElasticNet, MultiTaskLasso, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from .config import DEM_SUMMARY_FILE, OUT_DIR
from .daily_data import (
    CLS_TARGET_COL,
    REG_TARGET_COLS,
    DailyMultiTaskSeqDataset,
    Standardizer,
    build_cmip_daily_contexts,
    build_daily_station_frames,
    build_modal_selection_global_daily,
    make_daily_samples,
)
from .data import available_stations, load_station_static_features, resolve_modal_method, split_by_time, static_fallback_row
from .daily_models import DailyHybridModel

REG_LABELS = {0: "idx_30", 1: "idx_90"}


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    warnings.filterwarnings("ignore", message=".*flash attention.*")


class SeqDailyBaseline(nn.Module):
    def __init__(self, input_dim: int, model_type: str = "cnn", hidden_dim: int = 64):
        super().__init__()
        self.model_type = str(model_type).lower()
        h = int(hidden_dim)
        if self.model_type == "cnn":
            self.enc = nn.Sequential(
                nn.Conv1d(input_dim, h, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(h, h, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(input_dim, h, num_layers=1, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, h, num_layers=1, batch_first=True)
        self.reg_head = nn.Sequential(nn.Linear(h + 2, h), nn.ReLU(), nn.Linear(h, 2))
        self.cls_head = nn.Sequential(nn.Linear(h + 2, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x_seq: torch.Tensor, prev_reg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "cnn":
            h = self.enc(x_seq.transpose(1, 2)).mean(dim=2)
        else:
            o, _ = self.rnn(x_seq)
            h = o[:, -1, :]
        reg = self.reg_head(torch.cat([h, prev_reg], dim=1))
        cls = self.cls_head(torch.cat([h, reg], dim=1))
        return reg, cls


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Pointwise regression metrics on a single vector of observations.

    ``nse`` is set equal to ``r2`` (1 - SS_res/SS_tot with SS_tot vs the sample mean of *y*),
    i.e. the same expression as Nash–Sutcliffe efficiency with observed mean as reference; it is
    not specific to streamflow. ``kge`` uses the Gupta et al. decomposition (also general).
    """
    y = y_true.astype(np.float64)
    p = y_pred.astype(np.float64)
    e = p - y
    mse = float(np.mean(e**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(e)))
    mu_y = float(np.mean(y))
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y - mu_y) ** 2) + 1e-8)
    r2 = float(1.0 - ss_res / ss_tot)
    nse = r2
    mu_p = float(np.mean(p))
    sig_y = float(np.std(y))
    sig_p = float(np.std(p))
    pr = float("nan")
    kge = float("nan")
    if sig_y >= 1e-8 and sig_p >= 1e-8 and len(y) > 1:
        pr = float(np.corrcoef(y, p)[0, 1])
        if np.isnan(pr):
            pr = 0.0
        alpha = sig_p / sig_y
        beta = mu_p / (mu_y + 1e-8)
        kge = float(1.0 - np.sqrt((pr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    pbias = float(100.0 * np.sum(e) / (np.sum(np.abs(y)) + 1e-8))
    mape = float(100.0 * np.mean(np.abs(e) / (np.abs(y) + 1e-6)))
    num_d = float(np.sum(e**2))
    den_d = float(np.sum((np.abs(p - mu_y) + np.abs(y - mu_y)) ** 2) + 1e-8)
    willmott_d = float(1.0 - num_d / den_d)
    pear = pr
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nse": nse,
        "kge": kge,
        "pbias": pbias,
        "pearson_r": float(pear),
        "mape_pct": mape,
        "willmott_d": willmott_d,
    }


def tune_cls_threshold_max_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lo: float = 0.02,
    hi: float = 0.98,
    n_steps: int = 193,
) -> Tuple[float, Dict[str, float]]:
    """Backward-compatible alias for :func:`tune_cls_threshold_by_valid`."""
    return tune_cls_threshold_by_valid(y_true, y_prob, objective="f1", lo=lo, hi=hi, n_steps=n_steps)


def tune_cls_threshold_by_valid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "f1",
    lo: float = 0.02,
    hi: float = 0.98,
    n_steps: int = 193,
) -> Tuple[float, Dict[str, float]]:
    """Grid-search probability threshold on validation to maximize F1 or MCC (flash drought, pooled).

    Returns best threshold and the full metric dict at that threshold. Falls back to 0.5 if no positives.
    """
    y_true = y_true.astype(np.float32).reshape(-1)
    y_prob = np.clip(y_prob.astype(np.float64).reshape(-1), 0.0, 1.0)
    if int(np.sum(y_true >= 0.5)) == 0 or int(np.sum(y_true < 0.5)) == 0:
        m = _metrics_cls(y_true.astype(np.float32), y_prob.astype(np.float32), thr=0.5)
        return 0.5, m
    thrs = np.linspace(float(lo), float(hi), max(3, int(n_steps)))
    best_thr, best_score, best_metrics = 0.5, -1e18, _metrics_cls(y_true.astype(np.float32), y_prob.astype(np.float32), thr=0.5)
    obj = str(objective).lower().strip()
    for t in thrs:
        m = _metrics_cls(y_true.astype(np.float32), y_prob.astype(np.float32), thr=float(t))
        if obj == "mcc":
            s = float(m.get("mcc", float("nan")))
            if np.isnan(s):
                s = -1e18
        else:
            s = float(m.get("f1", 0.0))
        if s > best_score:
            best_score, best_thr, best_metrics = s, float(t), m
    return best_thr, best_metrics


def pooled_cls_thresholds_from_valid(
    pred_df: pd.DataFrame, fit_phase: str = "valid", objective: str = "f1"
) -> Dict[str, float]:
    """One decision threshold per ``model``, tuned on pooled flash samples in ``fit_phase``."""
    sub = pred_df[(pred_df["target"] == CLS_TARGET_COL) & (pred_df["phase"] == fit_phase)].copy()
    out: Dict[str, float] = {}
    if len(sub) == 0:
        return out
    for model, g in sub.groupby("model"):
        thr, _ = tune_cls_threshold_by_valid(
            g["y_true"].values.astype(np.float32),
            g["y_pred"].values.astype(np.float32),
            objective=objective,
        )
        out[str(model)] = float(thr)
    return out


def _metrics_cls(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_hat = (y_prob >= thr).astype(np.int32)
    y_true_i = y_true.astype(np.int32)
    tp = int(np.sum((y_hat == 1) & (y_true_i == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true_i == 0)))
    tn = int(np.sum((y_hat == 0) & (y_true_i == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true_i == 1)))
    prec = float(tp / max(tp + fp, 1))
    rec = float(tp / max(tp + fn, 1))
    f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
    acc = float((tp + tn) / max(tp + tn + fp + fn, 1))
    spec = float(tn / max(tn + fp, 1))
    npv = float(tn / max(tn + fn, 1))
    csi = float(tp / max(tp + fp + fn, 1))
    far = float(fp / max(tp + fp, 1))
    brier = float(np.mean((y_prob.astype(np.float64) - y_true_i.astype(np.float64)) ** 2))
    mcc = float("nan")
    if SKLEARN_AVAILABLE:
        try:
            from sklearn.metrics import matthews_corrcoef as _mcc

            mcc = float(_mcc(y_true_i, y_hat))
        except Exception:
            mcc = float("nan")
    auc = np.nan
    if SKLEARN_AVAILABLE and len(np.unique(y_true_i)) > 1:
        try:
            auc = float(roc_auc_score(y_true_i, y_prob))
        except Exception:
            auc = np.nan
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
        "brier": brier,
        "csi": csi,
        "far": far,
        "specificity": spec,
        "npv": npv,
    }


def evaluate_daily_long(
    pred_df: pd.DataFrame, cls_threshold_by_model: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Aggregate metrics with **pooled samples across stations** (default reporting).

    For each (model, phase, target), all rows from every station are concatenated; :func:`_metrics_reg`
    or :func:`_metrics_cls` is applied once to that pool. ``target == reg_macro`` is the **mean over
    targets** ``idx_30`` and ``idx_90`` of those per-target metrics (same columns), not a separate
    pooled RMSE across both targets at once.

    For **per-station** metrics (same formulas, evaluated inside each station), see
    :func:`evaluate_daily_long_per_station`.

    If ``cls_threshold_by_model`` is provided, flash metrics use ``thr=model_threshold`` instead of 0.5
    (typical: thresholds tuned on validation to maximize F1).
    """
    rows = []
    reg_df = pred_df[pred_df["target"].isin(REG_TARGET_COLS)]
    cls_df = pred_df[pred_df["target"] == CLS_TARGET_COL]
    for (model, phase, target), g in reg_df.groupby(["model", "phase", "target"]):
        m = _metrics_reg(g["y_true"].values.astype(np.float32), g["y_pred"].values.astype(np.float32))
        m.update({"model": model, "phase": phase, "target": target, "task": "reg"})
        rows.append(m)
    if len(cls_df) > 0:
        for (model, phase), g in cls_df.groupby(["model", "phase"]):
            thr = (
                float(cls_threshold_by_model.get(str(model), 0.5))
                if cls_threshold_by_model is not None
                else 0.5
            )
            m = _metrics_cls(g["y_true"].values.astype(np.float32), g["y_pred"].values.astype(np.float32), thr=thr)
            m.update({"model": model, "phase": phase, "target": CLS_TARGET_COL, "task": "cls"})
            if cls_threshold_by_model is not None:
                m["cls_threshold"] = float(thr)
            rows.append(m)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    reg = out[out["task"] == "reg"]
    if len(reg) > 0:
        reg_macro_cols = ["rmse", "mae", "r2", "nse", "kge", "pbias", "pearson_r", "mape_pct", "willmott_d"]
        use_cols = [c for c in reg_macro_cols if c in reg.columns]
        macro = reg.groupby(["model", "phase"], as_index=False)[use_cols].mean()
        macro["target"] = "reg_macro"
        macro["task"] = "reg"
        out = pd.concat([out, macro], ignore_index=True)
    return out


def evaluate_daily_long_per_station(
    pred_df: pd.DataFrame, cls_threshold_by_model: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Same metric definitions as :func:`evaluate_daily_long`, but **within each station**.

    Rows include ``station``. ``reg_macro`` is the mean of ``idx_30`` and ``idx_90`` rows per
    (model, phase, station). Classification metrics are computed per (model, phase, station); rare
    classes at a single site may yield NaN AUC.

    Returns an empty DataFrame if ``station`` is missing from ``pred_df``.
    """
    if pred_df is None or len(pred_df) == 0 or "station" not in pred_df.columns:
        return pd.DataFrame()
    rows = []
    reg_df = pred_df[pred_df["target"].isin(REG_TARGET_COLS)]
    cls_df = pred_df[pred_df["target"] == CLS_TARGET_COL]
    for (model, phase, target, station), g in reg_df.groupby(["model", "phase", "target", "station"]):
        m = _metrics_reg(g["y_true"].values.astype(np.float32), g["y_pred"].values.astype(np.float32))
        m.update({"model": model, "phase": phase, "target": target, "station": station, "task": "reg"})
        rows.append(m)
    if len(cls_df) > 0:
        for (model, phase, station), g in cls_df.groupby(["model", "phase", "station"]):
            thr = (
                float(cls_threshold_by_model.get(str(model), 0.5))
                if cls_threshold_by_model is not None
                else 0.5
            )
            m = _metrics_cls(g["y_true"].values.astype(np.float32), g["y_pred"].values.astype(np.float32), thr=thr)
            m.update({"model": model, "phase": phase, "target": CLS_TARGET_COL, "station": station, "task": "cls"})
            if cls_threshold_by_model is not None:
                m["cls_threshold"] = float(thr)
            rows.append(m)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    reg = out[out["task"] == "reg"]
    if len(reg) > 0:
        reg_macro_cols = ["rmse", "mae", "r2", "nse", "kge", "pbias", "pearson_r", "mape_pct", "willmott_d"]
        use_cols = [c for c in reg_macro_cols if c in reg.columns]
        macro = reg.groupby(["model", "phase", "station"], as_index=False)[use_cols].mean()
        macro["target"] = "reg_macro"
        macro["task"] = "reg"
        out = pd.concat([out, macro], ignore_index=True)
    return out


METRICS_AGGREGATION_NOTES = """日尺度多任务指标汇总说明（与 daily_trainer.evaluate_daily_long 一致）

【池化指标】metrics_daily_model_comparison.csv / metrics_daily_model_comparison_recursive.csv
- 回归：对每个 (model, phase, target)，将所有站点的 (date, station) 样本拼成一条长序列，再计算 RMSE、MAE、R²、NSE 等。
- NSE 在代码中与 R² 相同：1 - SS_res/SS_tot（相对观测均值的技能），并非仅适用于径流；干旱指数同样可报，论文中也可只写 R² 避免水文读者误解。
- reg_macro：对 idx_30 与 idx_90 两行回归指标在列上取算术平均（即两目标指标均值），不是把两目标混在一个向量里算一个 RMSE。
- 分类 flash_label：同样为跨站池化后的样本计算 F1、AUC 等；**主文件** ``metrics_daily_model_comparison*.csv`` 中闪旱 F1/MCC 等**始终**使用决策阈值 **0.5**（与历史归档一致）。若训练时使用 ``--flash-threshold-mode valid_f1``，会**额外**写出 ``*_cls_thr_on_valid_sidecar.csv``（验证集网格搜索阈值后的分类指标），不覆盖主表。

【分站点指标】metrics_daily_model_comparison_per_station.csv / *_recursive_per_station.csv
- 公式与上相同，但在每个 (model, phase, target, station) 内单独计算；reg_macro 为站内 idx_30/idx_90 指标列均值。
- 单站样本少或闪旱正例极少时，部分指标（如 AUC）可能为 NaN。

【per_model_results】各模型子表：池化指标见 *_metrics_one_step.csv / *_metrics_recursive.csv；分站点见 *_metrics_*_per_station.csv（若存在）。
"""


def write_metrics_aggregation_notes(out_dir: Path) -> None:
    p = out_dir / "metrics_daily_aggregation_notes.txt"
    p.write_text(METRICS_AGGREGATION_NOTES, encoding="utf-8")


def write_training_progress(logs: List[Dict[str, float]], out_csv: Path, out_png: Path, title: str) -> None:
    if len(logs) == 0:
        return
    df = pd.DataFrame(logs)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].plot(df["epoch"], df["train_loss"], color="tab:blue")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[1].plot(df["epoch"], df["val_rmse_reg_macro"], color="tab:orange")
    axes[1].set_title("Valid RMSE (Reg Macro)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[2].plot(df["epoch"], df["lr"], color="tab:green")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25, linestyle="--")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_inference_hybrid(model: DailyHybridModel, loader: DataLoader, y_scaler: Standardizer, device: torch.device, model_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for x_seq, y_reg, y_cls, pr_t, et0_t, prev_reg, sid, static_vec, station, phase, date in loader:
            reg, cls, _, _, gate = model(x_seq.to(device), pr_t.to(device), et0_t.to(device), prev_reg.to(device), sid.to(device), static_vec.to(device))
            y_reg_raw = y_scaler.inverse_transform(y_reg.numpy())
            reg_raw = y_scaler.inverse_transform(reg.cpu().numpy())
            cls_prob = torch.sigmoid(cls).cpu().numpy().reshape(-1)
            gate_np = gate.cpu().numpy().reshape(-1)
            for i in range(len(station)):
                for j, nm in REG_LABELS.items():
                    rows.append(
                        {
                            "station": station[i],
                            "phase": phase[i],
                            "date": str(date[i]),
                            "target": nm,
                            "y_true": float(y_reg_raw[i, j]),
                            "y_pred": float(reg_raw[i, j]),
                            "model": model_name,
                            "gate": float(gate_np[i]),
                        }
                    )
                rows.append(
                    {
                        "station": station[i],
                        "phase": phase[i],
                        "date": str(date[i]),
                        "target": CLS_TARGET_COL,
                        "y_true": float(y_cls.numpy()[i, 0]),
                        "y_pred": float(cls_prob[i]),
                        "model": model_name,
                        "gate": float(gate_np[i]),
                    }
                )
    return pd.DataFrame(rows)


def run_inference_seq_baseline(model: SeqDailyBaseline, loader: DataLoader, y_scaler: Standardizer, device: torch.device, model_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for x_seq, y_reg, y_cls, _, _, prev_reg, _, _, station, phase, date in loader:
            reg, cls = model(x_seq.to(device), prev_reg.to(device))
            y_reg_raw = y_scaler.inverse_transform(y_reg.numpy())
            reg_raw = y_scaler.inverse_transform(reg.cpu().numpy())
            cls_prob = torch.sigmoid(cls).cpu().numpy().reshape(-1)
            for i in range(len(station)):
                for j, nm in REG_LABELS.items():
                    rows.append({"station": station[i], "phase": phase[i], "date": str(date[i]), "target": nm, "y_true": float(y_reg_raw[i, j]), "y_pred": float(reg_raw[i, j]), "model": model_name})
                rows.append({"station": station[i], "phase": phase[i], "date": str(date[i]), "target": CLS_TARGET_COL, "y_true": float(y_cls.numpy()[i, 0]), "y_pred": float(cls_prob[i]), "model": model_name})
    return pd.DataFrame(rows)


def build_rollout_inputs(
    split_df: pd.DataFrame,
    feature_cols: List[str],
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    station_to_id: Dict[str, int],
    station_static_map: Dict[str, np.ndarray],
    static_scaler: Standardizer,
    seq_len: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for st in sorted(split_df["station"].unique()):
        dfr = split_df[split_df["station"] == st].sort_values("date").reset_index(drop=True)
        x_std = x_scaler.transform(dfr[feature_cols].values.astype(np.float32))
        y_reg_std = y_scaler.transform(dfr[REG_TARGET_COLS].values.astype(np.float32))
        y_reg_raw = y_scaler.inverse_transform(y_reg_std)
        y_cls = dfr[[CLS_TARGET_COL]].values.astype(np.float32).reshape(-1)
        static_raw = static_fallback_row(station_static_map, st).astype(np.float32)
        static_std = static_scaler.transform(static_raw).reshape(-1)
        out[st] = {
            "x_std": x_std,
            "y_reg_std": y_reg_std,
            "y_reg_raw": y_reg_raw,
            "y_cls": y_cls,
            "pr": dfr["pr"].values.astype(np.float32),
            "et0": dfr["et0"].values.astype(np.float32),
            "date": dfr["date"].astype(str).values,
            "sid": station_to_id[st],
            "static_std": static_std.astype(np.float32),
            "seq_len": seq_len,
        }
    return out


def recursive_rollout_hybrid(
    model: DailyHybridModel,
    rollout_inputs: Dict[str, Dict[str, np.ndarray]],
    y_scaler: Standardizer,
    device: torch.device,
    model_name: str,
    phase_name: str,
    recursive_prev_blend: float = 1.0,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for st, d in rollout_inputs.items():
        x_std = d["x_std"]
        y_reg_std = d["y_reg_std"]
        y_reg_raw = d["y_reg_raw"]
        y_cls = d["y_cls"]
        pr = d["pr"]
        et0 = d["et0"]
        dates = d["date"]
        sid = d["sid"]
        static_std = d["static_std"]
        seq_len = int(d["seq_len"])
        if len(x_std) <= 1:
            continue
        prev_reg = y_reg_std[0].astype(np.float32).copy()
        for i in range(1, len(x_std)):
            win = x_std[max(0, i - seq_len) : i]
            if len(win) < seq_len:
                pad = np.repeat(win[:1], seq_len - len(win), axis=0)
                win = np.vstack([pad, win])
            with torch.no_grad():
                reg, cls, _, _, _ = model(
                    torch.tensor(win, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor([[pr[i]]], dtype=torch.float32, device=device),
                    torch.tensor([[et0[i]]], dtype=torch.float32, device=device),
                    torch.tensor(prev_reg.reshape(1, -1), dtype=torch.float32, device=device),
                    torch.tensor([sid], dtype=torch.long, device=device),
                    torch.tensor(static_std.reshape(1, -1), dtype=torch.float32, device=device),
                )
            reg_std = reg.squeeze(0).cpu().numpy().astype(np.float32)
            reg_raw = y_scaler.inverse_transform(reg_std.reshape(1, -1)).reshape(-1)
            cls_prob = float(torch.sigmoid(cls).cpu().numpy().reshape(-1)[0])
            blend = float(np.clip(recursive_prev_blend, 0.0, 1.0))
            prev_reg = blend * reg_std + (1.0 - blend) * prev_reg
            for j, nm in REG_LABELS.items():
                rows.append({"station": st, "phase": phase_name, "date": dates[i], "target": nm, "y_true": float(y_reg_raw[i, j]), "y_pred": float(reg_raw[j]), "model": model_name})
            rows.append({"station": st, "phase": phase_name, "date": dates[i], "target": CLS_TARGET_COL, "y_true": float(y_cls[i]), "y_pred": cls_prob, "model": model_name})
    return pd.DataFrame(rows)


def recursive_rollout_seq_baseline(
    model: SeqDailyBaseline,
    rollout_inputs: Dict[str, Dict[str, np.ndarray]],
    y_scaler: Standardizer,
    device: torch.device,
    model_name: str,
    phase_name: str,
    recursive_prev_blend: float = 1.0,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for st, d in rollout_inputs.items():
        x_std = d["x_std"]
        y_reg_std = d["y_reg_std"]
        y_reg_raw = d["y_reg_raw"]
        y_cls = d["y_cls"]
        dates = d["date"]
        seq_len = int(d["seq_len"])
        if len(x_std) <= 1:
            continue
        prev_reg = y_reg_std[0].astype(np.float32).copy()
        for i in range(1, len(x_std)):
            win = x_std[max(0, i - seq_len) : i]
            if len(win) < seq_len:
                pad = np.repeat(win[:1], seq_len - len(win), axis=0)
                win = np.vstack([pad, win])
            with torch.no_grad():
                reg, cls = model(
                    torch.tensor(win, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor(prev_reg.reshape(1, -1), dtype=torch.float32, device=device),
                )
            reg_std = reg.squeeze(0).cpu().numpy().astype(np.float32)
            reg_raw = y_scaler.inverse_transform(reg_std.reshape(1, -1)).reshape(-1)
            cls_prob = float(torch.sigmoid(cls).cpu().numpy().reshape(-1)[0])
            blend = float(np.clip(recursive_prev_blend, 0.0, 1.0))
            prev_reg = blend * reg_std + (1.0 - blend) * prev_reg
            for j, nm in REG_LABELS.items():
                rows.append({"station": st, "phase": phase_name, "date": dates[i], "target": nm, "y_true": float(y_reg_raw[i, j]), "y_pred": float(reg_raw[j]), "model": model_name})
            rows.append({"station": st, "phase": phase_name, "date": dates[i], "target": CLS_TARGET_COL, "y_true": float(y_cls[i]), "y_pred": cls_prob, "model": model_name})
    return pd.DataFrame(rows)


def _to_wide_reg(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    reg = df[df["target"].isin(REG_TARGET_COLS)].copy()
    piv = reg.pivot_table(index=["station", "phase", "date"], columns="target", values=pred_col).reset_index().rename(columns={"idx_30": "idx_30_pred", "idx_90": "idx_90_pred"})
    y = reg.pivot_table(index=["station", "phase", "date"], columns="target", values="y_true").reset_index().rename(columns={"idx_30": "idx_30_true", "idx_90": "idx_90_true"})
    out = y.merge(piv, on=["station", "phase", "date"], how="inner")
    return out


def merge_reg_predictions(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> pd.DataFrame:
    a = _to_wide_reg(pred_a, "y_pred").rename(columns={"idx_30_pred": "idx30_a", "idx_90_pred": "idx90_a"})
    b = _to_wide_reg(pred_b, "y_pred").rename(columns={"idx_30_pred": "idx30_b", "idx_90_pred": "idx90_b"})
    return a[["station", "phase", "date", "idx_30_true", "idx_90_true", "idx30_a", "idx90_a"]].merge(
        b[["station", "phase", "date", "idx30_b", "idx90_b"]], on=["station", "phase", "date"], how="inner"
    )


def fit_stack_reg(merged_valid: pd.DataFrame) -> np.ndarray:
    x = np.column_stack(
        [
            merged_valid["idx30_a"].values,
            merged_valid["idx90_a"].values,
            merged_valid["idx30_b"].values,
            merged_valid["idx90_b"].values,
            np.ones(len(merged_valid), dtype=np.float32),
        ]
    ).astype(np.float32)
    y = merged_valid[["idx_30_true", "idx_90_true"]].values.astype(np.float32)
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coef.astype(np.float32)


def apply_stack(pred_a: pd.DataFrame, pred_b: pd.DataFrame, coef: np.ndarray, model_name: str) -> pd.DataFrame:
    m = merge_reg_predictions(pred_a, pred_b)
    x = np.column_stack([m["idx30_a"].values, m["idx90_a"].values, m["idx30_b"].values, m["idx90_b"].values, np.ones(len(m), dtype=np.float32)]).astype(np.float32)
    yhat = x @ coef
    cls_a = pred_a[pred_a["target"] == CLS_TARGET_COL][["station", "phase", "date", "y_true", "y_pred"]].rename(columns={"y_true": "y_true_cls", "y_pred": "p_a"})
    cls_b = pred_b[pred_b["target"] == CLS_TARGET_COL][["station", "phase", "date", "y_pred"]].rename(columns={"y_pred": "p_b"})
    cls = cls_a.merge(cls_b, on=["station", "phase", "date"], how="inner")
    cls["p"] = 0.5 * cls["p_a"] + 0.5 * cls["p_b"]
    rows = []
    for i in range(len(m)):
        rows.append({"station": m.loc[i, "station"], "phase": m.loc[i, "phase"], "date": m.loc[i, "date"], "target": "idx_30", "y_true": float(m.loc[i, "idx_30_true"]), "y_pred": float(yhat[i, 0]), "model": model_name})
        rows.append({"station": m.loc[i, "station"], "phase": m.loc[i, "phase"], "date": m.loc[i, "date"], "target": "idx_90", "y_true": float(m.loc[i, "idx_90_true"]), "y_pred": float(yhat[i, 1]), "model": model_name})
    for i in range(len(cls)):
        rows.append({"station": cls.iloc[i]["station"], "phase": cls.iloc[i]["phase"], "date": cls.iloc[i]["date"], "target": CLS_TARGET_COL, "y_true": float(cls.iloc[i]["y_true_cls"]), "y_pred": float(cls.iloc[i]["p"]), "model": model_name})
    return pd.DataFrame(rows)


def _dirichlet_search_weights(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    mode: str,
    n_trials: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    k = int(x_fit.shape[1])
    if k == 1:
        return np.array([1.0], dtype=np.float32), 0.0
    best_w = np.ones(k, dtype=np.float32) / float(k)
    if mode == "rmse":
        y0 = x_fit @ best_w
        best_s = float(np.sqrt(np.mean((y0 - y_fit) ** 2)))
        for _ in range(max(1, int(n_trials))):
            w = rng.dirichlet(np.ones(k, dtype=np.float32)).astype(np.float32)
            yhat = x_fit @ w
            s = float(np.sqrt(np.mean((yhat - y_fit) ** 2)))
            if s < best_s:
                best_s, best_w = s, w
    else:
        thrs = np.linspace(0.2, 0.8, 25)
        y0 = x_fit @ best_w
        m0 = max(_metrics_cls(y_fit, y0, thr=float(t))["f1"] for t in thrs)
        best_s = float(m0)
        for _ in range(max(1, int(n_trials))):
            w = rng.dirichlet(np.ones(k, dtype=np.float32)).astype(np.float32)
            yhat = x_fit @ w
            s = max(_metrics_cls(y_fit, yhat, thr=float(t))["f1"] for t in thrs)
            if s > best_s:
                best_s, best_w = float(s), w
    return best_w, best_s


def build_meta_fusion(
    pred_df: pd.DataFrame,
    fit_phase: str,
    candidate_models: List[str],
    model_name: str,
    seed: int,
    n_trials: int = 1200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cands = [m for m in candidate_models if m in pred_df["model"].unique().tolist()]
    if len(cands) == 0:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    weight_rows = []

    for target in REG_TARGET_COLS:
        sub = pred_df[(pred_df["target"] == target) & (pred_df["model"].isin(cands))].copy()
        if len(sub) == 0:
            continue
        piv = sub.pivot_table(index=["station", "phase", "date"], columns="model", values="y_pred").reset_index()
        y = sub.groupby(["station", "phase", "date"], as_index=False)["y_true"].mean()
        merged = y.merge(piv, on=["station", "phase", "date"], how="inner").dropna()
        if len(merged) < 10:
            continue
        fit = merged[merged["phase"] == fit_phase].copy()
        if len(fit) < 10:
            continue
        x_fit = fit[cands].values.astype(np.float32)
        y_fit = fit["y_true"].values.astype(np.float32)
        w, fit_score = _dirichlet_search_weights(x_fit, y_fit, mode="rmse", n_trials=n_trials, seed=seed + len(rows) + 11)
        for i, m in enumerate(cands):
            weight_rows.append({"meta_model": model_name, "fit_phase": fit_phase, "target": target, "base_model": m, "weight": float(w[i]), "fit_score": float(fit_score), "metric": "rmse"})
        x_all = merged[cands].values.astype(np.float32)
        y_hat = x_all @ w
        for i in range(len(merged)):
            rows.append(
                {
                    "station": merged.iloc[i]["station"],
                    "phase": merged.iloc[i]["phase"],
                    "date": merged.iloc[i]["date"],
                    "target": target,
                    "y_true": float(merged.iloc[i]["y_true"]),
                    "y_pred": float(y_hat[i]),
                    "model": model_name,
                }
            )

    subc = pred_df[(pred_df["target"] == CLS_TARGET_COL) & (pred_df["model"].isin(cands))].copy()
    if len(subc) > 0:
        pivc = subc.pivot_table(index=["station", "phase", "date"], columns="model", values="y_pred").reset_index()
        yc = subc.groupby(["station", "phase", "date"], as_index=False)["y_true"].mean()
        mergedc = yc.merge(pivc, on=["station", "phase", "date"], how="inner").dropna()
        fitc = mergedc[mergedc["phase"] == fit_phase].copy()
        if len(fitc) >= 10 and len(np.unique(fitc["y_true"].values.astype(np.int32))) > 1:
            x_fit = fitc[cands].values.astype(np.float32)
            y_fit = fitc["y_true"].values.astype(np.float32)
            w, fit_score = _dirichlet_search_weights(x_fit, y_fit, mode="f1", n_trials=n_trials, seed=seed + 777)
            for i, m in enumerate(cands):
                weight_rows.append({"meta_model": model_name, "fit_phase": fit_phase, "target": CLS_TARGET_COL, "base_model": m, "weight": float(w[i]), "fit_score": float(fit_score), "metric": "f1"})
            x_all = mergedc[cands].values.astype(np.float32)
            y_hat = np.clip(x_all @ w, 0.0, 1.0)
            for i in range(len(mergedc)):
                rows.append(
                    {
                        "station": mergedc.iloc[i]["station"],
                        "phase": mergedc.iloc[i]["phase"],
                        "date": mergedc.iloc[i]["date"],
                        "target": CLS_TARGET_COL,
                        "y_true": float(mergedc.iloc[i]["y_true"]),
                        "y_pred": float(y_hat[i]),
                        "model": model_name,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def _build_baseline_tabular(
    split_df: pd.DataFrame,
    feature_cols: List[str],
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    static_scaler: Standardizer,
    station_to_id: Dict[str, int],
    station_static_map: Dict[str, np.ndarray],
    seq_len: int,
    phase: str,
):
    x_rows, y_reg_rows, y_cls_rows, meta = [], [], [], []
    for st in sorted(split_df["station"].unique()):
        df = split_df[split_df["station"] == st].sort_values("date").reset_index(drop=True)
        x_std = x_scaler.transform(df[feature_cols].values.astype(np.float32))
        y_reg_std = y_scaler.transform(df[REG_TARGET_COLS].values.astype(np.float32))
        y_cls = df[[CLS_TARGET_COL]].values.astype(np.float32)
        sid = float(station_to_id[st])
        static_raw = static_fallback_row(station_static_map, st).astype(np.float32)
        static_std = static_scaler.transform(static_raw).reshape(-1)
        for i in range(seq_len, len(df)):
            last_step = x_std[i - 1]
            mean_step = x_std[i - seq_len : i].mean(axis=0)
            prev_reg = np.array([y_reg_std[i - 1, 0], y_reg_std[i - 1, 1]], dtype=np.float32)
            feat = np.concatenate([last_step, mean_step, np.array([df.loc[i, "pr"], df.loc[i, "et0"]], dtype=np.float32), prev_reg, static_std, np.array([sid], dtype=np.float32)]).astype(np.float32)
            x_rows.append(feat)
            y_reg_rows.append(y_reg_std[i])
            y_cls_rows.append(y_cls[i, 0])
            meta.append((st, phase, str(df.loc[i, "date"])))
    return np.array(x_rows), np.array(y_reg_rows), np.array(y_cls_rows), meta


def _to_long_from_tab(meta: List[Tuple[str, str, str]], y_reg_true_raw: np.ndarray, y_reg_pred_raw: np.ndarray, y_cls_true: np.ndarray, y_cls_prob: np.ndarray, model_name: str) -> pd.DataFrame:
    rows = []
    for i, (st, ph, dt) in enumerate(meta):
        rows.append({"station": st, "phase": ph, "date": dt, "target": "idx_30", "y_true": float(y_reg_true_raw[i, 0]), "y_pred": float(y_reg_pred_raw[i, 0]), "model": model_name})
        rows.append({"station": st, "phase": ph, "date": dt, "target": "idx_90", "y_true": float(y_reg_true_raw[i, 1]), "y_pred": float(y_reg_pred_raw[i, 1]), "model": model_name})
        rows.append({"station": st, "phase": ph, "date": dt, "target": CLS_TARGET_COL, "y_true": float(y_cls_true[i]), "y_pred": float(y_cls_prob[i]), "model": model_name})
    return pd.DataFrame(rows)


def run_tabular_baselines(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    static_scaler: Standardizer,
    station_to_id: Dict[str, int],
    station_static_map: Dict[str, np.ndarray],
    seq_len: int,
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    x_tr, yreg_tr, ycls_tr, m_tr = _build_baseline_tabular(train_df, feature_cols, x_scaler, y_scaler, static_scaler, station_to_id, station_static_map, seq_len, "train")
    x_va, yreg_va, ycls_va, m_va = _build_baseline_tabular(valid_df, feature_cols, x_scaler, y_scaler, static_scaler, station_to_id, station_static_map, seq_len, "valid")
    x_te, yreg_te, ycls_te, m_te = _build_baseline_tabular(test_df, feature_cols, x_scaler, y_scaler, static_scaler, station_to_id, station_static_map, seq_len, "test")
    yreg_tr_raw, yreg_va_raw, yreg_te_raw = y_scaler.inverse_transform(yreg_tr), y_scaler.inverse_transform(yreg_va), y_scaler.inverse_transform(yreg_te)
    pred = {}
    fit_models: Dict[str, object] = {}

    prev_tr = np.vstack([yreg_tr[0:1], yreg_tr[:-1]])
    prev_va = np.vstack([yreg_va[0:1], yreg_va[:-1]])
    prev_te = np.vstack([yreg_te[0:1], yreg_te[:-1]])
    pred["persistence"] = pd.concat(
        [
            _to_long_from_tab(m_tr, yreg_tr_raw, y_scaler.inverse_transform(prev_tr), ycls_tr, (prev_tr[:, 0] < -1.0).astype(np.float32), "persistence"),
            _to_long_from_tab(m_va, yreg_va_raw, y_scaler.inverse_transform(prev_va), ycls_va, (prev_va[:, 0] < -1.0).astype(np.float32), "persistence"),
            _to_long_from_tab(m_te, yreg_te_raw, y_scaler.inverse_transform(prev_te), ycls_te, (prev_te[:, 0] < -1.0).astype(np.float32), "persistence"),
        ],
        ignore_index=True,
    )

    x_tr_lin = np.column_stack([x_tr, np.ones(len(x_tr), dtype=np.float32)])
    coef, _, _, _ = np.linalg.lstsq(x_tr_lin, yreg_tr, rcond=None)
    p_tr = x_tr_lin @ coef
    p_va = np.column_stack([x_va, np.ones(len(x_va), dtype=np.float32)]) @ coef
    p_te = np.column_stack([x_te, np.ones(len(x_te), dtype=np.float32)]) @ coef
    fit_models["linear_lstsq"] = coef
    p_cls_tr = (y_scaler.inverse_transform(p_tr)[:, 0] < -1.0).astype(np.float32)
    p_cls_va = (y_scaler.inverse_transform(p_va)[:, 0] < -1.0).astype(np.float32)
    p_cls_te = (y_scaler.inverse_transform(p_te)[:, 0] < -1.0).astype(np.float32)
    pred["linear_lstsq"] = pd.concat(
        [
            _to_long_from_tab(m_tr, yreg_tr_raw, y_scaler.inverse_transform(p_tr), ycls_tr, p_cls_tr, "linear_lstsq"),
            _to_long_from_tab(m_va, yreg_va_raw, y_scaler.inverse_transform(p_va), ycls_va, p_cls_va, "linear_lstsq"),
            _to_long_from_tab(m_te, yreg_te_raw, y_scaler.inverse_transform(p_te), ycls_te, p_cls_te, "linear_lstsq"),
        ],
        ignore_index=True,
    )

    if SKLEARN_AVAILABLE:
        reg_models = {
            "ridge": Ridge(alpha=1.0, random_state=seed),
            "random_forest": RandomForestRegressor(n_estimators=260, max_depth=10, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "extra_trees": ExtraTreesRegressor(n_estimators=320, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "knn": KNeighborsRegressor(n_neighbors=8, weights="distance"),
            "multitask_lasso": MultiTaskLasso(alpha=0.001, random_state=seed, max_iter=4000),
            "multitask_elasticnet": MultiTaskElasticNet(alpha=0.001, l1_ratio=0.35, random_state=seed, max_iter=4000),
            "gbrt_multi": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=220, learning_rate=0.05, max_depth=3, random_state=seed)),
        }
        cls_models = {
            "logistic": LogisticRegression(max_iter=1200, random_state=seed),
            "rf_cls": RandomForestClassifier(n_estimators=240, max_depth=8, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        }
        for nm, mdl in reg_models.items():
            try:
                mdl.fit(x_tr, yreg_tr)
                fit_models[nm] = mdl
                pr_tr = mdl.predict(x_tr)
                pr_va = mdl.predict(x_va)
                pr_te = mdl.predict(x_te)
                y_pred_raw_tr = y_scaler.inverse_transform(pr_tr)
                y_pred_raw_va = y_scaler.inverse_transform(pr_va)
                y_pred_raw_te = y_scaler.inverse_transform(pr_te)
                ycls_prob_tr = (y_pred_raw_tr[:, 0] < -1.0).astype(np.float32)
                ycls_prob_va = (y_pred_raw_va[:, 0] < -1.0).astype(np.float32)
                ycls_prob_te = (y_pred_raw_te[:, 0] < -1.0).astype(np.float32)
                if "logistic" in cls_models:
                    cls_models["logistic"].fit(x_tr, ycls_tr.astype(np.int32))
                    ycls_prob_tr = cls_models["logistic"].predict_proba(x_tr)[:, 1]
                    ycls_prob_va = cls_models["logistic"].predict_proba(x_va)[:, 1]
                    ycls_prob_te = cls_models["logistic"].predict_proba(x_te)[:, 1]
                pred[nm] = pd.concat(
                    [
                        _to_long_from_tab(m_tr, yreg_tr_raw, y_pred_raw_tr, ycls_tr, ycls_prob_tr, nm),
                        _to_long_from_tab(m_va, yreg_va_raw, y_pred_raw_va, ycls_va, ycls_prob_va, nm),
                        _to_long_from_tab(m_te, yreg_te_raw, y_pred_raw_te, ycls_te, ycls_prob_te, nm),
                    ],
                    ignore_index=True,
                )
            except Exception as ex:
                print(f"[daily baseline] skip {nm}: {ex}", flush=True)
    return pred, fit_models


def run_recursive_tabular_baselines(
    rollout_inputs: Dict[str, Dict[str, np.ndarray]],
    fit_models: Dict[str, object],
    y_scaler: Standardizer,
    phase_name: str,
) -> pd.DataFrame:
    rows = []
    tab_keys = [k for k, v in fit_models.items() if (k == "linear_lstsq" or hasattr(v, "predict"))]
    for st, d in rollout_inputs.items():
        x_std = d["x_std"]
        y_reg_std = d["y_reg_std"]
        y_reg_raw = d["y_reg_raw"]
        y_cls = d["y_cls"]
        pr = d["pr"]
        et0 = d["et0"]
        dates = d["date"]
        static_std = d["static_std"]
        sid = d["sid"]
        seq_len = int(d["seq_len"])
        if len(x_std) <= 1:
            continue
        prev_states = {"persistence": y_reg_std[0].copy()}
        for k in tab_keys:
            prev_states[k] = y_reg_std[0].copy()
        for i in range(1, len(x_std)):
            win = x_std[max(0, i - seq_len) : i]
            if len(win) < seq_len:
                pad = np.repeat(win[:1], seq_len - len(win), axis=0)
                win = np.vstack([pad, win])
            last_step = x_std[i - 1]
            mean_step = win.mean(axis=0)
            pred_now = {}
            for bk in tab_keys:
                prev_vec = prev_states[bk]
                feat = np.concatenate([last_step, mean_step, np.array([pr[i], et0[i]], dtype=np.float32), np.array([prev_vec[0], prev_vec[1]], dtype=np.float32), static_std, np.array([float(sid)], dtype=np.float32)]).astype(np.float32)
                if bk == "linear_lstsq":
                    pred_now[bk] = feat @ fit_models["linear_lstsq"][:-1, :] + fit_models["linear_lstsq"][-1, :]
                else:
                    pred_now[bk] = fit_models[bk].predict(feat.reshape(1, -1))[0]
            for j, nm in REG_LABELS.items():
                rows.append({"station": st, "phase": phase_name, "date": dates[i], "target": nm, "y_true": float(y_reg_raw[i, j]), "y_pred": float(y_scaler.inverse_transform(prev_states["persistence"].reshape(1, -1))[0, j]), "model": "persistence_recursive"})
                for bk in tab_keys:
                    rows.append({"station": st, "phase": phase_name, "date": dates[i], "target": nm, "y_true": float(y_reg_raw[i, j]), "y_pred": float(y_scaler.inverse_transform(pred_now[bk].reshape(1, -1))[0, j]), "model": f"{bk}_recursive"})
            rows.append({"station": st, "phase": phase_name, "date": dates[i], "target": CLS_TARGET_COL, "y_true": float(y_cls[i]), "y_pred": float(prev_states["persistence"][0] < -1.0), "model": "persistence_recursive"})
            for bk in tab_keys:
                rows.append({"station": st, "phase": phase_name, "date": dates[i], "target": CLS_TARGET_COL, "y_true": float(y_cls[i]), "y_pred": float(pred_now[bk][0] < -1.0), "model": f"{bk}_recursive"})
            for bk in tab_keys:
                prev_states[bk] = pred_now[bk]
    return pd.DataFrame(rows)


def save_per_model_results(
    one_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    one_metrics: pd.DataFrame,
    rec_metrics: pd.DataFrame,
    out_dir: Path,
    one_metrics_per_station: Optional[pd.DataFrame] = None,
    rec_metrics_per_station: Optional[pd.DataFrame] = None,
) -> None:
    pm = out_dir / "per_model_results"
    pm.mkdir(parents=True, exist_ok=True)
    for mdl in sorted(one_df["model"].unique().tolist()):
        safe = str(mdl).replace("/", "_")
        one_df[one_df["model"] == mdl].to_csv(pm / f"{safe}_predictions_one_step.csv", index=False, encoding="utf-8-sig")
        one_metrics[one_metrics["model"] == mdl].to_csv(pm / f"{safe}_metrics_one_step.csv", index=False, encoding="utf-8-sig")
        if one_metrics_per_station is not None and len(one_metrics_per_station) > 0:
            sub = one_metrics_per_station[one_metrics_per_station["model"] == mdl]
            if len(sub) > 0:
                sub.to_csv(pm / f"{safe}_metrics_one_step_per_station.csv", index=False, encoding="utf-8-sig")
    for mdl in sorted(rec_df["model"].unique().tolist()):
        safe = str(mdl).replace("/", "_")
        rec_df[rec_df["model"] == mdl].to_csv(pm / f"{safe}_predictions_recursive.csv", index=False, encoding="utf-8-sig")
        rec_metrics[rec_metrics["model"] == mdl].to_csv(pm / f"{safe}_metrics_recursive.csv", index=False, encoding="utf-8-sig")
        if rec_metrics_per_station is not None and len(rec_metrics_per_station) > 0:
            sub = rec_metrics_per_station[rec_metrics_per_station["model"] == mdl]
            if len(sub) > 0:
                sub.to_csv(pm / f"{safe}_metrics_recursive_per_station.csv", index=False, encoding="utf-8-sig")


def build_recursive_reg_calibration(
    rec_df: pd.DataFrame,
    base_models: List[str],
    fit_phase: str = "valid_recursive",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coef_rows = []
    for mdl in base_models:
        sub = rec_df[rec_df["model"] == mdl].copy()
        if len(sub) == 0:
            continue
        cal_name = f"{mdl}_regcal"
        for tgt in REG_TARGET_COLS:
            fit = sub[(sub["phase"] == fit_phase) & (sub["target"] == tgt)].copy()
            if len(fit) < 16:
                continue
            x_fit = np.column_stack([fit["y_pred"].values.astype(np.float32), np.ones(len(fit), dtype=np.float32)])
            y_fit = fit["y_true"].values.astype(np.float32)
            coef, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            coef_rows.append({"model": mdl, "cal_model": cal_name, "target": tgt, "fit_phase": fit_phase, "a": a, "b": b, "n_fit": int(len(fit))})
            for ph in sorted(sub["phase"].unique().tolist()):
                part = sub[(sub["phase"] == ph) & (sub["target"] == tgt)].copy()
                if len(part) == 0:
                    continue
                y_cal = a * part["y_pred"].values.astype(np.float32) + b
                for i in range(len(part)):
                    rows.append(
                        {
                            "station": part.iloc[i]["station"],
                            "phase": ph,
                            "date": part.iloc[i]["date"],
                            "target": tgt,
                            "y_true": float(part.iloc[i]["y_true"]),
                            "y_pred": float(y_cal[i]),
                            "model": cal_name,
                        }
                    )
        # Keep classification output unchanged to preserve flash skill.
        cls = sub[sub["target"] == CLS_TARGET_COL].copy()
        for i in range(len(cls)):
            rows.append(
                {
                    "station": cls.iloc[i]["station"],
                    "phase": cls.iloc[i]["phase"],
                    "date": cls.iloc[i]["date"],
                    "target": CLS_TARGET_COL,
                    "y_true": float(cls.iloc[i]["y_true"]),
                    "y_pred": float(cls.iloc[i]["y_pred"]),
                    "model": cal_name,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


def build_taskwise_recursive_fusion(
    rec_df: pd.DataFrame,
    fit_phase: str = "valid_recursive",
    model_name: str = "fusion_taskwise_recursive",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reg_fit = rec_df[(rec_df["phase"] == fit_phase) & (rec_df["target"].isin(REG_TARGET_COLS))].copy()
    cls_fit = rec_df[(rec_df["phase"] == fit_phase) & (rec_df["target"] == CLS_TARGET_COL)].copy()
    if len(reg_fit) == 0 or len(cls_fit) == 0:
        return pd.DataFrame(), pd.DataFrame()
    reg_scores = []
    for mdl, g in reg_fit.groupby("model"):
        rmse = float(np.sqrt(np.mean((g["y_pred"].values.astype(np.float32) - g["y_true"].values.astype(np.float32)) ** 2)))
        reg_scores.append((mdl, rmse))
    reg_scores = sorted(reg_scores, key=lambda x: x[1])
    best_reg = reg_scores[0][0]
    cls_scores = []
    for mdl, g in cls_fit.groupby("model"):
        m = _metrics_cls(g["y_true"].values.astype(np.float32), g["y_pred"].values.astype(np.float32), thr=0.5)
        cls_scores.append((mdl, float(m["f1"])))
    cls_scores = sorted(cls_scores, key=lambda x: x[1], reverse=True)
    best_cls = cls_scores[0][0]
    rows = []
    for ph in sorted(rec_df["phase"].unique().tolist()):
        for tgt in REG_TARGET_COLS:
            src = rec_df[(rec_df["phase"] == ph) & (rec_df["target"] == tgt) & (rec_df["model"] == best_reg)].copy()
            for i in range(len(src)):
                rows.append(
                    {
                        "station": src.iloc[i]["station"],
                        "phase": ph,
                        "date": src.iloc[i]["date"],
                        "target": tgt,
                        "y_true": float(src.iloc[i]["y_true"]),
                        "y_pred": float(src.iloc[i]["y_pred"]),
                        "model": model_name,
                    }
                )
        src_cls = rec_df[(rec_df["phase"] == ph) & (rec_df["target"] == CLS_TARGET_COL) & (rec_df["model"] == best_cls)].copy()
        for i in range(len(src_cls)):
            rows.append(
                {
                    "station": src_cls.iloc[i]["station"],
                    "phase": ph,
                    "date": src_cls.iloc[i]["date"],
                    "target": CLS_TARGET_COL,
                    "y_true": float(src_cls.iloc[i]["y_true"]),
                    "y_pred": float(src_cls.iloc[i]["y_pred"]),
                    "model": model_name,
                }
            )
    info = pd.DataFrame([{"model": model_name, "fit_phase": fit_phase, "best_reg_model": best_reg, "best_cls_model": best_cls}])
    return pd.DataFrame(rows), info


def build_recursive_targetwise_blend(
    rec_df: pd.DataFrame,
    fit_phase: str = "valid_recursive",
    model_name: str = "fusion_targetwise_blend_recursive",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    info_rows = []
    for tgt in REG_TARGET_COLS:
        fit = rec_df[(rec_df["phase"] == fit_phase) & (rec_df["target"] == tgt)].copy()
        if len(fit) < 20:
            continue
        main_fit = fit[fit["model"].astype(str).str.contains("hybrid|fusion", case=False, regex=True)].copy()
        base_fit = fit[~fit["model"].astype(str).str.contains("hybrid|fusion", case=False, regex=True)].copy()
        if len(main_fit) == 0 or len(base_fit) == 0:
            continue
        best_main = (
            main_fit.groupby("model")
            .apply(lambda g: float(np.sqrt(np.mean((g["y_pred"].values.astype(np.float32) - g["y_true"].values.astype(np.float32)) ** 2))))
            .sort_values()
            .index[0]
        )
        best_base = (
            base_fit.groupby("model")
            .apply(lambda g: float(np.sqrt(np.mean((g["y_pred"].values.astype(np.float32) - g["y_true"].values.astype(np.float32)) ** 2))))
            .sort_values()
            .index[0]
        )
        fit_base = fit[fit["model"] == best_base][["station", "phase", "date", "target", "y_true", "y_pred"]].rename(columns={"y_pred": "pred_base"})
        fit_main = fit[fit["model"] == best_main][["station", "phase", "date", "target", "y_pred"]].rename(columns={"y_pred": "pred_main"})
        merged_fit = fit_base.merge(fit_main, on=["station", "phase", "date", "target"], how="inner")
        if len(merged_fit) < 20:
            continue
        best_rmse, best_alpha = 1e18, 0.0
        for a in np.linspace(0.0, 1.0, 101):
            p = a * merged_fit["pred_base"].values.astype(np.float32) + (1.0 - a) * merged_fit["pred_main"].values.astype(np.float32)
            y = merged_fit["y_true"].values.astype(np.float32)
            r = float(np.sqrt(np.mean((p - y) ** 2)))
            if r < best_rmse:
                best_rmse, best_alpha = r, float(a)
        info_rows.append(
            {
                "model": model_name,
                "fit_phase": fit_phase,
                "target": tgt,
                "best_base_model": str(best_base),
                "best_main_model": str(best_main),
                "alpha_base": float(best_alpha),
                "fit_rmse": float(best_rmse),
            }
        )
        for ph in sorted(rec_df["phase"].unique().tolist()):
            pb = rec_df[(rec_df["phase"] == ph) & (rec_df["target"] == tgt) & (rec_df["model"] == best_base)][["station", "phase", "date", "target", "y_true", "y_pred"]].rename(columns={"y_pred": "pred_base"})
            pm = rec_df[(rec_df["phase"] == ph) & (rec_df["target"] == tgt) & (rec_df["model"] == best_main)][["station", "phase", "date", "target", "y_pred"]].rename(columns={"y_pred": "pred_main"})
            m = pb.merge(pm, on=["station", "phase", "date", "target"], how="inner")
            if len(m) == 0:
                continue
            pred = best_alpha * m["pred_base"].values.astype(np.float32) + (1.0 - best_alpha) * m["pred_main"].values.astype(np.float32)
            for i in range(len(m)):
                rows.append(
                    {
                        "station": m.iloc[i]["station"],
                        "phase": m.iloc[i]["phase"],
                        "date": m.iloc[i]["date"],
                        "target": tgt,
                        "y_true": float(m.iloc[i]["y_true"]),
                        "y_pred": float(pred[i]),
                        "model": model_name,
                    }
                )
    cls_fit = rec_df[(rec_df["phase"] == fit_phase) & (rec_df["target"] == CLS_TARGET_COL)].copy()
    if len(cls_fit) > 0:
        cls_scores = []
        for mdl, g in cls_fit.groupby("model"):
            m = _metrics_cls(g["y_true"].values.astype(np.float32), g["y_pred"].values.astype(np.float32), thr=0.5)
            cls_scores.append((mdl, float(m["f1"])))
        cls_scores = sorted(cls_scores, key=lambda x: x[1], reverse=True)
        best_cls = cls_scores[0][0]
        info_rows.append({"model": model_name, "fit_phase": fit_phase, "target": CLS_TARGET_COL, "best_cls_model": best_cls, "fit_f1": float(cls_scores[0][1])})
        src_cls = rec_df[(rec_df["model"] == best_cls) & (rec_df["target"] == CLS_TARGET_COL)].copy()
        for i in range(len(src_cls)):
            rows.append(
                {
                    "station": src_cls.iloc[i]["station"],
                    "phase": src_cls.iloc[i]["phase"],
                    "date": src_cls.iloc[i]["date"],
                    "target": CLS_TARGET_COL,
                    "y_true": float(src_cls.iloc[i]["y_true"]),
                    "y_pred": float(src_cls.iloc[i]["y_pred"]),
                    "model": model_name,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(info_rows)


def train_one_hybrid(
    model_name: str,
    model: DailyHybridModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    valid_rollout_inputs: Dict[str, Dict[str, np.ndarray]],
    y_scaler: Standardizer,
    device: torch.device,
    args,
    cmip_contexts: Dict[str, List[Dict[str, torch.Tensor]]],
    run_out: Path,
    tag: str,
    model_order: int = 1,
    total_models: int = 2,
) -> Tuple[DailyHybridModel, pd.DataFrame]:
    base_lr = float(args.lr)
    warm_ep = max(0, int(getattr(args, "lr_warmup_epochs", 0)))
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.min_lr)
    huber = nn.SmoothL1Loss(beta=0.5)
    mse = nn.MSELoss()
    cls_pos_weight = float(max(getattr(args, "_cls_pos_weight", 1.0), 1e-3))
    cls_pos_tensor = torch.tensor([cls_pos_weight], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=cls_pos_tensor)
    best = 1e18
    bad = 0
    best_state = None
    logs = []
    progress_csv = run_out / f"training_progress_{tag}.csv"
    progress_png = run_out / f"training_progress_{tag}.png"
    print(f"[TRAIN] start model={model_name} ({model_order}/{max(total_models,1)})", flush=True)
    if warm_ep > 0:
        print(f"[TRAIN] lr warmup epochs={warm_ep} (linear 0.1*lr -> lr)", flush=True)
    for ep in range(1, args.epochs + 1):
        if warm_ep > 0 and ep <= warm_ep:
            alpha = float(ep) / float(max(warm_ep, 1))
            lr_now = base_lr * (0.1 + 0.9 * alpha)
            for g in opt.param_groups:
                g["lr"] = lr_now
        model.train()
        tr_loss = 0.0
        for x_seq, y_reg, y_cls, pr_t, et0_t, prev_reg, sid, static_vec, _, _, _ in train_loader:
            x_seq = x_seq.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)
            pr_t = pr_t.to(device)
            et0_t = et0_t.to(device)
            prev_reg = prev_reg.to(device)
            sid = sid.to(device)
            static_vec = static_vec.to(device)
            reg, cls, _, _, _ = model(x_seq, pr_t, et0_t, prev_reg, sid, static_vec)
            loss_reg_30 = 0.7 * huber(reg[:, 0:1], y_reg[:, 0:1]) + 0.3 * mse(reg[:, 0:1], y_reg[:, 0:1])
            loss_reg_90 = 0.7 * huber(reg[:, 1:2], y_reg[:, 1:2]) + 0.3 * mse(reg[:, 1:2], y_reg[:, 1:2])
            loss_reg = loss_reg_30 + float(args.lambda_idx90) * loss_reg_90
            if str(args.cls_loss_type).lower() == "focal":
                ce = F.binary_cross_entropy_with_logits(cls, y_cls, pos_weight=cls_pos_tensor, reduction="none")
                p = torch.sigmoid(cls)
                pt = y_cls * p + (1.0 - y_cls) * (1.0 - p)
                loss_cls = ((1.0 - pt) ** float(args.focal_gamma) * ce).mean()
            else:
                loss_cls = bce(cls, y_cls)
            wb_proxy = torch.clamp((pr_t - et0_t) / (torch.abs(pr_t) + torch.abs(et0_t) + 1e-6), -3.0, 3.0)
            loss_phy = mse(reg[:, 0:1], wb_proxy)
            loss_anchor = mse(reg[:, 1:2], prev_reg[:, 1:2])
            loss = loss_reg + args.lambda_cls * loss_cls + args.lambda_wb * loss_phy + float(args.lambda_prev_anchor) * loss_anchor
            if args.recursive_consistency_weight > 0:
                mix = min(0.6, 0.6 * float(ep) / max(args.epochs, 1))
                blend = float(np.clip(args.recursive_prev_blend, 0.0, 1.0))
                roll_target = blend * reg.detach() + (1.0 - blend) * prev_reg
                prev_roll = (1.0 - mix) * prev_reg + mix * roll_target
                n_steps = max(int(args.recursive_unroll_steps), 1)
                decay = float(np.clip(args.recursive_unroll_decay, 0.1, 1.0))
                unroll_reg = 0.0
                unroll_cls = 0.0
                w_sum = 0.0
                cur_prev = prev_roll
                for step in range(n_steps):
                    reg_roll, cls_roll, _, _, _ = model(x_seq, pr_t, et0_t, cur_prev, sid, static_vec)
                    w = decay**step
                    w_sum += w
                    loss_roll_reg = mse(reg_roll[:, 0:1], y_reg[:, 0:1]) + float(args.lambda_idx90) * mse(reg_roll[:, 1:2], y_reg[:, 1:2])
                    if str(args.cls_loss_type).lower() == "focal":
                        ce_roll = F.binary_cross_entropy_with_logits(cls_roll, y_cls, pos_weight=cls_pos_tensor, reduction="none")
                        p_roll = torch.sigmoid(cls_roll)
                        pt_roll = y_cls * p_roll + (1.0 - y_cls) * (1.0 - p_roll)
                        cls_roll_loss = ((1.0 - pt_roll) ** float(args.focal_gamma) * ce_roll).mean()
                    else:
                        cls_roll_loss = bce(cls_roll, y_cls)
                    unroll_reg = unroll_reg + w * loss_roll_reg
                    unroll_cls = unroll_cls + w * cls_roll_loss
                    cur_prev = blend * reg_roll.detach() + (1.0 - blend) * cur_prev.detach()
                unroll_reg = unroll_reg / max(w_sum, 1e-6)
                unroll_cls = unroll_cls / max(w_sum, 1e-6)
                loss = loss + args.recursive_consistency_weight * (unroll_reg + 0.35 * unroll_cls)
            if args.lambda_cmip_hist > 0 and len(cmip_contexts.get("historical", [])) > 0:
                idx = (ep + int(sid[0].item())) % len(cmip_contexts["historical"])
                ctx = cmip_contexts["historical"][idx]
                reg_h, _, _, _, _ = model(ctx["x_seq"], ctx["pr_t"], ctx["et0_t"], ctx["prev_reg"], ctx["station_id"], ctx["static_vec"])
                loss = loss + float(args.lambda_cmip_hist) * mse(reg_h, ctx["target_reg_std"])
            if args.lambda_cmip_scenario > 0 and len(cmip_contexts.get("scenario", [])) > 0:
                idx2 = (ep + int(sid[0].item()) * 3) % len(cmip_contexts["scenario"])
                ctx2 = cmip_contexts["scenario"][idx2]
                reg_s, _, _, _, _ = model(ctx2["x_seq"], ctx2["pr_t"], ctx2["et0_t"], ctx2["prev_reg"], ctx2["station_id"], ctx2["static_vec"])
                loss = loss + float(args.lambda_cmip_scenario) * mse(reg_s, ctx2["target_reg_std"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item())
        pred_va = run_inference_hybrid(model, valid_loader, y_scaler, device, model_name)
        met = evaluate_daily_long(pred_va)
        v = float(met[(met["phase"] == "valid") & (met["target"] == "reg_macro")]["rmse"].iloc[0])
        if warm_ep <= 0 or ep > warm_ep:
            sch.step(v)
        logs.append({"model": model_name, "epoch": ep, "train_loss": tr_loss / max(len(train_loader), 1), "val_rmse_reg_macro": v, "lr": float(opt.param_groups[0]["lr"])})
        write_training_progress(logs, progress_csv, progress_png, f"{model_name} Training Progress")
        if v < best - args.early_stop_min_delta:
            best = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        global_pct = 100.0 * ((float(model_order - 1) + float(ep) / float(max(args.epochs, 1))) / float(max(total_models, 1)))
        print(
            f"[PROGRESS {global_pct:6.2f}%] model={model_name} ({model_order}/{max(total_models,1)}) "
            f"epoch={ep}/{args.epochs} train_loss={logs[-1]['train_loss']:.4f} "
            f"valid_rmse={v:.4f} lr={logs[-1]['lr']:.2e}",
            flush=True,
        )
        if bad >= args.early_stop_patience:
            print(f"[{model_name}] early stop at epoch={ep}, best={best:.4f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_out / f"best_checkpoint_{tag}.pt")
    model_end_pct = 100.0 * float(model_order) / float(max(total_models, 1))
    print(f"[PROGRESS {model_end_pct:6.2f}%] model={model_name} completed", flush=True)
    return model, pd.DataFrame(logs)


def parse_args():
    p = argparse.ArgumentParser(description="Daily multi-task drought forecasting")
    p.add_argument("--stations", type=str, default="all")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k-per-base", type=int, default=2)
    p.add_argument("--modal-method", type=str, default="moving", choices=["moving", "eemd", "vmd"])
    p.add_argument("--lambda-wb", type=float, default=0.06)
    p.add_argument("--lambda-cls", type=float, default=0.8)
    p.add_argument("--lambda-idx90", type=float, default=1.25)
    p.add_argument("--lambda-prev-anchor", type=float, default=0.08)
    p.add_argument("--cls-loss-type", type=str, default="focal", choices=["bce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument(
        "--flash-threshold-mode",
        type=str,
        default="fixed05",
        choices=["fixed05", "valid_f1"],
        help=(
            "fixed05: only canonical metrics (flash prob threshold 0.5). "
            "valid_f1: same canonical files, plus sidecar CSVs with per-model thresholds tuned on valid "
            "(does not change metrics_daily_model_comparison*.csv semantics)."
        ),
    )
    p.add_argument(
        "--flash-threshold-objective",
        type=str,
        default="f1",
        choices=["f1", "mcc"],
        help="when --flash-threshold-mode=valid_f1: maximize F1 or MCC on validation for threshold grid",
    )
    p.add_argument("--cls-pos-weight", type=float, default=-1.0)
    p.add_argument("--lambda-cmip-hist", type=float, default=0.03)
    p.add_argument("--lambda-cmip-scenario", type=float, default=0.02)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-patience", type=int, default=4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument(
        "--lr-warmup-epochs",
        type=int,
        default=0,
        help="Linear LR warmup for hybrid training: first N epochs ramp base lr from 0.1*lr to lr; 0 disables (default, unchanged behavior).",
    )
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--rolling-folds", type=int, default=1)
    p.add_argument("--dem-summary-file", type=str, default=str(DEM_SUMMARY_FILE))
    p.add_argument(
        "--station-static-mode",
        type=str,
        default="default",
        choices=["default", "moran_ref", "extended_dem"],
        help="Station static vector: default=[elev,lat,lon] from CSV; moran_ref=lat/lon from project reference dict; extended_dem adds log1p(elev) and elev/max_elev (5-D).",
    )
    p.add_argument("--recursive-consistency-weight", type=float, default=0.12)
    p.add_argument("--recursive-prev-blend", type=float, default=1.0)
    p.add_argument("--recursive-unroll-steps", type=int, default=1)
    p.add_argument("--recursive-unroll-decay", type=float, default=0.7)
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--tcn-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=1)
    p.add_argument("--transformer-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gate-hidden-dim", type=int, default=32)
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--disable-static-dem", action="store_true", default=False)
    p.add_argument(
        "--model-ablation-mode",
        type=str,
        default="full",
        choices=["full", "dl_only", "phy_only", "fixed_gate"],
    )
    p.add_argument("--fixed-gate-value", type=float, default=0.5, help="Used when --model-ablation-mode=fixed_gate.")
    p.add_argument("--station-emb-dim", type=int, default=8, help="0 disables station embedding (ablation).")
    p.add_argument("--deep-baseline-epochs", type=int, default=10)
    p.add_argument("--deep-baseline-hidden", type=int, default=64)
    p.add_argument("--deep-baseline-lr", type=float, default=8e-4)
    p.add_argument("--meta-trials", type=int, default=1600)
    p.add_argument("--skip-baselines", action="store_true", default=False)
    p.add_argument("--disable-recursive-calibration", action="store_true", default=False)
    p.add_argument("--daily-shap", action="store_true", default=False, help="After one-step predictions, run SHAP on tabular features (uses validation subset).")
    p.add_argument(
        "--replicate-seeds",
        type=int,
        default=1,
        help="Run the full pipeline for seed, seed+1, ... (each gets its own run-tag suffix _seed{seed}) and write aggregate uncertainty CSVs.",
    )
    return p.parse_args()


def build_daily_data_bundle(args, fold_idx: int, total_folds: int) -> Dict[str, object]:
    set_seed(args.seed + fold_idx)
    tag = f"_{args.run_tag}" if str(args.run_tag).strip() else ""
    run_out = OUT_DIR / f"daily_multitask_joint_{len(args.stations)}stations{tag}"
    run_out.mkdir(parents=True, exist_ok=True)

    station_frames = build_daily_station_frames(args.stations, fold_idx=fold_idx, total_folds=total_folds)
    enriched, score_df, selected_modal = build_modal_selection_global_daily(
        station_frames,
        modal_method=args.modal_method,
        top_k_per_base=args.top_k_per_base,
        target_col="idx_30",
        fold_idx=fold_idx,
        total_folds=total_folds,
    )
    feature_cols: List[str] = []
    for c in ["pr", "tmean", "tmax", "tmin", "et0", "wind", "rad_net", "rad_down"] + selected_modal:
        if c not in feature_cols:
            feature_cols.append(c)

    train_parts, valid_parts, test_parts = [], [], []
    for st in args.stations:
        sub = enriched[st].sort_values("date").reset_index(drop=True)
        tr, va, te = split_by_time(sub, fold_idx=fold_idx, total_folds=total_folds)
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            continue
        train_parts.append(tr)
        valid_parts.append(va)
        test_parts.append(te)
    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    t_train_max = pd.to_datetime(train_df["date"], errors="coerce").max()
    t_valid_min = pd.to_datetime(valid_df["date"], errors="coerce").min()
    t_valid_max = pd.to_datetime(valid_df["date"], errors="coerce").max()
    t_test_min = pd.to_datetime(test_df["date"], errors="coerce").min()
    if not (t_train_max < t_valid_min and t_valid_max < t_test_min):
        raise ValueError("Daily split leakage risk detected.")
    pos_n = float(train_df[CLS_TARGET_COL].sum())
    neg_n = float(len(train_df) - pos_n)
    auto_pos_weight = neg_n / max(pos_n, 1.0)
    args._cls_pos_weight = float(args.cls_pos_weight) if float(args.cls_pos_weight) > 0 else float(auto_pos_weight)

    split_rows = []
    for ph, dfx in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        dts = pd.to_datetime(dfx["date"], errors="coerce").dropna()
        split_rows.append({"phase": ph, "global_start": str(dts.min().date()), "global_end": str(dts.max().date()), "n_rows": int(len(dfx))})
        for st, gs in dfx.groupby("station"):
            sd = pd.to_datetime(gs["date"], errors="coerce").dropna()
            split_rows.append({"phase": ph, "station": st, "start": str(sd.min().date()), "end": str(sd.max().date()), "n_rows": int(len(gs))})

    station_static_map = load_station_static_features(
        args.stations,
        dem_summary_file=Path(args.dem_summary_file),
        static_mode=str(getattr(args, "station_static_mode", "default")),
    )
    if args.disable_static_dem:
        station_static_map = {k: np.zeros_like(v).astype(np.float32) for k, v in station_static_map.items()}
    x_scaler = Standardizer.fit(train_df[feature_cols].values.astype(np.float32))
    y_scaler = Standardizer.fit(train_df[REG_TARGET_COLS].values.astype(np.float32))
    static_scaler = Standardizer.fit(np.vstack([station_static_map[s] for s in args.stations]).astype(np.float32))
    station_to_id = {st: i for i, st in enumerate(args.stations)}
    return {
        "run_out": run_out,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "split_rows": split_rows,
        "score_df": score_df,
        "station_static_map": station_static_map,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "static_scaler": static_scaler,
        "station_to_id": station_to_id,
    }


def daily_tabular_feature_names(feature_cols: List[str], static_dim: int) -> List[str]:
    names: List[str] = [f"last_{c}" for c in feature_cols]
    names += [f"mean_{c}" for c in feature_cols]
    names += ["pr_raw", "et0_raw", "prev_idx30_std", "prev_idx90_std"]
    names += [f"static_{i}" for i in range(int(static_dim))]
    names += ["station_id_float"]
    return names


def run_daily_shap_analysis(
    run_out: Path,
    feature_cols: List[str],
    static_dim: int,
    valid_df: pd.DataFrame,
    train_df: pd.DataFrame,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    static_scaler: Standardizer,
    station_to_id: Dict[str, int],
    station_static_map: Dict[str, np.ndarray],
    seq_len: int,
    fit_models: Dict[str, object],
    one_df: pd.DataFrame,
    seed: int,
) -> None:
    if not SHAP_AVAILABLE:
        print("[SHAP] shap package not installed, skip daily SHAP (pip install shap).", flush=True)
        return
    if not SKLEARN_AVAILABLE:
        print("[SHAP] sklearn unavailable, skip daily SHAP.", flush=True)
        return
    x_tr, yreg_tr, ycls_tr, _ = _build_baseline_tabular(train_df, feature_cols, x_scaler, y_scaler, static_scaler, station_to_id, station_static_map, seq_len, "train")
    x_va, yreg_va, ycls_va, meta_va = _build_baseline_tabular(valid_df, feature_cols, x_scaler, y_scaler, static_scaler, station_to_id, station_static_map, seq_len, "valid")
    if len(x_va) < 20:
        print("[SHAP] too few validation tabular rows, skip daily SHAP.", flush=True)
        return
    f_names = daily_tabular_feature_names(feature_cols, static_dim)
    shp_dir = run_out / "shap_analysis"
    shp_dir.mkdir(parents=True, exist_ok=True)
    bg_n = min(256, len(x_tr))
    ev_n = min(200, len(x_va))
    x_bg = x_tr[:bg_n]
    x_ev = x_va[:ev_n]
    key_df = pd.DataFrame(meta_va[:ev_n], columns=["station", "phase", "date"])

    def _linear_explainer_expected_scalar(ev_raw: object, target_idx: int) -> float:
        if isinstance(ev_raw, (list, tuple)):
            return float(np.asarray(ev_raw[int(target_idx)], dtype=np.float64).ravel()[0])
        a = np.asarray(ev_raw, dtype=np.float64).ravel()
        if a.size == 0:
            return 0.0
        if a.size > int(target_idx):
            return float(a[int(target_idx)])
        return float(a[0])

    def _save_linear_shap(model, name: str, target_idx: int, target_name: str, *, save_detail: bool = False) -> None:
        expl = shap.LinearExplainer(model, x_bg)
        sv = expl.shap_values(x_ev)
        if isinstance(sv, list):
            sv_mat = sv[int(target_idx)]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv_mat = sv[:, :, int(target_idx)]
        else:
            sv_mat = sv
        imp = np.mean(np.abs(sv_mat), axis=0)
        pd.DataFrame({"feature": f_names, "mean_abs_shap": imp, "model": name, "target": target_name}).sort_values("mean_abs_shap", ascending=False).to_csv(
            shp_dir / f"shap_{name}_{target_name}.csv", index=False, encoding="utf-8-sig"
        )
        if save_detail:
            try:
                ev_scalar = _linear_explainer_expected_scalar(expl.expected_value, target_idx)
                np.savez_compressed(
                    shp_dir / f"shap_{name}_{target_name}_detail.npz",
                    shap_values=np.asarray(sv_mat, dtype=np.float32),
                    x=np.asarray(x_ev, dtype=np.float32),
                    expected_value=np.float32(ev_scalar),
                    feature_names=np.asarray(f_names, dtype=object),
                )
            except Exception as ex:
                print(f"[SHAP] detail npz skip {name} {target_name}: {ex}", flush=True)

    ridge_m = fit_models.get("ridge")
    if ridge_m is None:
        ridge_m = Ridge(alpha=1.0, random_state=seed)
        ridge_m.fit(x_tr, yreg_tr)
    if hasattr(ridge_m, "coef_"):
        _save_linear_shap(ridge_m, "ridge", 0, "idx_30", save_detail=True)
        _save_linear_shap(ridge_m, "ridge", 1, "idx_90", save_detail=True)

    rf_m = fit_models.get("random_forest")
    if rf_m is not None:
        try:
            expl = shap.TreeExplainer(rf_m)
            sv = expl.shap_values(x_ev)
            if isinstance(sv, list):
                for ti, tnm in enumerate(["idx_30", "idx_90"]):
                    imp = np.mean(np.abs(sv[ti]), axis=0)
                    pd.DataFrame({"feature": f_names, "mean_abs_shap": imp, "model": "random_forest", "target": tnm}).sort_values("mean_abs_shap", ascending=False).to_csv(
                        shp_dir / f"shap_random_forest_{tnm}.csv", index=False, encoding="utf-8-sig"
                    )
            else:
                if sv.ndim == 3:
                    for ti, tnm in enumerate(["idx_30", "idx_90"]):
                        imp = np.mean(np.abs(sv[:, :, ti]), axis=0)
                        pd.DataFrame({"feature": f_names, "mean_abs_shap": imp, "model": "random_forest", "target": tnm}).sort_values("mean_abs_shap", ascending=False).to_csv(
                            shp_dir / f"shap_random_forest_{tnm}.csv", index=False, encoding="utf-8-sig"
                        )
        except Exception as ex:
            print(f"[SHAP] random_forest TreeExplainer skipped: {ex}", flush=True)

    logi = LogisticRegression(max_iter=1200, random_state=seed)
    try:
        logi.fit(x_tr, ycls_tr.astype(np.int32))
        expl = shap.LinearExplainer(logi, x_bg)
        sv = expl.shap_values(x_ev)
        if isinstance(sv, list):
            sv_mat = sv[1] if len(sv) > 1 else sv[0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv_mat = sv[:, :, 1]
        else:
            sv_mat = sv
        imp = np.mean(np.abs(sv_mat), axis=0)
        pd.DataFrame({"feature": f_names, "mean_abs_shap": imp, "model": "logistic_flash", "target": CLS_TARGET_COL}).sort_values("mean_abs_shap", ascending=False).to_csv(
            shp_dir / "shap_logistic_flash_label.csv", index=False, encoding="utf-8-sig"
        )
        try:
            ev_raw = expl.expected_value
            if isinstance(ev_raw, (list, tuple)):
                ev_scalar = float(np.asarray(ev_raw[1], dtype=np.float64).ravel()[0])
            else:
                er = np.asarray(ev_raw, dtype=np.float64).ravel()
                ev_scalar = float(er[1] if er.size > 1 else er[0])
            np.savez_compressed(
                shp_dir / "shap_logistic_flash_label_detail.npz",
                shap_values=np.asarray(sv_mat, dtype=np.float32),
                x=np.asarray(x_ev, dtype=np.float32),
                expected_value=np.float32(ev_scalar),
                feature_names=np.asarray(f_names, dtype=object),
            )
        except Exception as ex_npz:
            print(f"[SHAP] logistic detail npz skip: {ex_npz}", flush=True)
    except Exception as ex:
        print(f"[SHAP] logistic flash skipped: {ex}", flush=True)

    for mdl, tcol in [
        ("tcn_daily_hybrid", "idx_30"),
        ("tcn_daily_hybrid", "idx_90"),
        ("gru_daily_hybrid", "idx_30"),
        ("gru_daily_hybrid", "idx_90"),
        ("fusion_daily_stacking", "idx_30"),
        ("fusion_daily_stacking", "idx_90"),
    ]:
        sub = one_df[(one_df["model"] == mdl) & (one_df["phase"] == "valid") & (one_df["target"] == tcol)]
        if len(sub) == 0:
            continue
        yhat = key_df.merge(sub[["station", "phase", "date", "y_pred"]], on=["station", "phase", "date"], how="left")["y_pred"].fillna(0.0).values.astype(np.float32)
        sur = Ridge(alpha=1.0, random_state=42)
        sur.fit(x_ev, yhat)
        expl = shap.LinearExplainer(sur, x_bg)
        sv = expl.shap_values(x_ev)
        imp = np.mean(np.abs(sv), axis=0)
        safe = mdl.replace(" ", "_")
        pd.DataFrame({"feature": f_names, "mean_abs_shap": imp, "model": f"{mdl}_surrogate", "target": tcol}).sort_values("mean_abs_shap", ascending=False).to_csv(
            shp_dir / f"shap_{safe}_surrogate_{tcol}.csv", index=False, encoding="utf-8-sig"
        )
        if mdl == "fusion_daily_stacking" and tcol in ("idx_30", "idx_90"):
            try:
                ev_scalar = float(np.asarray(expl.expected_value, dtype=np.float64).ravel()[0])
                np.savez_compressed(
                    shp_dir / f"shap_{safe}_surrogate_{tcol}_detail.npz",
                    shap_values=np.asarray(sv, dtype=np.float32),
                    x=np.asarray(x_ev, dtype=np.float32),
                    expected_value=np.float32(ev_scalar),
                    feature_names=np.asarray(f_names, dtype=object),
                )
            except Exception as ex_npz:
                print(f"[SHAP] fusion surrogate detail npz skip: {ex_npz}", flush=True)

    print(f"[SHAP] daily CSVs saved under {shp_dir} (use regenerate_daily_shap_and_ablation.plot_shap_overview_agg for PNG)", flush=True)


def train_joint_once(args, fold_idx: int, total_folds: int):
    d = build_daily_data_bundle(args, fold_idx, total_folds)
    run_out = d["run_out"]
    feature_cols = d["feature_cols"]
    train_df = d["train_df"]
    valid_df = d["valid_df"]
    test_df = d["test_df"]
    split_rows = d["split_rows"]
    score_df = d["score_df"]
    station_static_map = d["station_static_map"]
    x_scaler = d["x_scaler"]
    y_scaler = d["y_scaler"]
    static_scaler = d["static_scaler"]
    station_to_id = d["station_to_id"]

    print(f"[CLS] loss={args.cls_loss_type}, focal_gamma={args.focal_gamma:.2f}, pos_weight={args._cls_pos_weight:.4f}", flush=True)
    print(
        f"[REC] consistency_weight={args.recursive_consistency_weight:.3f}, prev_blend={args.recursive_prev_blend:.2f}, "
        f"unroll_steps={int(args.recursive_unroll_steps)}, unroll_decay={args.recursive_unroll_decay:.2f}",
        flush=True,
    )

    tr_samples = make_daily_samples(train_df, x_scaler, y_scaler, static_scaler, feature_cols, station_to_id, station_static_map, args.seq_len, "train")
    va_samples = make_daily_samples(valid_df, x_scaler, y_scaler, static_scaler, feature_cols, station_to_id, station_static_map, args.seq_len, "valid")
    te_samples = make_daily_samples(test_df, x_scaler, y_scaler, static_scaler, feature_cols, station_to_id, station_static_map, args.seq_len, "test")
    tr_loader = DataLoader(DailyMultiTaskSeqDataset(tr_samples), batch_size=args.batch_size, shuffle=True)
    tr_eval_loader = DataLoader(DailyMultiTaskSeqDataset(tr_samples), batch_size=args.batch_size, shuffle=False)
    va_loader = DataLoader(DailyMultiTaskSeqDataset(va_samples), batch_size=args.batch_size, shuffle=False)
    te_loader = DataLoader(DailyMultiTaskSeqDataset(te_samples), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_dim = len(next(iter(station_static_map.values())))
    _hybrid_kw = dict(
        model_dim=args.model_dim,
        n_heads=args.n_heads,
        tcn_layers=args.tcn_layers,
        gru_layers=args.gru_layers,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        gate_hidden_dim=args.gate_hidden_dim,
        fusion_mode=args.model_ablation_mode,
        fixed_gate_value=float(args.fixed_gate_value),
        station_emb_dim=int(args.station_emb_dim),
    )
    model_a = DailyHybridModel(
        len(feature_cols), len(args.stations), static_dim, deep_type="tcn_transformer", **_hybrid_kw
    ).to(device)
    model_b = DailyHybridModel(
        len(feature_cols), len(args.stations), static_dim, deep_type="gru_transformer", **_hybrid_kw
    ).to(device)
    if float(args.lambda_cmip_hist) <= 0.0 and float(args.lambda_cmip_scenario) <= 0.0:
        cmip_ctx = {"historical": [], "scenario": []}
    else:
        cmip_ctx = build_cmip_daily_contexts(
            stations=args.stations,
            feature_cols=feature_cols,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            static_scaler=static_scaler,
            station_to_id=station_to_id,
            station_static_map=station_static_map,
            seq_len=args.seq_len,
            modal_method=args.modal_method,
            device=device,
        )
    print(f"[CMIP-DAILY] historical_ctx={len(cmip_ctx.get('historical', []))}, scenario_ctx={len(cmip_ctx.get('scenario', []))}", flush=True)
    va_roll = build_rollout_inputs(valid_df, feature_cols, x_scaler, y_scaler, station_to_id, station_static_map, static_scaler, args.seq_len)
    te_roll = build_rollout_inputs(test_df, feature_cols, x_scaler, y_scaler, station_to_id, station_static_map, static_scaler, args.seq_len)

    total_hybrid_models = 2
    model_a, log_a = train_one_hybrid(
        "tcn_daily_hybrid",
        model_a,
        tr_loader,
        va_loader,
        va_roll,
        y_scaler,
        device,
        args,
        cmip_ctx,
        run_out,
        "tcn_daily_hybrid",
        model_order=1,
        total_models=total_hybrid_models,
    )
    model_b, log_b = train_one_hybrid(
        "gru_daily_hybrid",
        model_b,
        tr_loader,
        va_loader,
        va_roll,
        y_scaler,
        device,
        args,
        cmip_ctx,
        run_out,
        "gru_daily_hybrid",
        model_order=2,
        total_models=total_hybrid_models,
    )

    pred_tr_a = run_inference_hybrid(model_a, tr_eval_loader, y_scaler, device, "tcn_daily_hybrid")
    pred_va_a = run_inference_hybrid(model_a, va_loader, y_scaler, device, "tcn_daily_hybrid")
    pred_te_a = run_inference_hybrid(model_a, te_loader, y_scaler, device, "tcn_daily_hybrid")
    pred_tr_b = run_inference_hybrid(model_b, tr_eval_loader, y_scaler, device, "gru_daily_hybrid")
    pred_va_b = run_inference_hybrid(model_b, va_loader, y_scaler, device, "gru_daily_hybrid")
    pred_te_b = run_inference_hybrid(model_b, te_loader, y_scaler, device, "gru_daily_hybrid")
    coef_stack = fit_stack_reg(merge_reg_predictions(pred_va_a, pred_va_b))
    pred_fusion = pd.concat(
        [
            apply_stack(pred_tr_a, pred_tr_b, coef_stack, "fusion_daily_stacking"),
            apply_stack(pred_va_a, pred_va_b, coef_stack, "fusion_daily_stacking"),
            apply_stack(pred_te_a, pred_te_b, coef_stack, "fusion_daily_stacking"),
        ],
        ignore_index=True,
    )

    base_pred, fit_models, deep_pred = {}, {}, {}
    if not bool(args.skip_baselines):
        base_pred, fit_models = run_tabular_baselines(train_df, valid_df, test_df, feature_cols, x_scaler, y_scaler, static_scaler, station_to_id, station_static_map, args.seq_len, args.seed + fold_idx)
        for typ in ["cnn", "lstm", "rnn"]:
            mdl = SeqDailyBaseline(input_dim=len(feature_cols), model_type=typ, hidden_dim=args.deep_baseline_hidden).to(device)
            opt = torch.optim.AdamW(mdl.parameters(), lr=args.deep_baseline_lr, weight_decay=1e-4)
            loss_reg = nn.SmoothL1Loss(beta=0.5)
            loss_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args._cls_pos_weight], dtype=torch.float32, device=device))
            for ep in range(max(int(args.deep_baseline_epochs), 1)):
                mdl.train()
                for x_seq, y_reg, y_cls, _, _, prev_reg, _, _, _, _, _ in tr_loader:
                    reg, cls = mdl(x_seq.to(device), prev_reg.to(device))
                    loss = loss_reg(reg, y_reg.to(device)) + 0.8 * loss_cls(cls, y_cls.to(device))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                    opt.step()
            deep_pred[f"{typ}_baseline"] = pd.concat(
                [
                    run_inference_seq_baseline(mdl, tr_eval_loader, y_scaler, device, f"{typ}_baseline"),
                    run_inference_seq_baseline(mdl, va_loader, y_scaler, device, f"{typ}_baseline"),
                    run_inference_seq_baseline(mdl, te_loader, y_scaler, device, f"{typ}_baseline"),
                ],
                ignore_index=True,
            )
            fit_models[f"{typ}_baseline"] = mdl

    one_all = {"fusion_daily_stacking": pred_fusion, "tcn_daily_hybrid": pd.concat([pred_tr_a, pred_va_a, pred_te_a], ignore_index=True), "gru_daily_hybrid": pd.concat([pred_tr_b, pred_va_b, pred_te_b], ignore_index=True)}
    for k, v in base_pred.items():
        one_all[k] = v
    for k, v in deep_pred.items():
        one_all[k] = v
    one_df = pd.concat(one_all.values(), ignore_index=True)
    one_meta_pred, one_meta_w = build_meta_fusion(
        one_df,
        fit_phase="valid",
        candidate_models=list(one_all.keys()),
        model_name="fusion_meta_all_daily",
        seed=args.seed + fold_idx,
        n_trials=args.meta_trials,
    )
    if len(one_meta_pred) > 0:
        one_df = pd.concat([one_df, one_meta_pred], ignore_index=True)
    one_metrics = evaluate_daily_long(one_df, cls_threshold_by_model=None)
    one_metrics_per_station = evaluate_daily_long_per_station(one_df, cls_threshold_by_model=None)
    if str(getattr(args, "flash_threshold_mode", "fixed05")) == "valid_f1":
        obj = str(getattr(args, "flash_threshold_objective", "f1"))
        one_cls_thr = pooled_cls_thresholds_from_valid(one_df, "valid", objective=obj)
        if len(one_cls_thr) > 0:
            fn_thr = f"flash_cls_thresholds_valid_{obj}_one_step.csv"
            thr_out = pd.DataFrame(
                [{"model": k, f"threshold_valid_{obj}": float(v)} for k, v in sorted(one_cls_thr.items(), key=lambda kv: kv[0])]
            )
            thr_out.to_csv(run_out / fn_thr, index=False, encoding="utf-8-sig")
            print(f"[CLS] sidecar: flash thresholds on valid ({obj.upper()}), n_models={len(one_cls_thr)}", flush=True)
            one_tuned = evaluate_daily_long(one_df, cls_threshold_by_model=one_cls_thr)
            one_tuned_ps = evaluate_daily_long_per_station(one_df, cls_threshold_by_model=one_cls_thr)
            one_tuned.to_csv(
                run_out / f"metrics_daily_model_comparison_cls_thr_on_valid_sidecar_{obj}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if len(one_tuned_ps) > 0:
                one_tuned_ps.to_csv(
                    run_out / f"metrics_daily_model_comparison_per_station_cls_thr_on_valid_sidecar_{obj}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    if bool(getattr(args, "daily_shap", False)):
        sd = int(len(next(iter(station_static_map.values()))))
        try:
            run_daily_shap_analysis(
                run_out,
                feature_cols,
                sd,
                valid_df,
                train_df,
                x_scaler,
                y_scaler,
                static_scaler,
                station_to_id,
                station_static_map,
                int(args.seq_len),
                fit_models,
                one_df,
                seed=int(args.seed + fold_idx),
            )
        except Exception as ex:
            print(f"[SHAP] daily analysis failed: {ex}", flush=True)
        else:
            try:
                from regenerate_daily_shap_and_ablation import plot_shap_overview_agg

                plot_shap_overview_agg(run_out / "shap_analysis")
            except Exception as ex2:
                print(f"[SHAP] overview PNG skipped: {ex2}", flush=True)

    rec_a_va = recursive_rollout_hybrid(model_a, va_roll, y_scaler, device, "tcn_daily_hybrid_recursive", "valid_recursive", recursive_prev_blend=args.recursive_prev_blend)
    rec_b_va = recursive_rollout_hybrid(model_b, va_roll, y_scaler, device, "gru_daily_hybrid_recursive", "valid_recursive", recursive_prev_blend=args.recursive_prev_blend)
    rec_a_te = recursive_rollout_hybrid(model_a, te_roll, y_scaler, device, "tcn_daily_hybrid_recursive", "test_recursive", recursive_prev_blend=args.recursive_prev_blend)
    rec_b_te = recursive_rollout_hybrid(model_b, te_roll, y_scaler, device, "gru_daily_hybrid_recursive", "test_recursive", recursive_prev_blend=args.recursive_prev_blend)
    rec_df = pd.concat(
        [
            rec_a_va,
            rec_b_va,
            apply_stack(rec_a_va.rename(columns={"model": "tmp_a"}), rec_b_va.rename(columns={"model": "tmp_b"}), coef_stack, "fusion_daily_stacking_recursive"),
            rec_a_te,
            rec_b_te,
            apply_stack(rec_a_te.rename(columns={"model": "tmp_a"}), rec_b_te.rename(columns={"model": "tmp_b"}), coef_stack, "fusion_daily_stacking_recursive"),
        ],
        ignore_index=True,
    )
    if not bool(args.skip_baselines):
        rec_df = pd.concat(
            [
                rec_df,
                run_recursive_tabular_baselines(va_roll, fit_models, y_scaler, phase_name="valid_recursive"),
                run_recursive_tabular_baselines(te_roll, fit_models, y_scaler, phase_name="test_recursive"),
            ],
            ignore_index=True,
        )
        for typ in ["cnn", "lstm", "rnn"]:
            if f"{typ}_baseline" in fit_models:
                rec_df = pd.concat(
                    [
                        rec_df,
                        recursive_rollout_seq_baseline(fit_models[f"{typ}_baseline"], va_roll, y_scaler, device, f"{typ}_baseline_recursive", "valid_recursive", recursive_prev_blend=args.recursive_prev_blend),
                        recursive_rollout_seq_baseline(fit_models[f"{typ}_baseline"], te_roll, y_scaler, device, f"{typ}_baseline_recursive", "test_recursive", recursive_prev_blend=args.recursive_prev_blend),
                    ],
                    ignore_index=True,
                )
    rec_meta_pred, rec_meta_w = build_meta_fusion(
        rec_df,
        fit_phase="valid_recursive",
        candidate_models=sorted(rec_df["model"].unique().tolist()),
        model_name="fusion_meta_all_daily_recursive",
        seed=args.seed + fold_idx + 99,
        n_trials=max(1000, int(args.meta_trials * 0.8)),
    )
    if len(rec_meta_pred) > 0:
        rec_df = pd.concat([rec_df, rec_meta_pred], ignore_index=True)
    rec_cal_coef = pd.DataFrame()
    if not bool(args.disable_recursive_calibration):
        cand = [m for m in sorted(rec_df["model"].unique().tolist()) if ("hybrid_recursive" in m or "fusion_daily_stacking_recursive" in m or "fusion_meta_all_daily_recursive" in m)]
        rec_cal_pred, rec_cal_coef = build_recursive_reg_calibration(rec_df, base_models=cand, fit_phase="valid_recursive")
        if len(rec_cal_pred) > 0:
            rec_df = pd.concat([rec_df, rec_cal_pred], ignore_index=True)
    taskwise_info = pd.DataFrame()
    taskwise_pred, taskwise_info = build_taskwise_recursive_fusion(rec_df, fit_phase="valid_recursive", model_name="fusion_taskwise_recursive")
    if len(taskwise_pred) > 0:
        rec_df = pd.concat([rec_df, taskwise_pred], ignore_index=True)
    blend_info = pd.DataFrame()
    blend_pred, blend_info = build_recursive_targetwise_blend(rec_df, fit_phase="valid_recursive", model_name="fusion_targetwise_blend_recursive")
    if len(blend_pred) > 0:
        rec_df = pd.concat([rec_df, blend_pred], ignore_index=True)
    rec_metrics = evaluate_daily_long(rec_df, cls_threshold_by_model=None)
    rec_metrics_per_station = evaluate_daily_long_per_station(rec_df, cls_threshold_by_model=None)
    if str(getattr(args, "flash_threshold_mode", "fixed05")) == "valid_f1":
        obj = str(getattr(args, "flash_threshold_objective", "f1"))
        rec_cls_thr = pooled_cls_thresholds_from_valid(rec_df, "valid_recursive", objective=obj)
        if len(rec_cls_thr) > 0:
            thr_rec = pd.DataFrame(
                [{"model": k, f"threshold_valid_{obj}": float(v)} for k, v in sorted(rec_cls_thr.items(), key=lambda kv: kv[0])]
            )
            thr_rec.to_csv(run_out / f"flash_cls_thresholds_valid_{obj}_recursive.csv", index=False, encoding="utf-8-sig")
            print(f"[CLS] sidecar: recursive flash thresholds on valid_recursive ({obj.upper()}), n_models={len(rec_cls_thr)}", flush=True)
            rec_tuned = evaluate_daily_long(rec_df, cls_threshold_by_model=rec_cls_thr)
            rec_tuned_ps = evaluate_daily_long_per_station(rec_df, cls_threshold_by_model=rec_cls_thr)
            rec_tuned.to_csv(
                run_out / f"metrics_daily_model_comparison_recursive_cls_thr_on_valid_sidecar_{obj}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if len(rec_tuned_ps) > 0:
                rec_tuned_ps.to_csv(
                    run_out / f"metrics_daily_model_comparison_recursive_per_station_cls_thr_on_valid_sidecar_{obj}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    score_df.to_csv(run_out / "modal_feature_scores_all_stations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"selected_features": feature_cols}).to_csv(run_out / "selected_feature_list.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(split_rows).to_csv(run_out / "split_time_ranges_daily.csv", index=False, encoding="utf-8-sig")
    pd.concat([log_a, log_b], ignore_index=True).to_csv(run_out / "training_log_daily.csv", index=False, encoding="utf-8-sig")
    one_df.to_csv(run_out / "predictions_daily_all_models.csv", index=False, encoding="utf-8-sig")
    rec_df.to_csv(run_out / "predictions_daily_recursive_all_models.csv", index=False, encoding="utf-8-sig")
    one_metrics.to_csv(run_out / "metrics_daily_model_comparison.csv", index=False, encoding="utf-8-sig")
    rec_metrics.to_csv(run_out / "metrics_daily_model_comparison_recursive.csv", index=False, encoding="utf-8-sig")
    write_metrics_aggregation_notes(run_out)
    if len(one_metrics_per_station) > 0:
        one_metrics_per_station.to_csv(run_out / "metrics_daily_model_comparison_per_station.csv", index=False, encoding="utf-8-sig")
    if len(rec_metrics_per_station) > 0:
        rec_metrics_per_station.to_csv(
            run_out / "metrics_daily_model_comparison_recursive_per_station.csv", index=False, encoding="utf-8-sig"
        )
    save_per_model_results(
        one_df,
        rec_df,
        one_metrics,
        rec_metrics,
        run_out,
        one_metrics_per_station=one_metrics_per_station,
        rec_metrics_per_station=rec_metrics_per_station,
    )
    pd.DataFrame(coef_stack).to_csv(run_out / "stacking_weights_daily.csv", index=False, encoding="utf-8-sig")
    if len(one_meta_w) > 0:
        one_meta_w.to_csv(run_out / "meta_weights_one_step_daily.csv", index=False, encoding="utf-8-sig")
    if len(rec_meta_w) > 0:
        rec_meta_w.to_csv(run_out / "meta_weights_recursive_daily.csv", index=False, encoding="utf-8-sig")
    if len(rec_cal_coef) > 0:
        rec_cal_coef.to_csv(run_out / "recursive_reg_calibration_coef.csv", index=False, encoding="utf-8-sig")
    if len(taskwise_info) > 0:
        taskwise_info.to_csv(run_out / "recursive_taskwise_fusion_info.csv", index=False, encoding="utf-8-sig")
    if len(blend_info) > 0:
        blend_info.to_csv(run_out / "recursive_targetwise_blend_info.csv", index=False, encoding="utf-8-sig")

    reg_show = ["model", "rmse", "mae", "r2", "nse", "kge", "pearson_r", "pbias", "willmott_d"]
    reg_show = [c for c in reg_show if c in one_metrics.columns]
    test_reg = one_metrics[(one_metrics["phase"] == "test") & (one_metrics["target"] == "reg_macro")][reg_show].sort_values("rmse")
    cls_show = ["model", "f1", "mcc", "brier", "csi", "far", "auc", "accuracy", "specificity"]
    cls_show = [c for c in cls_show if c in one_metrics.columns]
    test_cls = one_metrics[(one_metrics["phase"] == "test") & (one_metrics["target"] == CLS_TARGET_COL)][cls_show].sort_values("f1", ascending=False)
    print("[DAILY] one-step test reg macro", flush=True)
    print(test_reg, flush=True)
    print("[DAILY] one-step test flash cls", flush=True)
    print(test_cls.head(12), flush=True)


def write_seed_bundle_aggregate(base_run_tag: str, seeds: List[int], n_stations: int) -> None:
    summ_one: List[pd.DataFrame] = []
    summ_rec: List[pd.DataFrame] = []
    for sd in seeds:
        suf = f"{base_run_tag}_seed{sd}" if base_run_tag else f"seed{sd}"
        run_dir = OUT_DIR / f"daily_multitask_joint_{n_stations}stations_{suf}"
        p1 = run_dir / "metrics_daily_model_comparison.csv"
        if not p1.exists():
            continue
        d1 = pd.read_csv(p1)
        d1["run_seed"] = int(sd)
        summ_one.append(d1)
        p2 = run_dir / "metrics_daily_model_comparison_recursive.csv"
        if p2.exists():
            d2 = pd.read_csv(p2)
            d2["run_seed"] = int(sd)
            summ_rec.append(d2)
    slug = base_run_tag.replace("/", "_") if base_run_tag else "run"
    if summ_one:
        all_one = pd.concat(summ_one, ignore_index=True)
        all_one.to_csv(OUT_DIR / f"daily_journal_seed_bundle_{slug}_one_step_all.csv", index=False, encoding="utf-8-sig")
        sub = all_one[(all_one["phase"] == "test") & (all_one["target"] == "reg_macro")].copy()
        agg_cols = [c for c in ["rmse", "mae", "r2", "nse", "kge", "pearson_r", "pbias", "willmott_d"] if c in sub.columns]
        if len(sub) > 0 and agg_cols:
            rows_agg = []
            for mdl, gx in sub.groupby("model"):
                row = {"model": str(mdl)}
                for c in agg_cols:
                    row[f"{c}_mean"] = float(np.mean(gx[c].values.astype(np.float64)))
                    row[f"{c}_std"] = float(np.std(gx[c].values.astype(np.float64), ddof=0))
                rows_agg.append(row)
            pd.DataFrame(rows_agg).to_csv(OUT_DIR / f"daily_journal_seed_bundle_{slug}_test_reg_macro_meanstd.csv", index=False, encoding="utf-8-sig")
    if summ_rec:
        all_rec = pd.concat(summ_rec, ignore_index=True)
        all_rec.to_csv(OUT_DIR / f"daily_journal_seed_bundle_{slug}_recursive_all.csv", index=False, encoding="utf-8-sig")
        sub = all_rec[(all_rec["phase"] == "test_recursive") & (all_rec["target"] == "reg_macro")].copy()
        agg_cols = [c for c in ["rmse", "mae", "r2", "nse", "kge", "pearson_r", "pbias", "willmott_d"] if c in sub.columns]
        if len(sub) > 0 and agg_cols:
            rows_agg = []
            for mdl, gx in sub.groupby("model"):
                row = {"model": str(mdl)}
                for c in agg_cols:
                    row[f"{c}_mean"] = float(np.mean(gx[c].values.astype(np.float64)))
                    row[f"{c}_std"] = float(np.std(gx[c].values.astype(np.float64), ddof=0))
                rows_agg.append(row)
            pd.DataFrame(rows_agg).to_csv(OUT_DIR / f"daily_journal_seed_bundle_{slug}_test_recursive_reg_macro_meanstd.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] multi-seed aggregate written for slug={slug}", flush=True)


def main():
    configure_console_output()
    args = parse_args()
    args.modal_method = resolve_modal_method(args.modal_method)
    args.stations = available_stations() if args.stations.lower() == "all" else [s.strip() for s in args.stations.split(",") if s.strip()]
    base_run_tag = str(args.run_tag).strip()
    rep = max(1, int(getattr(args, "replicate_seeds", 1)))
    seeds = [int(args.seed) + k for k in range(rep)]
    nst = len(args.stations)
    print(f"[INFO] DAILY targets={REG_TARGET_COLS}+{CLS_TARGET_COL}, station_count={nst}, epochs={args.epochs}, replicate_seeds={rep}", flush=True)
    print("[INFO] Modal feature scoring uses TRAIN split only (no test/valid target leakage).", flush=True)
    for sd in seeds:
        args.seed = int(sd)
        if rep > 1:
            args.run_tag = f"{base_run_tag}_seed{sd}" if base_run_tag else f"seed{sd}"
        else:
            args.run_tag = base_run_tag
        set_seed(args.seed)
        print(f"[INFO] run_seed={args.seed} run_tag={args.run_tag}", flush=True)
        for fidx in range(max(1, int(args.rolling_folds))):
            train_joint_once(args, fold_idx=fidx, total_folds=max(1, int(args.rolling_folds)))
    if rep > 1:
        write_seed_bundle_aggregate(base_run_tag, seeds, nst)


if __name__ == "__main__":
    main()

