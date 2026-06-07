from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR, MODE_NAMES

try:
    from PyEMD import EEMD

    EEMD_AVAILABLE = True
except Exception:
    EEMD_AVAILABLE = False

try:
    from vmdpy import VMD

    VMD_AVAILABLE = True
except Exception:
    VMD_AVAILABLE = False


def safe_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.interpolate(limit_direction="both")
    return x.fillna(x.median())


def available_stations() -> List[str]:
    files = sorted(DATA_DIR.glob("*_daily.csv"))
    return [f.name.replace("_daily.csv", "") for f in files]


def static_fallback_row(station_static_map: Dict[str, np.ndarray], station: str) -> np.ndarray:
    if station in station_static_map:
        return station_static_map[station].astype(np.float32).reshape(1, -1)
    proto = next(iter(station_static_map.values()))
    return np.zeros_like(proto, dtype=np.float32).reshape(1, -1)


def load_station_static_features(
    stations: List[str],
    dem_summary_file: Path,
    static_mode: str = "default",
) -> Dict[str, np.ndarray]:
    """Load station static vectors from the public metadata table.

    static_mode:
      - default/moran_ref: [elevation_m, latitude, longitude]
      - extended_dem: [elevation_m, latitude, longitude, log1p(elevation_m), elevation_m/max_elevation]
    """
    mode = str(static_mode or "default").lower().strip()
    dim = 5 if mode == "extended_dem" else 3
    static = {s: np.zeros(dim, dtype=np.float32) for s in stations}
    if not dem_summary_file.exists():
        return static

    dem = pd.read_csv(dem_summary_file)
    station_col = "station_slug" if "station_slug" in dem.columns else "station"
    need = [station_col, "elevation_m", "latitude", "longitude"]
    if not all(c in dem.columns for c in need):
        return static

    elev_by = {str(row[station_col]): float(row["elevation_m"]) for _, row in dem.iterrows()}
    max_e = max([elev_by.get(s, 0.0) for s in stations] + [1.0])

    for _, row in dem.iterrows():
        st = str(row[station_col])
        if st not in stations:
            continue
        elev = float(row["elevation_m"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        if mode == "extended_dem":
            static[st] = np.array(
                [elev, lat, lon, float(np.log1p(max(elev, 0.0))), elev / float(max_e)],
                dtype=np.float32,
            )
        else:
            static[st] = np.array([elev, lat, lon], dtype=np.float32)
    return static


def read_station_data(station_name: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{station_name}_daily.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Station file not found: {file_path}")
    df = pd.read_csv(file_path)
    required = ["date", "pr", "tmean", "tmax", "tmin", "et0", "wind", "rad_net", "rad_down", "runoff"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{file_path} is missing required column: {col}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "pr": safe_series(df["pr"]),
            "tmean": safe_series(df["tmean"]),
            "tmax": safe_series(df["tmax"]),
            "tmin": safe_series(df["tmin"]),
            "et0": safe_series(df["et0"]),
            "wind": safe_series(df["wind"]),
            "rad_net": safe_series(df["rad_net"]),
            "rad_down": safe_series(df["rad_down"]),
            "runoff": safe_series(df["runoff"]),
        }
    )
    return out.sort_values("date").dropna().reset_index(drop=True)


def moving_modal_decompose(series: pd.Series, short_win: int = 7, long_win: int = 30) -> Dict[str, pd.Series]:
    s = safe_series(series)
    low = s.rolling(long_win, min_periods=1).mean()
    short = s.rolling(short_win, min_periods=1).mean()
    return {"low": low, "mid": short - low, "high": s - short}


def eemd_modal_decompose(series: pd.Series, seed: int = 42) -> Dict[str, pd.Series]:
    if not EEMD_AVAILABLE:
        s = safe_series(series).values.astype(np.float64)
        rng = np.random.default_rng(seed)
        hs, ms, ls = [], [], []
        base_std = np.std(s) + 1e-8
        for _ in range(20):
            noise = rng.normal(0.0, 0.03 * base_std, size=len(s))
            modes = moving_modal_decompose(pd.Series(s + noise, index=series.index))
            hs.append(modes["high"].values)
            ms.append(modes["mid"].values)
            ls.append(modes["low"].values)
        return {
            "low": pd.Series(np.mean(np.array(ls), axis=0), index=series.index),
            "mid": pd.Series(np.mean(np.array(ms), axis=0), index=series.index),
            "high": pd.Series(np.mean(np.array(hs), axis=0), index=series.index),
        }

    s = safe_series(series).values.astype(np.float64)
    eemd = EEMD(trials=30, noise_width=0.03)
    np.random.seed(seed)
    imfs = eemd.eemd(s)
    if imfs.shape[0] < 3:
        return moving_modal_decompose(pd.Series(s, index=series.index))
    high = pd.Series(imfs[0], index=series.index)
    mid = pd.Series(np.mean(imfs[1:-1], axis=0), index=series.index) if imfs.shape[0] > 2 else pd.Series(imfs[1], index=series.index)
    low = pd.Series(imfs[-1], index=series.index)
    return {"low": low, "mid": mid, "high": high}


def vmd_modal_decompose(series: pd.Series) -> Dict[str, pd.Series]:
    if not VMD_AVAILABLE:
        return moving_modal_decompose(series)
    s = safe_series(series).values.astype(np.float64)
    u, _, _ = VMD(s, alpha=2000, tau=0.0, K=3, DC=0, init=1, tol=1e-7)
    return {"low": pd.Series(u[0], index=series.index), "mid": pd.Series(u[1], index=series.index), "high": pd.Series(u[2], index=series.index)}


def decompose(series: pd.Series, method: str) -> Dict[str, pd.Series]:
    method = str(method).lower()
    if method == "eemd":
        return eemd_modal_decompose(series)
    if method == "vmd":
        return vmd_modal_decompose(series)
    return moving_modal_decompose(series)


def resolve_modal_method(method: str) -> str:
    method = str(method).lower()
    if method == "eemd" and not EEMD_AVAILABLE:
        print("[INFO] PyEMD is unavailable; using the built-in EEMD approximation.")
        return "eemd"
    if method == "vmd" and not VMD_AVAILABLE:
        print("[INFO] vmdpy is unavailable; falling back to moving-average decomposition.")
        return "moving"
    return method


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    fold_idx: int = 0,
    total_folds: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if total_folds <= 1:
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        return df.iloc[:n_train].copy(), df.iloc[n_train : n_train + n_valid].copy(), df.iloc[n_train + n_valid :].copy()

    min_train = max(int(n * 0.5), 1)
    tail = max(n - min_train, 1)
    win = max(tail // (total_folds + 2), 1)
    step = max(tail // (total_folds + 1), 1)
    train_end = min(min_train + fold_idx * step, n - 2 * win)
    valid_end = train_end + win
    test_end = min(valid_end + win, n)
    if train_end <= 0 or valid_end <= train_end or test_end <= valid_end:
        return df.iloc[:0].copy(), df.iloc[:0].copy(), df.iloc[:0].copy()
    return df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy(), df.iloc[valid_end:test_end].copy()
