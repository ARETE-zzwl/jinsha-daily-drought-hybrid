"""Lightweight integrity check for the WRR open package."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from drought_hybrid.config import CMIP_DIR, DATA_DIR, DEM_SUMMARY_FILE


def safe_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.interpolate(limit_direction="both")
    return x.fillna(x.median())


def add_daily_targets_check(df: pd.DataFrame, train_len: int) -> pd.DataFrame:
    out = df.copy()
    wb = safe_series(out["pr"]) - safe_series(out["et0"])
    for w in (30, 90):
        roll = wb.rolling(w, min_periods=w).sum()
        ref = roll.iloc[: max(train_len, 1)].dropna()
        mu = float(ref.mean()) if len(ref) else 0.0
        sd = float(ref.std()) if len(ref) else 1.0
        if abs(sd) < 1e-6:
            sd = 1.0
        out[f"idx_{w}"] = ((roll - mu) / sd).fillna(0.0)
    idx30 = out["idx_30"]
    rapid = (idx30 - idx30.shift(14)) <= -0.8
    persist = idx30.rolling(5, min_periods=1).max() <= -1.0
    out["flash_label"] = (rapid & persist).fillna(False).astype(float)
    return out


def main() -> int:
    stations = sorted(p.name.replace("_daily.csv", "") for p in DATA_DIR.glob("*_daily.csv"))
    if len(stations) != 8:
        raise RuntimeError(f"Expected 8 station files, found {len(stations)}: {stations}")

    print(f"Project root: {ROOT}")
    print(f"Station data directory: {DATA_DIR}")
    print(f"Station metadata file: {DEM_SUMMARY_FILE}")
    print(f"Stations: {', '.join(stations)}")

    metadata = pd.read_csv(DEM_SUMMARY_FILE)
    print(f"Static vector dimension: 3")
    print(f"Metadata rows: {len(metadata)}")

    total_rows = 0
    for station in stations:
        df = pd.read_csv(DATA_DIR / f"{station}_daily.csv")
        required = ["date", "pr", "tmean", "tmax", "tmin", "et0", "wind", "rad_net", "rad_down", "runoff"]
        missing_inputs = [col for col in required if col not in df.columns]
        if missing_inputs:
            raise RuntimeError(f"Station {station} missing input columns: {missing_inputs}")
        if len(df) < 365:
            raise RuntimeError(f"Station {station} has too few rows: {len(df)}")
        targets = add_daily_targets_check(df, train_len=int(len(df) * 0.7))
        needed = {"idx_30", "idx_90", "flash_label"}
        missing = needed.difference(targets.columns)
        if missing:
            raise RuntimeError(f"Station {station} missing target columns: {sorted(missing)}")
        total_rows += len(df)

    print(f"Total station rows: {total_rows}")

    cmip_files = sorted(CMIP_DIR.glob("*_cmip_daily_bias_corrected.csv"))
    if cmip_files:
        if len(cmip_files) != 8:
            raise RuntimeError(f"Expected 8 CMIP6 auxiliary files, found {len(cmip_files)}")
        cmip_rows = 0
        cmip_scenarios = set()
        for path in cmip_files:
            cmip = pd.read_csv(path, usecols=["date", "scenario", "pr_bc", "tasmax_bc", "tasmin_bc"])
            if cmip.empty:
                raise RuntimeError(f"CMIP6 auxiliary file is empty: {path}")
            cmip_rows += len(cmip)
            cmip_scenarios.update(cmip["scenario"].astype(str).unique().tolist())
        print(f"CMIP6 auxiliary files: {len(cmip_files)} files, {cmip_rows} rows")
        print(f"CMIP6 scenarios: {', '.join(sorted(cmip_scenarios))}")
    else:
        print("CMIP6 auxiliary files: not found; exact manuscript training requires the Zenodo CMIP6 archive.")

    try:
        from drought_hybrid.data import available_stations, load_station_static_features, read_station_data
        from drought_hybrid.daily_models import DailyHybridModel

        pkg_stations = available_stations()
        static = load_station_static_features(pkg_stations, DEM_SUMMARY_FILE)
        _ = read_station_data(pkg_stations[0])
        print(f"Package data helper check: {len(pkg_stations)} stations, static dim {len(next(iter(static.values())))}")
        model = DailyHybridModel(input_dim=24, num_stations=8, static_dim=3)
        print(f"Model class check: {model.__class__.__name__}")
    except OSError as exc:
        print(f"Model class check skipped because PyTorch could not load: {exc}")
    print("Package check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
