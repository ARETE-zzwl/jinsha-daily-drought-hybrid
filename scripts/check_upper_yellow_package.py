"""Validate the normalized 16-station Upper Yellow River release data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "date",
    "pr",
    "tmean",
    "tmax",
    "tmin",
    "et0",
    "wind",
    "rad_net",
    "rad_down",
    "runoff",
]
EXPECTED_START = pd.Timestamp("2010-01-01")
EXPECTED_END = pd.Timestamp("2024-03-31")
EXPECTED_ROWS = (EXPECTED_END - EXPECTED_START).days + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "external" / "upper_yellow_station_daily",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data" / "external" / "upper_yellow_station_metadata.csv",
    )
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*_daily.csv"))
    if len(files) != 16:
        raise RuntimeError(f"Expected 16 station files, found {len(files)}")
    metadata = pd.read_csv(args.metadata)
    if len(metadata) != 16:
        raise RuntimeError(f"Expected 16 metadata rows, found {len(metadata)}")

    total_rows = 0
    for path in files:
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED if column not in frame.columns]
        if missing:
            raise RuntimeError(f"{path.name} is missing columns: {missing}")
        dates = pd.to_datetime(frame["date"], errors="raise")
        if len(frame) != EXPECTED_ROWS:
            raise RuntimeError(f"{path.name} has {len(frame)} rows, expected {EXPECTED_ROWS}")
        if dates.iloc[0] != EXPECTED_START or dates.iloc[-1] != EXPECTED_END:
            raise RuntimeError(
                f"{path.name} spans {dates.iloc[0].date()} to {dates.iloc[-1].date()}"
            )
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise RuntimeError(f"{path.name} has duplicate or unordered dates")
        if frame[REQUIRED[1:]].isna().any().any():
            raise RuntimeError(f"{path.name} contains missing required values")
        total_rows += len(frame)

    print(f"Upper Yellow River package check passed: 16 stations, {total_rows} station-days.")
    print("Available period: 2010-01-01 to 2024-03-31.")
    print("Formal evaluation period: 2022-02-09 to 2024-03-31.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
