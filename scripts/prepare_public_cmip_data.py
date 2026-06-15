"""Prepare station-matched CMIP6 auxiliary CSVs for the public archive.

This helper renames the working-project station files to the ASCII station
slugs expected by the open training code. The resulting files are intended for
Zenodo rather than direct GitHub storage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STATIONS = {
    "华弹": "huatan",
    "屏山": "pingshan",
    "岗拖": "gangtuo",
    "巴塘": "batang",
    "攀枝花": "panzhihua",
    "石鼓": "shigu",
    "金江街": "jinjiangjie",
    "阿海": "ahai",
}

KEEP_COLUMNS = [
    "station",
    "date",
    "scenario",
    "pr",
    "tasmax",
    "tasmin",
    "tas",
    "pr_bc",
    "tasmax_bc",
    "tasmin_bc",
    "tas_bc",
]


def prepare_cmip_files(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for station_name, slug in STATIONS.items():
        source_file = source_dir / f"{station_name}_cmip_daily_bias_corrected.csv"
        if not source_file.exists():
            raise FileNotFoundError(source_file)
        df = pd.read_csv(source_file)
        missing = [col for col in KEEP_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{source_file} is missing required columns: {missing}")
        df = df[KEEP_COLUMNS].copy()
        df["station_slug"] = slug
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        ordered = ["station_slug"] + KEEP_COLUMNS
        df[ordered].to_csv(output_dir / f"{slug}_cmip_daily_bias_corrected.csv", index=False, encoding="utf-8")

    grid_file = source_dir / "all_station_grid_match_info.csv"
    if grid_file.exists():
        pd.read_csv(grid_file).to_csv(output_dir / grid_file.name, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public CMIP6 auxiliary station CSVs.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_cmip_files(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
