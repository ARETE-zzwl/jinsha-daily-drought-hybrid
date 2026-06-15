"""Prepare publication-ready station CSVs from working-project station files.

This helper is included for transparency. The open repository stores the
processed station files produced by this script, so reviewers normally do not
need to run it unless they have access to the original working-project data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STATIONS = {
    "华弹": ("huatan", "Huatan"),
    "屏山": ("pingshan", "Pingshan"),
    "岗拖": ("gangtuo", "Gangtuo"),
    "巴塘": ("batang", "Batang"),
    "攀枝花": ("panzhihua", "Panzhihua"),
    "石鼓": ("shigu", "Shigu"),
    "金江街": ("jinjiangjie", "Jinjiangjie"),
    "阿海": ("ahai", "Ahai"),
}


COLUMN_MAP = {
    "日期(UTC)": "date",
    "降水量(mm)": "pr",
    "平均气温(℃)": "tmean",
    "最高气温2m(℃)": "tmax",
    "最低气温2m(℃)": "tmin",
    "ET0(mm/day)": "et0",
    "平均风速(m/s)": "wind",
    "太阳辐射净强度(net,J/m2)": "rad_net",
    "太阳辐射总强度(down,J/m2)": "rad_down",
    "径流量(m3/s)": "runoff",
    "经度(lon)": "longitude",
    "纬度(lat)": "latitude",
    "年份": "year",
    "地面气压(hPa)": "surface_pressure_hpa",
    "露点温度(℃)": "dewpoint",
    "经向风速(V,m/s)": "wind_v",
    "纬向风速(U,m/s)": "wind_u",
}


ORDERED_COLUMNS = [
    "date",
    "station_slug",
    "station_name",
    "station_en",
    "pr",
    "tmean",
    "tmax",
    "tmin",
    "et0",
    "wind",
    "rad_net",
    "rad_down",
    "runoff",
    "longitude",
    "latitude",
    "year",
    "surface_pressure_hpa",
    "dewpoint",
    "wind_v",
    "wind_u",
]


def prepare_station_files(source_dir: Path, metadata_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_file)
    meta_rows = []

    for station_name, (slug, station_en) in STATIONS.items():
        source_file = source_dir / f"{station_name}_merged_data.csv"
        if not source_file.exists():
            raise FileNotFoundError(source_file)

        df = pd.read_csv(source_file).rename(columns=COLUMN_MAP)
        missing = [col for col in COLUMN_MAP.values() if col not in df.columns]
        if missing:
            raise ValueError(f"{source_file} is missing mapped columns: {missing}")

        out = df[list(COLUMN_MAP.values())].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["station_slug"] = slug
        out["station_name"] = station_name
        out["station_en"] = station_en
        out = out[ORDERED_COLUMNS]
        out.to_csv(output_dir / f"{slug}_daily.csv", index=False, encoding="utf-8")

        meta = metadata[metadata["station"] == station_name]
        if len(meta) == 0:
            elev = float("nan")
            lon = float(out["longitude"].iloc[0])
            lat = float(out["latitude"].iloc[0])
        else:
            elev = float(meta["elevation_m"].iloc[0])
            lon = float(meta["longitude"].iloc[0])
            lat = float(meta["latitude"].iloc[0])
        meta_rows.append(
            {
                "station_slug": slug,
                "station_name": station_name,
                "station_en": station_en,
                "longitude": lon,
                "latitude": lat,
                "elevation_m": elev,
            }
        )

    pd.DataFrame(meta_rows).to_csv(output_dir.parent / "station_metadata.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean station daily CSVs for the WRR open package.")
    parser.add_argument("--source-dir", type=Path, default=Path("DATA/merged_station_data2"))
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("output/dem_supplement_experiment_v2/station_elevation_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wrr_submission_open_project/data/processed/station_daily"),
    )
    args = parser.parse_args()
    prepare_station_files(args.source_dir, args.metadata_file, args.output_dir)


if __name__ == "__main__":
    main()
