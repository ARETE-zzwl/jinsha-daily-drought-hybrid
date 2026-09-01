"""Build normalized Upper Yellow River data and artifact archives for Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import pandas as pd


VERSION = "v1.1.0-wrr-revision"
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
}
TRANSFER_FILES = [
    "best_checkpoint_tcn_daily_hybrid.pt",
    "best_checkpoint_gru_daily_hybrid.pt",
    "selected_feature_list.csv",
    "stacking_weights_daily.csv",
    "jinsha_zero_shot_preprocessors.npz",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256")
    rows = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def zip_tree(staging: Path, archive: Path) -> None:
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=path.relative_to(staging).as_posix())


def build_data_archive(source: Path, metadata_source: Path, staging: Path) -> None:
    daily_dir = staging / "data" / "external" / "upper_yellow_station_daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    station_names = []
    for path in sorted(source.glob("*_merged_data.csv")):
        station = path.name.removesuffix("_merged_data.csv")
        if station == "下河沿":
            raise ValueError("Xiaheyan must not be present in the verified 16-station source directory")
        frame = pd.read_csv(path)
        missing = [column for column in COLUMN_MAP if column not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing source columns: {missing}")
        normalized = frame[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
        normalized.to_csv(daily_dir / f"{station}_daily.csv", index=False, encoding="utf-8-sig")
        station_names.append(station)
    if len(station_names) != 16:
        raise ValueError(f"Expected 16 verified station files, found {len(station_names)}")

    metadata = pd.read_csv(metadata_source)
    metadata = metadata[metadata["station"].astype(str).isin(station_names)].copy()
    if len(metadata) != 16:
        raise ValueError(f"Expected 16 matching metadata rows, found {len(metadata)}")
    metadata.insert(0, "station_slug", metadata["station"].astype(str))
    metadata.rename(columns={"station": "station_name"}, inplace=True)
    metadata.to_csv(
        staging / "data" / "external" / "upper_yellow_station_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (staging / "UPPER_YELLOW_DATA_README.md").write_text(
        "# Upper Yellow River external-validation data\n\n"
        "This archive contains 16 normalized daily station files from 2010-01-01 "
        "through 2024-03-31. Xiaheyan was excluded because the meteorological and "
        "runoff coordinates did not match. The reported test period is 2022-02-09 "
        "through 2024-03-31. Columns and units follow docs/DATA_DICTIONARY.md in the source release. "
        "Data are released under CC BY 4.0.\n",
        encoding="utf-8",
    )
    write_manifest(staging)


def copy_directory_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def build_artifact_archive(
    formal_run: Path,
    zero_shot_run: Path,
    transfer_run: Path,
    stdout_log: Path | None,
    stderr_log: Path | None,
    staging: Path,
) -> None:
    archive_root = staging / "results" / "archived"
    copy_directory_files(formal_run, archive_root / "upper_yellow_local_recalibration")
    copy_directory_files(zero_shot_run, archive_root / "upper_yellow_zero_shot")
    transfer_destination = archive_root / "jinsha_no_station_embedding"
    transfer_destination.mkdir(parents=True, exist_ok=True)
    for name in TRANSFER_FILES:
        shutil.copy2(transfer_run / name, transfer_destination / name)
    logs = archive_root / "upper_yellow_local_recalibration" / "logs"
    if stdout_log and stdout_log.is_file():
        logs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stdout_log, logs / "training_stdout.log")
    if stderr_log and stderr_log.is_file():
        logs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stderr_log, logs / "training_stderr.log")
    (staging / "UPPER_YELLOW_ARTIFACTS_README.md").write_text(
        "# Upper Yellow River external-validation artifacts\n\n"
        "This archive contains full one-step and recursive prediction tables, compact "
        "metrics, model-selection files, training logs, TCN/GRU checkpoints, and the "
        "Jinsha no-station-embedding transfer assets. The local-recalibration run used "
        "one random seed (42). The selected recursive aggregate chose KNN for both "
        "regression targets and must not be interpreted as a hybrid-only result.\n",
        encoding="utf-8",
    )
    write_manifest(staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--metadata-source", type=Path, required=True)
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--zero-shot-run", type=Path, required=True)
    parser.add_argument("--transfer-run", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_staging = args.output_dir / f"jinsha-daily-drought-hybrid-{VERSION}-upper-yellow-data"
    artifact_staging = args.output_dir / f"jinsha-daily-drought-hybrid-{VERSION}-upper-yellow-artifacts"
    for staging in (data_staging, artifact_staging):
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

    build_data_archive(args.prepared_data_dir, args.metadata_source, data_staging)
    build_artifact_archive(
        args.formal_run,
        args.zero_shot_run,
        args.transfer_run,
        args.stdout_log,
        args.stderr_log,
        artifact_staging,
    )
    data_zip = args.output_dir / f"jinsha-daily-drought-hybrid-{VERSION}-upper-yellow-data.zip"
    artifact_zip = args.output_dir / f"jinsha-daily-drought-hybrid-{VERSION}-upper-yellow-artifacts.zip"
    zip_tree(data_staging, data_zip)
    zip_tree(artifact_staging, artifact_zip)
    print(data_zip)
    print(artifact_zip)


if __name__ == "__main__":
    main()
