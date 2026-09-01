"""Run the from-scratch Upper Yellow River external-basin replication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--upper-yellow-data-dir",
        type=Path,
        default=ROOT / "data" / "external" / "upper_yellow_station_daily",
    )
    parser.add_argument(
        "--upper-yellow-metadata",
        type=Path,
        default=ROOT / "data" / "external" / "upper_yellow_station_metadata.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "runs",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper, trainer_args = parse_wrapper_args(sys.argv[1:])
    data_dir = wrapper.upper_yellow_data_dir.resolve()
    metadata = wrapper.upper_yellow_metadata.resolve()
    output_root = wrapper.output_root.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Upper Yellow River station data not found: {data_dir}. "
            "Download and extract the Zenodo data archive first."
        )
    if not metadata.is_file():
        raise FileNotFoundError(f"Upper Yellow River metadata not found: {metadata}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import drought_hybrid.config as config_module
    import drought_hybrid.data as data_module

    data_module.DATA_DIR = data_dir
    config_module.OUT_DIR = output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if "--dem-summary-file" not in trainer_args:
        trainer_args.extend(["--dem-summary-file", str(metadata)])
    sys.argv = [sys.argv[0], *trainer_args]

    from drought_hybrid.daily_trainer import main as train_main

    print(f"[UPPER-YELLOW] data_dir={data_dir}", flush=True)
    print(f"[UPPER-YELLOW] output_root={output_root}", flush=True)
    train_main()


if __name__ == "__main__":
    main()
