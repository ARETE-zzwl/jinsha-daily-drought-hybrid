"""Package large manuscript artifacts for Zenodo deposition.

The GitHub repository intentionally excludes full prediction tables and model
checkpoints. This script collects those generated artifacts from the original
working-project output directory, writes a checksum manifest, and creates a
compressed archive suitable for a Zenodo upload.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


ARTIFACTS = [
    "predictions_daily_all_models.csv",
    "predictions_daily_recursive_all_models.csv",
    "best_checkpoint_tcn_daily_hybrid.pt",
    "best_checkpoint_gru_daily_hybrid.pt",
]


README_TEXT = """# Zenodo Large Artifacts

This archive contains large generated artifacts for the manuscript
"Physics-Deep Learning Hybrid Framework for Multi-Station Daily Drought Indices
and Flash Drought Forecasting in the Jinsha River Basin".

Files:

- `predictions_daily_all_models.csv`: full one-step prediction table from the manuscript main run.
- `predictions_daily_recursive_all_models.csv`: full recursive prediction table from the manuscript main run.
- `best_checkpoint_tcn_daily_hybrid.pt`: trained TCN-Transformer hybrid checkpoint.
- `best_checkpoint_gru_daily_hybrid.pt`: trained GRU-Transformer hybrid checkpoint.
- `MANIFEST.sha256`: SHA256 checksums for the files in this archive.

These files are generated outputs and are intentionally not committed to GitHub.
They should be cited through the Zenodo DOI associated with the manuscript's
open research package.
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_assets(source_run_dir: Path, output_dir: Path, version: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"jinsha-daily-drought-hybrid-{version}-large-artifacts"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    copied = []
    for name in ARTIFACTS:
        src = source_run_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = staging / name
        shutil.copy2(src, dst)
        copied.append(dst)

    readme = staging / "README_ZENODO_ARTIFACTS.md"
    readme.write_text(README_TEXT, encoding="utf-8")

    manifest_rows = []
    for path in copied + [readme]:
        manifest_rows.append(f"{sha256_file(path)}  {path.name}")
    (staging / "MANIFEST.sha256").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    archive = output_dir / f"jinsha-daily-drought-hybrid-{version}-large-artifacts.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(staging.iterdir()):
            zf.write(path, arcname=f"{staging.name}/{path.name}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Zenodo archive for large generated artifacts.")
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "output"
        / "hybrid_modal_physics_joint"
        / "daily_multitask_joint_8stations_journal_tier1_leakfree",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "zenodo_artifacts_v1.1.0",
    )
    parser.add_argument("--version", default="v1.1.0-wrr-revision")
    args = parser.parse_args()
    archive = package_assets(args.source_run_dir, args.output_dir, args.version)
    print(archive)


if __name__ == "__main__":
    main()
