"""Create a Zenodo deposition for the release source and large artifacts.

Requirements:
  - Set ZENODO_ACCESS_TOKEN or ZENODO_TOKEN with scopes deposit:write and,
    if using --publish, deposit:actions.
  - Run scripts/prepare_zenodo_assets.py first.
  - Create the release source archive with git archive, or pass --source-zip.

By default this creates an unpublished draft and reserves a DOI. Publishing is
irreversible, so use --publish only after reviewing the draft metadata/files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


TITLE = "Reproducibility package for a physics-deep learning daily drought forecasting framework"


def token_from_env() -> str:
    token = os.environ.get("ZENODO_ACCESS_TOKEN") or os.environ.get("ZENODO_TOKEN")
    if not token:
        raise RuntimeError("Set ZENODO_ACCESS_TOKEN or ZENODO_TOKEN before running this script.")
    return token


def create_deposition(api_url: str, token: str) -> dict:
    url = f"{api_url.rstrip('/')}/deposit/depositions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    metadata = {
        "metadata": {
            "title": TITLE,
            "upload_type": "dataset",
            "description": (
                "Reproducibility package for the manuscript 'Physics-Deep Learning Hybrid Framework "
                "for Multi-Station Daily Drought Indices and Flash Drought Forecasting in the Jinsha River Basin'. "
                "This record contains the GitHub release source archive, full prediction tables, and trained model "
                "checkpoints. The source code is released under the MIT License; processed data and derived artifacts "
                "are released under CC BY 4.0. The GitHub repository is "
                "https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid."
            ),
            "creators": [
                {"name": "Zhu, Shibang"},
                {"name": "Wang, Zhaocai"},
                {"name": "Zhang, Gengxi"},
                {"name": "Zhong, Huayu"},
            ],
            "access_right": "open",
            "license": "cc-by-4.0",
            "prereserve_doi": True,
            "keywords": [
                "drought forecasting",
                "flash drought",
                "Jinsha River Basin",
                "deep learning",
                "model checkpoint",
                "prediction table",
                "reproducibility package",
                "source code",
            ],
            "related_identifiers": [
                {
                    "identifier": "https://github.com/ARETE-zzwl/jinsha-daily-drought-hybrid",
                    "relation": "isSupplementTo",
                    "scheme": "url",
                }
            ],
        }
    }
    response = requests.post(url, data=json.dumps(metadata), headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def upload_file(deposition: dict, artifact_zip: Path, token: str) -> dict:
    bucket_url = deposition["links"]["bucket"].rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    with artifact_zip.open("rb") as f:
        response = requests.put(f"{bucket_url}/{artifact_zip.name}", data=f, headers=headers, timeout=600)
    response.raise_for_status()
    return response.json()


def publish_deposition(api_url: str, deposition_id: int, token: str) -> dict:
    url = f"{api_url.rstrip('/')}/deposit/depositions/{deposition_id}/actions/publish"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(url, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Zenodo draft deposition for manuscript reproducibility files.")
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "zenodo_artifacts"
        / "jinsha-daily-drought-hybrid-v1.0.0-wrr-submission-source.zip",
    )
    parser.add_argument(
        "--artifact-zip",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "zenodo_artifacts"
        / "jinsha-daily-drought-hybrid-v1.0.0-wrr-submission-large-artifacts.zip",
    )
    parser.add_argument("--api-url", default="https://zenodo.org/api")
    parser.add_argument("--publish", action="store_true", help="Publish immediately. This is irreversible.")
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "zenodo_artifacts" / "zenodo_deposition_result.json",
    )
    args = parser.parse_args()

    for archive in (args.source_zip, args.artifact_zip):
        if not archive.exists():
            raise FileNotFoundError(archive)

    token = token_from_env()
    deposition = create_deposition(args.api_url, token)
    upload_results = [
        upload_file(deposition, args.source_zip, token),
        upload_file(deposition, args.artifact_zip, token),
    ]

    result = {"deposition": deposition, "uploads": upload_results, "published": None}
    if args.publish:
        result["published"] = publish_deposition(args.api_url, deposition["id"], token)

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    reserved = deposition.get("metadata", {}).get("prereserve_doi", {})
    print(json.dumps({"deposition_id": deposition["id"], "reserved_doi": reserved.get("doi"), "result_json": str(args.result_json)}, indent=2))


if __name__ == "__main__":
    main()
