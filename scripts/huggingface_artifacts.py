"""Upload or restore safe reproducibility artifacts from a private Hugging Face repo."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "GOVINDFROM/odin-toothfairy4-artifacts"

# Raw clinical data, photos, per-case reports/predictions, and identifier-bearing feature
# tables are deliberately absent from this allowlist.
UPLOADS = (
    ("artifacts/models", "artifacts/models"),
    ("artifacts/reports", "artifacts/reports"),
    ("artifacts/eval/captioning_summary.json", "artifacts/eval/captioning_summary.json"),
    ("artifacts/eval/dental_health_tuning.json", "artifacts/eval/dental_health_tuning.json"),
    ("artifacts/eval/holdout.json", "artifacts/eval/holdout.json"),
    ("artifacts/eval/holdout_photo.json", "artifacts/eval/holdout_photo.json"),
    ("artifacts/eval/photo_findings_report.json", "artifacts/eval/photo_findings_report.json"),
    ("artifacts/geom/calibration.json", "artifacts/geom/calibration.json"),
    ("artifacts/bite2text-submission-source.zip", "archives/bite2text-submission-source.zip"),
    ("submission/resources", "submission/resources"),
    ("docs/assets/data_audit/bilateral_agreement.png", "plots/data_audit/bilateral_agreement.png"),
    ("docs/assets/data_audit/geometry_distributions.png", "plots/data_audit/geometry_distributions.png"),
    ("docs/assets/data_audit/integrity.png", "plots/data_audit/integrity.png"),
    ("docs/assets/data_audit/label_distributions.png", "plots/data_audit/label_distributions.png"),
    ("docs/assets/data_audit/split_drift.png", "plots/data_audit/split_drift.png"),
    ("docs/assets/data_audit/label_counts.json", "plots/data_audit/label_counts.json"),
    ("docs/assets/data_audit/summary.json", "plots/data_audit/summary.json"),
    # Keep the largest transfer last so a transient failure cannot prevent small artifacts.
    ("odin2026-bite2text-geometry.tar.gz", "archives/odin2026-bite2text-geometry.tar.gz"),
)


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def credentials() -> tuple[str, str]:
    load_dotenv()
    token = os.environ.get("hf")
    if not token:
        raise SystemExit("Set the hf variable in .env or the process environment.")
    return token, os.environ.get("HF_REPO_ID", DEFAULT_REPO)


def upload(token: str, repo_id: str) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    uploaded = 0
    for local_name, remote_name in UPLOADS:
        source = ROOT / local_name
        if not source.exists():
            continue
        if source.is_file() and source.stat().st_size > 50 * 1024 * 1024:
            upload_chunked(api, repo_id, source, remote_name)
        else:
            retry_upload(api, repo_id, source, remote_name)
        uploaded += 1
        print(f"uploaded {local_name} -> {remote_name}")
    print(f"Uploaded {uploaded} artifact groups to https://huggingface.co/{repo_id}")


def retry_upload(api: HfApi, repo_id: str, source: Path, remote_name: str) -> None:
    for attempt in range(1, 6):
        try:
            if source.is_dir():
                api.upload_folder(
                    repo_id=repo_id, repo_type="model", folder_path=source,
                    path_in_repo=remote_name, commit_message=f"Upload {remote_name}",
                )
            else:
                api.upload_file(
                    repo_id=repo_id, repo_type="model", path_or_fileobj=source,
                    path_in_repo=remote_name, commit_message=f"Upload {remote_name}",
                )
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def upload_chunked(api: HfApi, repo_id: str, source: Path, remote_name: str) -> None:
    """Upload large archives in 25 MiB restartable pieces."""
    with TemporaryDirectory(prefix="bite2text-hf-") as temp:
        chunk_dir = Path(temp)
        with source.open("rb") as stream:
            index = 0
            while block := stream.read(25 * 1024 * 1024):
                part = chunk_dir / f"{source.name}.part{index:03d}"
                part.write_bytes(block)
                retry_upload(api, repo_id, part, f"{remote_name}.parts/{part.name}")
                index += 1


def download(token: str, repo_id: str, destination: Path) -> None:
    snapshot_download(repo_id=repo_id, repo_type="model", token=token, local_dir=destination)
    for parts in destination.rglob("*.tar.gz.parts"):
        target = parts.with_suffix("")
        with target.open("wb") as output:
            for part in sorted(parts.glob("*.part*")):
                output.write(part.read_bytes())
        print(f"Reassembled {target}")
    print(f"Restored {repo_id} into {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upload")
    restore = sub.add_parser("download")
    restore.add_argument("--output", type=Path, default=ROOT / "restored_artifacts")
    args = parser.parse_args()
    token, repo_id = credentials()
    if args.command == "upload":
        upload(token, repo_id)
    else:
        download(token, repo_id, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
