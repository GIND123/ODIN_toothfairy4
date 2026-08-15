import subprocess
import sys
from pathlib import Path

import pandas as pd

from bite2text.audit import run_audit
from bite2text.baseline import predict, train, validate_submission
from bite2text.index import build_manifest


def test_end_to_end(tmp_path: Path):
    data = tmp_path / "fixture"
    subprocess.run(
        [sys.executable, "scripts/make_fixture.py", "--output", str(data), "--cases", "3"],
        check=True,
    )
    manifest = tmp_path / "manifest.csv"
    frame = build_manifest(data, manifest)
    assert len(frame) == 3
    assert (pd.read_csv(manifest)["ios_count"] == 2).all()
    summary = run_audit(manifest, tmp_path / "audit")
    assert summary["invalid_files"] == 0
    assert (tmp_path / "audit" / "report.html").exists()
    model, submission = tmp_path / "model.pkl", tmp_path / "submission.json"
    train(manifest, model)
    assert len(predict(manifest, model, submission)) == 3
    assert validate_submission(submission, manifest)["valid"] is True
