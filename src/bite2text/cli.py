from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import run_audit
from .baseline import predict, train, validate_submission
from .index import build_manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bite2text", description="Bite2Text data and baseline toolkit")
    sub = p.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--data-root", required=True)
    index.add_argument("--output", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--output-dir", required=True)
    training = sub.add_parser("train-baseline")
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--manifest", required=True)
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--output", required=True)
    validate = sub.add_parser("validate-submission")
    validate.add_argument("--input", required=True)
    validate.add_argument("--manifest", required=True)
    all_cmd = sub.add_parser("all")
    all_cmd.add_argument("--data-root", required=True)
    all_cmd.add_argument("--output-dir", required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "index":
        result = {"cases": len(build_manifest(args.data_root, args.output))}
    elif args.command == "audit":
        result = run_audit(args.manifest, args.output_dir)
    elif args.command == "train-baseline":
        train(args.manifest, args.output)
        result = {"model": args.output}
    elif args.command == "predict":
        result = {"cases": len(predict(args.manifest, args.model, args.output))}
    elif args.command == "validate-submission":
        result = validate_submission(args.input, args.manifest)
    else:
        out = Path(args.output_dir)
        manifest = out / "manifest.csv"
        frame = build_manifest(args.data_root, manifest)
        summary = run_audit(manifest, out / "audit")
        result = {"cases": len(frame), "manifest": str(manifest), "audit": summary}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
