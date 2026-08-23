"""Verify the CPU photo model against the GPU predictions, and probe the few-view case.

Two things must hold before the photo model can ship:

1. **The CPU path reproduces the GPU path.** The container re-implements the model rather than
   importing the training code, so agreement on argmax is what proves the re-implementation is
   faithful. Small probability differences are expected (GPU ran in bf16 autocast).

2. **Degradation with fewer views is understood.** Training always saw five standardised views,
   but the challenge sample ships a single ``intraoral-photo.tiff``. If accuracy collapses when
   views are duplicated, the model must be gated off rather than fused in — a confidently wrong
   prediction is worse than falling back to geometry.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bite2text.photo import PhotoFieldModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/photo_fields.pt"))
    ap.add_argument("--photos", type=Path, default=Path("artifacts/photos_small"))
    ap.add_argument("--gpu-predictions", type=Path, default=Path("artifacts/eval/photo_fields_v2.json"))
    ap.add_argument("--labels", type=Path, default=Path("artifacts/eval/photo_labels.json"))
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    gpu = json.loads(args.gpu_predictions.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    gpu_test = gpu["probabilities"]["test"]
    fields = gpu["fields"]
    vocab = gpu["vocab"]

    model = PhotoFieldModel(args.checkpoint)
    case_ids = [c for c in list(gpu_test)[: args.limit] if (args.photos / c).is_dir()]
    print(f"checking {len(case_ids)} cases\n")

    agree = total = 0
    max_delta = 0.0
    acc_full = {f: [0, 0] for f in fields}
    acc_one = {f: [0, 0] for f in fields}

    for case_id in case_ids:
        views = sorted((args.photos / case_id).glob("view*.jpg"))
        full = model.predict(views)
        single = model.predict(views[:1])
        if full is None:
            print(f"  {case_id}: CPU predict returned None")
            continue

        for field in fields:
            cpu_p = full.probabilities[field]
            gpu_p = np.array(gpu_test[case_id][field])
            max_delta = max(max_delta, float(np.abs(cpu_p - gpu_p).max()))
            agree += int(cpu_p.argmax() == gpu_p.argmax())
            total += 1

            truth = labels.get(case_id, {}).get(field)
            if truth is None:
                continue
            classes = vocab[field]
            if truth in classes:
                gold = classes.index(truth)
                acc_full[field][0] += int(cpu_p.argmax() == gold)
                acc_full[field][1] += 1
                if single is not None:
                    acc_one[field][0] += int(single.probabilities[field].argmax() == gold)
                    acc_one[field][1] += 1

    print(f"CPU vs GPU argmax agreement: {agree}/{total} = {agree / max(1, total):.3f}")
    print(f"max probability difference : {max_delta:.4f}   (bf16 autocast on GPU)\n")

    print(f"{'field':<18}{'5 views':>9}{'1 view':>9}{'drop':>8}")
    drops = []
    for field in fields:
        c5, n5 = acc_full[field]
        c1, n1 = acc_one[field]
        if not n5:
            continue
        a5, a1 = c5 / n5, c1 / max(1, n1)
        drops.append(a5 - a1)
        print(f"{field:<18}{a5:9.3f}{a1:9.3f}{a5 - a1:+8.3f}")
    if drops:
        print(f"{'MEAN':<18}{st.mean(a for a in [acc_full[f][0] / acc_full[f][1] for f in fields if acc_full[f][1]]):9.3f}"
              f"{st.mean(acc_one[f][0] / acc_one[f][1] for f in fields if acc_one[f][1]):9.3f}"
              f"{st.mean(drops):+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
