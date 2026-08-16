"""Downscale the intraoral photographs for vision-model inference.

The released photographs are ~10 MP each and there are five per case, which is far more
resolution than a vision-language model consumes and far too much to ship to a GPU worker
(≈7 GB raw). Downscaling to a model-native edge length first cuts that to a few hundred
megabytes with no loss of the findings we care about — gingival inflammation, restorations,
sealants, plaque are all coarse, high-contrast cues.

The five standardised views (frontal, right buccal, left buccal, upper occlusal, lower
occlusal) are kept as separate files so the model can be told which is which.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def process_case(args: tuple[Path, Path, int, int]) -> tuple[str, int, str]:
    case_dir, out_root, edge, quality = args
    case_id = case_dir.name
    src = case_dir / "intraoral-photo"
    if not src.is_dir():
        return case_id, 0, "no photo directory"

    photos = sorted(p for p in src.iterdir() if p.suffix in SUFFIXES)
    if not photos:
        return case_id, 0, "no photos"

    dest = out_root / case_id
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, photo in enumerate(photos[:5], start=1):
        target = dest / f"view{i}.jpg"
        if target.exists() and target.stat().st_size > 0:
            written += 1
            continue
        try:
            with Image.open(photo) as img:
                img = img.convert("RGB")
                scale = edge / max(img.size)
                if scale < 1.0:
                    img = img.resize(
                        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                        Image.LANCZOS,
                    )
                img.save(target, "JPEG", quality=quality, optimize=True)
            written += 1
        except Exception as exc:  # noqa: BLE001 - a broken photo must not stop the run
            return case_id, written, f"{type(exc).__name__}: {exc}"
    return case_id, written, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/bite2text"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/photos_small"))
    ap.add_argument("--edge", type=int, default=560, help="longest edge in pixels")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cases = sorted(p for p in args.root.iterdir() if p.is_dir())
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases -> {args.output} (edge={args.edge}px)")

    done = failed = total_photos = 0
    jobs = [(c, args.output, args.edge, args.quality) for c in cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_case, j) for j in jobs]
        for fut in as_completed(futures):
            case_id, n, err = fut.result()
            done += 1
            total_photos += n
            if err:
                failed += 1
                if failed <= 10:
                    print(f"  {case_id}: {err}", file=sys.stderr)
            if done % 100 == 0:
                print(f"  {done}/{len(cases)}", flush=True)

    size = sum(f.stat().st_size for f in args.output.rglob("*.jpg"))
    print(f"\n{total_photos} photos from {done} cases, {failed} with issues")
    print(f"total {size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
