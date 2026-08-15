from __future__ import annotations

import argparse
from pathlib import Path

import trimesh
from PIL import Image

FOLDERS = [
    "ios",
    "intraoral-photo",
    "reports_ios_it",
    "reports_ios_en",
    "reports_intraoral-photo_it",
    "reports_intraoral-photo_en",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.output)
    for folder in FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    for index in range(1, args.cases + 1):
        case = f"fixture{index:03d}"
        for jaw in ("upper", "lower"):
            trimesh.creation.icosphere(subdivisions=1, radius=1 + index / 20).export(
                root / "ios" / f"{case}_{jaw}.stl"
            )
        for view, color in (
            ("front", (170, 90, 90)),
            ("left", (140, 80, 80)),
            ("right", (190, 100, 100)),
        ):
            Image.new("RGB", (160 + index, 100 + index), color).save(
                root / "intraoral-photo" / f"{case}_{view}.png"
            )
        texts = {
            "reports_ios_it": f"Caso {index} con lieve affollamento dentale e relazione occlusale stabile.",
            "reports_ios_en": f"Case {index} shows mild dental crowding and a stable occlusal relationship.",
            "reports_intraoral-photo_it": f"Le fotografie del caso {index} mostrano lieve affollamento.",
            "reports_intraoral-photo_en": f"Photographs for case {index} show mild crowding.",
        }
        for folder, text in texts.items():
            (root / folder / f"{case}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
