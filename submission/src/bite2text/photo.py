"""CPU inference for the photograph field model.

Mirrors the training-time model in ``modal_apps/photo_fields.py``: one shared ConvNeXt backbone
over the five standardised views, attention-pooled into a case embedding, then one linear head
per clinical field. Loading is lazy and every failure path returns ``None`` so the caller falls
back to geometry — a missing photo prediction must never cost the case its report.

Two input realities this has to survive on the hidden test:

* **Fewer than five photographs.** The challenge sample ships a single ``intraoral-photo.tiff``
  per case, while the training data has five separate views. Views are padded by repetition,
  and ``MIN_VIEWS_FOR_TRUST`` gates whether the prediction is used at all.
* **Multi-page TIFF.** A single file may hold all five views as pages, so pages are expanded
  before padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["PhotoFieldModel", "MIN_VIEWS_FOR_TRUST"]

#: Below this many distinct views the model is running on duplicated inputs and its accuracy is
#: not the accuracy that was validated, so its predictions are discarded rather than fused.
MIN_VIEWS_FOR_TRUST = 3

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_N_VIEWS = 5


@dataclass
class PhotoPrediction:
    """Per-field class probabilities, keyed by field then aligned to the model's vocabulary."""

    probabilities: dict[str, np.ndarray]
    vocab: dict[str, list[str]]
    n_views: int


class PhotoFieldModel:
    """Lazily-loaded ConvNeXt multi-task classifier over intraoral photographs."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._model: Any = None
        self._fields: list[str] = []
        self._vocab: dict[str, list[str]] = {}
        self._image_size = 288
        self._failed = False

    # -- loading ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed or not self.checkpoint_path.exists():
            return False
        try:
            import timm
            import torch
            import torch.nn as nn

            torch.set_num_threads(1)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self._fields = list(checkpoint["fields"])
            self._vocab = {k: list(v) for k, v in checkpoint["vocab"].items()}
            self._image_size = int(checkpoint.get("image_size", 288))
            fields, vocab = self._fields, self._vocab

            class MultiTask(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.backbone = timm.create_model(
                        checkpoint["backbone"], pretrained=False, num_classes=0
                    )
                    dim = self.backbone.num_features
                    self.attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
                    self.dropout = nn.Dropout(0.3)
                    self.heads = nn.ModuleList([nn.Linear(dim, len(vocab[f])) for f in fields])

                def forward(self, x):  # (B, V, 3, H, W)
                    b, v = x.shape[:2]
                    feats = self.backbone(x.flatten(0, 1)).view(b, v, -1)
                    weights = torch.softmax(self.attn(feats), dim=1)
                    pooled = self.dropout((feats * weights).sum(dim=1))
                    return [head(pooled) for head in self.heads]

            model = MultiTask()
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self._model = model
            return True
        except Exception:  # noqa: BLE001 - inference must degrade to geometry, never raise
            self._failed = True
            self._model = None
            return False

    # -- image handling -----------------------------------------------------------

    def _load_views(self, photo_paths: list[Path]) -> tuple[np.ndarray, int]:
        """Return a (V, 3, H, W) float array and the number of *distinct* views found."""
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        frames = []
        for path in photo_paths:
            try:
                with Image.open(path) as img:
                    # A single file may hold several pages (multi-page TIFF).
                    pages = getattr(img, "n_frames", 1)
                    for page in range(min(pages, _N_VIEWS)):
                        img.seek(page)
                        frames.append(self._to_array(img.convert("RGB")))
                        if len(frames) >= _N_VIEWS:
                            break
            except Exception:  # noqa: BLE001 - skip unreadable photos
                continue
            if len(frames) >= _N_VIEWS:
                break

        distinct = len(frames)
        if not frames:
            return np.zeros((0, 3, self._image_size, self._image_size), dtype=np.float32), 0
        while len(frames) < _N_VIEWS:
            frames.append(frames[0].copy())
        return np.stack(frames), distinct

    def _to_array(self, image) -> np.ndarray:
        from PIL import Image

        resized = image.resize((self._image_size, self._image_size), Image.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        return arr.transpose(2, 0, 1)

    # -- public API ---------------------------------------------------------------

    def predict(self, photo_paths: list[Path]) -> PhotoPrediction | None:
        """Predict field probabilities from a case's photographs, or ``None`` if unusable."""
        if not photo_paths or not self._ensure_loaded():
            return None
        try:
            import torch

            views, distinct = self._load_views(photo_paths)
            if distinct == 0:
                return None
            with torch.no_grad():
                logits = self._model(torch.from_numpy(views).unsqueeze(0))
            probabilities = {
                field: torch.softmax(logits[i].float(), dim=1)[0].numpy()
                for i, field in enumerate(self._fields)
            }
            return PhotoPrediction(probabilities, self._vocab, distinct)
        except Exception:  # noqa: BLE001
            return None
