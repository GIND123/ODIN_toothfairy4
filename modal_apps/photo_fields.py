"""Train a small vision model to predict the occlusal fields from the photographs.

Geometry reads the arches well but is blind to several fields the photographs show plainly:
midlines (0.50 accuracy), crowding (0.52), canine class (0.53). Those weak fields are what caps
the report — with *perfect* field values the same renderer scores BLEU-4 0.334 against 0.284
achieved, and unlike padding the report with base-rate sentences, better field values transfer
to the hidden test because they fill the same slots more often correctly.

Deliberately a ~30M-parameter ConvNeXt-Tiny rather than a 7B VLM, for two reasons: 600 training
cases cannot support a large model, and the result has to run on CPU inside the submission
container alongside the geometry pipeline.

The five standardised views share one backbone; their features are attention-pooled into a case
embedding, which feeds one linear head per field. Output is per-field probabilities so they can
be fused with the geometry model rather than replacing it.
"""

from __future__ import annotations

import modal

app = modal.App("b2t-photo-fields")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "timm==1.0.15",
        "pillow>=10", "numpy<3", "scikit-learn>=1.4",
        "huggingface_hub[hf_transfer]>=0.26,<1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

model_cache = modal.Volume.from_name("b2t-model-cache", create_if_missing=True)
photos = modal.Volume.from_name("b2t-photos", create_if_missing=True)
outputs = modal.Volume.from_name("b2t-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={"/root/.cache/huggingface": model_cache, "/photos": photos, "/out": outputs},
)
def train(
    labels: dict[str, dict[str, str]],
    splits: dict[str, list[str]],
    epochs: int = 12,
    lr: float = 2e-4,
    batch_size: int = 12,
    image_size: int = 224,
    backbone: str = "convnext_tiny",
) -> dict:
    """Train the multi-task photo classifier and return per-field probabilities.

    ``labels``  : case_id -> {field: value}; missing fields are skipped in the loss.
    ``splits``  : {"fit": [...], "val": [...], "test": [...]}
    """
    import json
    import subprocess
    import tarfile
    import time
    from pathlib import Path

    import numpy as np
    import timm
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    # The volume holds one tarball rather than 5k small files: extracting once here is far
    # faster than uploading a large directory tree.
    photo_root = Path("/tmp/photos_small")
    if not photo_root.is_dir():
        started = time.time()
        with tarfile.open("/photos/photos_small.tar.gz") as tar:
            tar.extractall("/tmp")
        print(f"extracted photos in {time.time() - started:.0f}s", flush=True)
    n_cases = len(list(photo_root.iterdir()))
    print(f"{n_cases} case directories", flush=True)

    fields = sorted({f for v in labels.values() for f in v})
    vocab = {f: sorted({labels[c][f] for c in labels if f in labels[c]}) for f in fields}
    index = {f: {v: i for i, v in enumerate(vals)} for f, vals in vocab.items()}
    print("fields: " + ", ".join(f"{f}({len(v)})" for f, v in vocab.items()), flush=True)

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(0.25, 0.25, 0.2, 0.03),
        transforms.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])

    N_VIEWS = 5

    class Cases(Dataset):
        def __init__(self, case_ids: list[str], tf, with_labels: bool = True):
            self.ids = [c for c in case_ids if (photo_root / c).is_dir()]
            self.tf, self.with_labels = tf, with_labels

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, i: int):
            case_id = self.ids[i]
            views = sorted((photo_root / case_id).glob("view*.jpg"))[:N_VIEWS]
            tensors = [self.tf(Image.open(v).convert("RGB")) for v in views]
            while len(tensors) < N_VIEWS:  # pad short cases by repeating the first view
                tensors.append(tensors[0].clone() if tensors else torch.zeros(3, image_size, image_size))
            stack = torch.stack(tensors)
            if not self.with_labels:
                return stack, case_id
            target = torch.full((len(fields),), -100, dtype=torch.long)
            for j, f in enumerate(fields):
                value = labels.get(case_id, {}).get(f)
                if value is not None and value in index[f]:
                    target[j] = index[f][value]
            return stack, target

    class MultiTask(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            dim = self.backbone.num_features
            # Attention pooling over the five views: the model learns which view answers which
            # question (buccal views for sagittal class, occlusal views for crowding).
            self.attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
            self.dropout = nn.Dropout(0.3)
            self.heads = nn.ModuleList([nn.Linear(dim, len(vocab[f])) for f in fields])

        def forward(self, x):  # x: (B, V, 3, H, W)
            b, v = x.shape[:2]
            feats = self.backbone(x.flatten(0, 1)).view(b, v, -1)
            weights = torch.softmax(self.attn(feats), dim=1)
            pooled = self.dropout((feats * weights).sum(dim=1))
            return [head(pooled) for head in self.heads]

    device = "cuda"
    model = MultiTask().to(device)
    fit_loader = DataLoader(Cases(splits["fit"], train_tf), batch_size=batch_size,
                            shuffle=True, num_workers=4, drop_last=True)
    loaders = {
        name: DataLoader(Cases(splits[name], eval_tf), batch_size=batch_size, num_workers=4)
        for name in ("val", "test")
    }
    print(f"fit={len(fit_loader.dataset)} val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset)}", flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, total_steps=epochs * max(1, len(fit_loader)), pct_start=0.25)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")

    def evaluate(loader) -> tuple[dict[str, float], dict]:
        model.eval()
        correct = {f: [0, 0] for f in fields}
        probs: dict[str, dict[str, list[float]]] = {}
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                for j, f in enumerate(fields):
                    p = torch.softmax(logits[j].float(), dim=1).cpu().numpy()
                    pred = p.argmax(1)
                    mask = (targets[:, j] != -100).numpy()
                    correct[f][0] += int((pred[mask] == targets[:, j].numpy()[mask]).sum())
                    correct[f][1] += int(mask.sum())
        return {f: (c / t if t else float("nan")) for f, (c, t) in correct.items()}, probs

    best_mean, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for images, targets in fit_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = sum(loss_fn(logits[j], targets[:, j]) for j in range(len(fields)))
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += float(loss)
        val_acc, _ = evaluate(loaders["val"])
        mean_acc = float(np.nanmean(list(val_acc.values())))
        marker = ""
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  *best"
        print(f"epoch {epoch:2d}  loss={total / max(1, len(fit_loader)):6.3f}  "
              f"val mean acc={mean_acc:.4f}{marker}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final per-field probabilities on val and test, for fusing with the geometry model.
    def predict(loader_ids: list[str]) -> dict[str, dict[str, list[float]]]:
        ds = Cases(loader_ids, eval_tf, with_labels=False)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
        out: dict[str, dict[str, list[float]]] = {}
        model.eval()
        with torch.no_grad():
            for images, case_ids in loader:
                images = images.to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                for k, case_id in enumerate(case_ids):
                    out[case_id] = {
                        f: torch.softmax(logits[j].float(), dim=1)[k].cpu().tolist()
                        for j, f in enumerate(fields)
                    }
        return out

    val_acc, _ = evaluate(loaders["val"])
    test_acc, _ = evaluate(loaders["test"])
    print("\nper-field accuracy (val / test):", flush=True)
    for f in fields:
        print(f"  {f:18s} {val_acc[f]:.3f} / {test_acc[f]:.3f}", flush=True)

    Path("/out").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "fields": fields, "vocab": vocab,
                "backbone": backbone, "image_size": image_size},
               "/out/photo_fields.pt")
    outputs.commit()
    print("saved /out/photo_fields.pt", flush=True)

    return {
        "fields": fields,
        "vocab": vocab,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "probabilities": {"val": predict(splits["val"]), "test": predict(splits["test"])},
    }


@app.local_entrypoint()
def main(labels: str, splits: str, output: str = "artifacts/eval/photo_fields.json",
         epochs: int = 12, backbone: str = "convnext_tiny", image_size: int = 224,
         batch_size: int = 12, lr: float = 2e-4):
    import json
    from pathlib import Path

    result = train.remote(
        json.loads(Path(labels).read_text(encoding="utf-8")),
        json.loads(Path(splits).read_text(encoding="utf-8")),
        epochs=epochs, backbone=backbone, image_size=image_size,
        batch_size=batch_size, lr=lr,
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Wrote {out}")
