"""Predict photograph findings with a vision-language model on Modal GPU.

The occlusal half of a report comes from arch geometry, but the photograph half — gingival
inflammation, restorations, fissure sealants, carious processes, plaque, gingival recession —
is only visible in the pictures, which the geometry pipeline ignores entirely. Those findings
are ~3.6 sentences of every photograph report, and asserting them from base rates alone is
what currently limits precision.

This runs Qwen2.5-VL over the five standardised views per case and asks for the six findings
as strict JSON. The output is compared against labels parsed from the reports, so the decision
to use a prediction or fall back to its base rate is made on measured accuracy, per finding —
the same rule the geometry fields follow.

Photos are read from the ``b2t-photos`` volume, pre-downscaled by ``scripts/prepare_photos.py``.
"""

from __future__ import annotations

import modal

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

app = modal.App("b2t-photo-findings")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",  # qwen_vl_utils imports it for image preprocessing
        "transformers==4.51.3",
        "accelerate>=1.0",
        "qwen-vl-utils[decord]==0.0.11",
        "pillow>=10",
        "huggingface_hub[hf_transfer]>=0.26,<1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

model_cache = modal.Volume.from_name("b2t-model-cache", create_if_missing=True)
photos = modal.Volume.from_name("b2t-photos", create_if_missing=True)

VIEW_NAMES = [
    "right buccal (side) view",
    "frontal view",
    "left buccal (side) view",
    "lower occlusal (biting surface of the lower arch)",
    "upper occlusal (biting surface of the upper arch)",
]

SYSTEM_PROMPT = (
    "You are an orthodontist examining standardised intraoral photographs of one patient. "
    "Report only what is directly visible. Answer strictly as JSON with these six boolean "
    "keys and nothing else:\n"
    '{"gingival_inflammation": bool,  // gums red, swollen or bleeding\n'
    ' "gingival_recession": bool,     // gum margin receded exposing root surface\n'
    ' "caries": bool,                 // visible active carious lesion / decay\n'
    ' "restorations": bool,           // existing fillings or restorative work\n'
    ' "sealants": bool,               // pit-and-fissure sealants on molar grooves\n'
    ' "plaque": bool}                 // visible plaque, tartar or calculus\n'
    "Judge each independently. Use false when the sign is not visible."
)

FINDING_KEYS = [
    "gingival_inflammation", "gingival_recession", "caries",
    "restorations", "sealants", "plaque",
]

#: The occlusal fields are also visible in the photographs — sagittal class in the buccal
#: views, midlines and overjet frontally, crowding in the occlusal views. Geometry already
#: predicts these from the meshes, so the value here is a second opinion to ensemble against,
#: particularly for the sagittal classes where geometry is weakest (0.52-0.63).
OCCLUSAL_PROMPT = (
    "You are an orthodontist examining standardised intraoral photographs of one patient. "
    "Classify the occlusion. Answer strictly as JSON with these keys and nothing else:\n"
    '{"molar_right": "I"|"II edge-to-edge"|"II full"|"III",\n'
    ' "molar_left":  "I"|"II edge-to-edge"|"II full"|"III",\n'
    ' "canine_right":"I"|"II edge-to-edge"|"II full"|"III",\n'
    ' "canine_left": "I"|"II edge-to-edge"|"II full"|"III",\n'
    ' "overbite": "normal"|"increased"|"reduced"|"open",\n'
    ' "overjet": "normal"|"increased"|"reduced"|"negative",\n'
    ' "midlines": "centered"|"deviated",\n'
    ' "crossbite": "present"|"absent",\n'
    ' "crowding_upper": "absent"|"mild"|"moderate"|"severe",\n'
    ' "crowding_lower": "absent"|"mild"|"moderate"|"severe"}\n'
    "Angle class: I is the normal cusp-to-groove relationship; II means the lower arch sits "
    "distally (edge-to-edge is a half-cusp, full is a whole cusp); III means the lower arch "
    "sits mesially. Judge the right and left sides independently from their buccal views."
)

OCCLUSAL_KEYS = {
    "molar_right": ("I", "II edge-to-edge", "II full", "III"),
    "molar_left": ("I", "II edge-to-edge", "II full", "III"),
    "canine_right": ("I", "II edge-to-edge", "II full", "III"),
    "canine_left": ("I", "II edge-to-edge", "II full", "III"),
    "overbite": ("normal", "increased", "reduced", "open"),
    "overjet": ("normal", "increased", "reduced", "negative"),
    "midlines": ("centered", "deviated"),
    "crossbite": ("present", "absent"),
    "crowding_upper": ("absent", "mild", "moderate", "severe"),
    "crowding_lower": ("absent", "mild", "moderate", "severe"),
}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/root/.cache/huggingface": model_cache, "/photos": photos},
)
def predict(
    case_ids: list[str],
    batch_size: int = 8,
    model_name: str = MODEL_NAME,
    task: str = "dental",
) -> dict:
    """Return ``{case_id: {field: value}}``.

    ``task`` selects the question asked of the photographs: ``"dental"`` for the six
    photograph findings, ``"occlusal"`` for the Angle classes and related occlusal fields.
    """
    import json
    import re
    import tarfile
    import time
    from pathlib import Path

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Photos ship as one tarball rather than ~5,000 loose files: uploading many small files
    # to a volume is dramatically slower, and a single archive survives an interrupted
    # session. Extract once into container-local storage.
    photo_root = Path("/tmp/photos")
    if not photo_root.exists():
        archive = Path("/photos/photos_small.tar.gz")
        if archive.exists():
            print("extracting photo archive...", flush=True)
            started_extract = time.time()
            photo_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tar:
                tar.extractall(photo_root, filter="data")
            print(f"extracted in {time.time() - started_extract:.0f}s", flush=True)
        else:
            photo_root = Path("/photos")
    inner = photo_root / "photos_small"
    if inner.is_dir():
        photo_root = inner

    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"loaded {model_name}", flush=True)

    system_prompt = SYSTEM_PROMPT if task == "dental" else OCCLUSAL_PROMPT
    closing = ("Report the six findings as JSON." if task == "dental"
               else "Classify the occlusion as JSON.")

    def build_messages(case_id: str):
        case_dir = photo_root / case_id
        views = sorted(case_dir.glob("view*.jpg"))
        if not views:
            return None
        content = []
        for i, view in enumerate(views):
            label = VIEW_NAMES[i] if i < len(VIEW_NAMES) else f"view {i + 1}"
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "image", "image": f"file://{view}"})
        content.append({"type": "text", "text": closing})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def parse(raw: str) -> dict | None:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        payload = {}
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = {}

        out: dict = {}
        if task == "dental":
            for key in FINDING_KEYS:
                value = payload.get(key)
                if value is None:
                    m = re.search(rf'"{key}"\s*:\s*(true|false)', raw, re.I)
                    value = m.group(1) if m else None
                if isinstance(value, bool):
                    out[key] = value
                elif isinstance(value, str):
                    out[key] = value.strip().lower() in ("true", "yes", "present")
        else:
            for key, allowed in OCCLUSAL_KEYS.items():
                value = payload.get(key)
                if not isinstance(value, str):
                    m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', raw)
                    value = m.group(1) if m else None
                if not isinstance(value, str):
                    continue
                # Snap to the allowed vocabulary; the renderer only understands these.
                norm = value.strip().lower()
                for option in allowed:
                    if norm == option.lower():
                        out[key] = option
                        break
                else:
                    if key.startswith(("molar", "canine")):
                        if "iii" in norm or norm.endswith("3"):
                            out[key] = "III"
                        elif "ii" in norm or "2" in norm:
                            out[key] = "II edge-to-edge" if ("edge" in norm or "end" in norm) else "II full"
                        elif "i" in norm or "1" in norm:
                            out[key] = "I"
        return out or None

    from qwen_vl_utils import process_vision_info

    results: dict[str, dict] = {}
    started = time.time()
    pending = [c for c in case_ids if (photo_root / c).is_dir()]
    print(f"{len(pending)} cases with photos", flush=True)

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        batch_messages, ids = [], []
        for case_id in chunk:
            msgs = build_messages(case_id)
            if msgs:
                batch_messages.append(msgs)
                ids.append(case_id)
        if not batch_messages:
            continue

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)

        for case_id, raw in zip(ids, decoded):
            parsed = parse(raw)
            if parsed:
                results[case_id] = parsed

        done = start + len(chunk)
        if done % 40 == 0 or done >= len(pending):
            rate = done / max(1e-6, time.time() - started)
            print(f"  {done}/{len(pending)}  ({rate:.2f} cases/s)", flush=True)

    print(f"done in {time.time() - started:.0f}s, {len(results)} parsed", flush=True)
    return results


@app.local_entrypoint()
def main(cases: str = "", output: str = "artifacts/eval/photo_findings.json", limit: int = 0,
         batch_size: int = 8, task: str = "dental"):
    """Predict findings for cases listed in a JSON array file, or all cases on the volume.

    ``task`` is ``"dental"`` (the six photograph findings) or ``"occlusal"`` (Angle classes).
    """
    import json
    from pathlib import Path

    if cases:
        case_ids = json.loads(Path(cases).read_text(encoding="utf-8"))
    else:
        case_ids = sorted(p.name for p in Path("artifacts/photos_small").iterdir() if p.is_dir())
    if limit:
        case_ids = case_ids[:limit]
    print(f"requesting {len(case_ids)} cases")

    results = predict.remote(case_ids, batch_size=batch_size, task=task)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(results)} cases)")
