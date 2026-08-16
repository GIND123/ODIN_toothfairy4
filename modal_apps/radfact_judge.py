"""RadFact judge on Modal GPU.

The challenge's clinical score is RadFact-F1, worth 0.8 of the final ranking, computed by an
LLM judge (``gpt-4o-mini`` by default in the organisers' evaluator, GPT-5.2 in the paper). With
no hosted API key available, this runs an open instruct model on a Modal GPU.

**Prompt fidelity is preserved**: the system messages, few-shot examples and JSON schemas are
loaded from the organisers' own ``radfact_lite`` package, and the precision/recall arithmetic
uses their ``metric`` module. Only the judge model differs, which affects absolute calibration
but not the relative comparison between our own candidate strategies — which is what this is
for.

Inference is plain ``transformers`` with manual batching rather than a served engine: it is
slower, but it has far fewer moving parts than a vLLM server and does not break when the
CUDA/transformers/vllm version triangle shifts.

Responses are parsed leniently. Entailment only needs one of two labels, so we look for the
label rather than demanding well-formed JSON, and report parsing falls back to sentence
splitting if the model's JSON is unusable. A judge that silently returns nothing would be far
worse than one that occasionally degrades.
"""

from __future__ import annotations

import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

app = modal.App("b2t-radfact-judge")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.48.3",
        "accelerate>=1.0",
        "radfact_lite>=0.1.0",
        "openai>=1.0.0",
        "huggingface_hub[hf_transfer]>=0.26,<1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

model_cache = modal.Volume.from_name("b2t-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": model_cache},
)
def radfact(candidates: dict[str, str], references: dict[str, str], model: str = MODEL_NAME) -> dict:
    """Score candidate reports against references; returns RadFact aggregates and per-case rows."""
    import json
    import re
    import time
    from dataclasses import asdict

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from radfact_lite.metric import aggregate_results, sample_result
    from radfact_lite.prompts import load_few_shot_json, load_system_prompt
    from radfact_lite.entailment import _expand_single_phrase_examples
    from radfact_lite.rf_types import ReportType

    report_type = ReportType.TOOTHFAIRY
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="cuda")
    llm.eval()
    print(f"loaded {model}", flush=True)

    def generate(chats: list[list[dict]], max_new_tokens: int, batch_size: int = 32) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(chats), batch_size):
            batch = chats[start : start + batch_size]
            texts = [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in batch
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=6144).to("cuda")
            with torch.no_grad():
                gen = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
            for i in range(len(batch)):
                outputs.append(
                    tokenizer.decode(gen[i][enc["input_ids"].shape[1] :], skip_special_tokens=True)
                )
            print(f"  generated {min(start + batch_size, len(chats))}/{len(chats)}", flush=True)
        return outputs

    # ---- Stage 1: report -> phrases (organisers' prompt + few-shots) ----
    parse_system = load_system_prompt("report_to_phrases", report_type, "system_message.txt")
    parse_shots = load_few_shot_json("report_to_phrases", report_type)

    def parse_chat(text: str) -> list[dict]:
        msgs = [{"role": "system", "content": parse_system +
                 '\nRespond with JSON: {"sentence_list": [{"orig": str, "new": [str]}]}'}]
        for shot in parse_shots:
            msgs.append({"role": "user", "content": shot["findings_text"]})
            msgs.append({"role": "assistant", "content": json.dumps(shot["parsed_report"], ensure_ascii=False)})
        msgs.append({"role": "user", "content": text})
        return msgs

    def sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def extract_phrases(raw: str, fallback: str) -> list[str]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                phrases = [
                    p.strip()
                    for s in payload.get("sentence_list", [])
                    for p in s.get("new", [])
                    if isinstance(p, str) and p.strip()
                ]
                if phrases:
                    return phrases
            except json.JSONDecodeError:
                pass
        return sentences(fallback)

    ids = sorted(set(candidates) & set(references))
    texts = [candidates[i] for i in ids] + [references[i] for i in ids]
    started = time.time()
    print(f"parsing {len(texts)} reports", flush=True)
    parsed_raw = generate([parse_chat(t) for t in texts], max_new_tokens=768)
    parsed = [extract_phrases(r, t) for r, t in zip(parsed_raw, texts)]
    cand_phrases = {i: parsed[k] for k, i in enumerate(ids)}
    ref_phrases = {i: parsed[len(ids) + k] for k, i in enumerate(ids)}

    # ---- Stage 2: phrase-level entailment ----
    ent_system = load_system_prompt("entailment", report_type, "system_message_ev_singlephrase.txt")
    ent_shots = load_few_shot_json("entailment", report_type)
    ent_prefix = [{"role": "system", "content": ent_system +
                   '\nRespond with JSON: {"phrase": str, "evidence": [str], '
                   '"status": "entailment" | "not_entailment"}'}]
    for shot in ent_shots:
        for pair in _expand_single_phrase_examples(shot):
            ent_prefix.append({"role": "user", "content": json.dumps(pair["input"], ensure_ascii=False)})
            ent_prefix.append({"role": "assistant", "content": json.dumps(pair["output"], ensure_ascii=False)})

    jobs: list[tuple[str, str, list[str]]] = []  # (case_id, direction, ...)
    chats: list[list[dict]] = []
    for case_id in ids:
        for phrase in cand_phrases[case_id]:
            jobs.append((case_id, "candidate", phrase))
            chats.append(ent_prefix + [{"role": "user", "content": json.dumps(
                {"reference": ref_phrases[case_id], "hypothesis": phrase}, ensure_ascii=False)}])
        for phrase in ref_phrases[case_id]:
            jobs.append((case_id, "reference", phrase))
            chats.append(ent_prefix + [{"role": "user", "content": json.dumps(
                {"reference": cand_phrases[case_id], "hypothesis": phrase}, ensure_ascii=False)}])

    print(f"judging {len(chats)} phrase pairs", flush=True)
    verdicts_raw = generate(chats, max_new_tokens=160)

    status_rx = re.compile(r'"status"\s*:\s*"(not_entailment|entailment)"', re.I)
    entailed = {case_id: {"candidate": 0, "reference": 0} for case_id in ids}
    for (case_id, direction, _), raw in zip(jobs, verdicts_raw):
        # Prefer the explicit status field; fall back to a substring test if the model's JSON
        # is malformed. "not_entailment" contains "entailment", so the negative is tested first.
        match = status_rx.search(raw)
        if match:
            if match.group(1).lower() == "entailment":
                entailed[case_id][direction] += 1
            continue
        low = raw.lower()
        if "not_entailment" in low or "not entailment" in low:
            continue
        if "entailment" in low:
            entailed[case_id][direction] += 1

    results = [
        sample_result(
            case_id,
            entailed[case_id]["candidate"],
            entailed[case_id]["reference"],
            len(cand_phrases[case_id]),
            len(ref_phrases[case_id]),
        )
        for case_id in ids
    ]
    aggregate = aggregate_results(results, 0)
    print(
        f"done in {time.time() - started:.0f}s  P={aggregate.logical_precision:.4f} "
        f"R={aggregate.logical_recall:.4f} F1={aggregate.logical_f1:.4f}",
        flush=True,
    )
    return {
        "model": model,
        "aggregate": asdict(aggregate),
        "per_case": [asdict(r) for r in results],
    }


@app.local_entrypoint()
def main(predictions: str, references: str, output: str = "artifacts/eval/radfact.json", limit: int = 0):
    """Score a predictions JSON against a references JSON (both ``{case_id: report}``)."""
    import json
    from pathlib import Path

    cand = json.loads(Path(predictions).read_text(encoding="utf-8"))
    refs = json.loads(Path(references).read_text(encoding="utf-8"))
    keys = sorted(set(cand) & set(refs))
    if limit:
        keys = keys[:limit]
    cand = {k: cand[k] for k in keys}
    refs = {k: refs[k] for k in keys}

    result = radfact.remote(cand, refs)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    agg = result["aggregate"]
    print(
        f"RadFact  P={agg['logical_precision']:.4f}  R={agg['logical_recall']:.4f}  "
        f"F1={agg['logical_f1']:.4f}  n={agg['num_samples']}"
    )
    print(f"Wrote {out}")
