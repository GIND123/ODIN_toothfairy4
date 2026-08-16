"""Byte-faithful replica of the ODIN 2026 Bite2Text offline scorer.

The official evaluation container sets ``RUNNING_ON_GRAND_CHALLENGE=1``, which routes
``evaluate.py`` away from HuggingFace ``evaluate`` and into its own local implementations. So
the number that lands on the leaderboard is *not* standard BLEU/METEOR:

* **BLEU-4** is corpus-level NLTK ``corpus_bleu`` with uniform 4-gram weights and smoothing
  method 1 — computed over the whole case set at once, not averaged per case.
* **METEOR** is a bespoke "METEOR-lite": exact token matching only, with no stemming, no
  WordNet synonyms and a first-match greedy alignment. Its F-mean is recall-weighted 9:1.

Both functions below are transcribed from the organisers' ``evaluation/evaluate.py`` so that
local optimisation targets the real objective rather than a lookalike. Reproduced under the
challenge repository's MIT licence.

Final ranking is ``0.8 * RadFact-F1 + 0.2 * mean(BLEU-4, METEOR)``; see ``score_final``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "bleu_4_local",
    "meteor_lite_batch",
    "meteor_lite_score",
    "tokenize",
    "CaptioningScore",
    "score_captioning",
    "score_final",
]


def tokenize(text: str) -> list[str]:
    """Organisers' tokenizer: word characters or single punctuation, lowercased."""
    return [token for token in re.findall(r"\w+|[^\w\s]", text.lower()) if token.strip()]


def bleu_4_local(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU-4 with smoothing method 1, exactly as the scorer computes it."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    predicted_tokens = [tokenize(prediction) for prediction in predictions]
    reference_tokens = [[tokenize(reference)] for reference in references]
    if not predicted_tokens:
        return 0.0

    return float(
        corpus_bleu(
            list_of_references=reference_tokens,
            hypotheses=predicted_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def _greedy_match_indices(prediction_tokens: list[str], reference_tokens: list[str]) -> list[int]:
    used = [False] * len(reference_tokens)
    indices: list[int] = []
    for token in prediction_tokens:
        for index, ref_token in enumerate(reference_tokens):
            if used[index] or token != ref_token:
                continue
            used[index] = True
            indices.append(index)
            break
    return indices


def _chunk_count(indices: list[int]) -> int:
    if not indices:
        return 0
    chunks = 1
    for current, previous in zip(indices[1:], indices[:-1]):
        if current != previous + 1:
            chunks += 1
    return chunks


def meteor_lite_score(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    """Single-pair METEOR-lite. Recall-weighted 9:1 with a fragmentation penalty."""
    matched_reference_indices = _greedy_match_indices(prediction_tokens, reference_tokens)
    matches = len(matched_reference_indices)
    if matches == 0:
        return 0.0

    precision = matches / len(prediction_tokens)
    recall = matches / len(reference_tokens)
    denominator = recall + 9.0 * precision
    if denominator == 0.0:
        return 0.0

    f_mean = (10.0 * precision * recall) / denominator
    chunks = _chunk_count(matched_reference_indices)
    penalty = 0.5 * (chunks / matches) ** 3
    return float((1.0 - penalty) * f_mean)


def meteor_lite_batch(predictions: list[str], references: list[str]) -> float:
    """Mean METEOR-lite over the case set."""
    if not predictions:
        return 0.0

    scores = []
    for prediction, reference in zip(predictions, references):
        prediction_tokens = tokenize(prediction)
        reference_tokens = tokenize(reference)
        if not prediction_tokens and not reference_tokens:
            scores.append(1.0)
            continue
        if not prediction_tokens or not reference_tokens:
            scores.append(0.0)
            continue
        scores.append(meteor_lite_score(prediction_tokens, reference_tokens))
    return float(sum(scores) / len(scores)) if scores else 0.0


@dataclass(frozen=True)
class CaptioningScore:
    bleu_4: float
    meteor: float

    @property
    def captioning(self) -> float:
        """The challenge's secondary score: the mean of BLEU-4 and METEOR."""
        return 0.5 * (self.bleu_4 + self.meteor)


def score_captioning(predictions: list[str], references: list[str]) -> CaptioningScore:
    return CaptioningScore(
        bleu_4=bleu_4_local(predictions, references),
        meteor=meteor_lite_batch(predictions, references),
    )


def score_final(radfact_f1: float, captioning: float) -> float:
    """``0.8 * clinical + 0.2 * captioning`` as published on the Ranking page."""
    return 0.8 * radfact_f1 + 0.2 * captioning
