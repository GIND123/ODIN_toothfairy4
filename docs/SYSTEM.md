# Bite2Text system design

Task 2 of ODIN 2026: generate an orthodontic report from a registered upper/lower intraoral
scan pair plus standardised intraoral photographs.

## What the challenge actually scores

Reverse-engineered from the organisers' `evaluation/evaluate.py` and the published Ranking
page, not assumed:

```
final = 0.8 * RadFact-F1 + 0.2 * mean(BLEU-4, METEOR)
```

* **RadFact-F1** (`radfact_lite` 0.1.0, on PyPI) parses both reports into phrases with an LLM,
  then asks the LLM whether each phrase is entailed by the other report's phrase list.
  *Precision* = fraction of our phrases the reference entails (penalises hallucination);
  *recall* = fraction of reference phrases we entail (penalises omission). The aggregate F1 is
  the harmonic mean of the **mean** precision and **mean** recall across cases.
* **BLEU-4 / METEOR are not the standard implementations.** The evaluation container sets
  `RUNNING_ON_GRAND_CHALLENGE=1`, which routes past HuggingFace `evaluate` into local
  fallbacks: corpus-level NLTK BLEU-4 with smoothing method 1, and a bespoke "METEOR-lite"
  with exact token matching only — no stemming, no WordNet — whose F-mean is
  `10PR/(R+9P)`, i.e. **recall weighted 9:1**. Longer, more complete reports are rewarded.

`src/bite2text/eval/gc_metrics.py` reproduces both, so local optimisation targets the real
objective. `modal_apps/radfact_judge.py` runs the organisers' own `radfact_lite` package
against a vLLM-served open model for the clinical half.

## What the corpus looks like

Measured over 996 released cases / 1,496 English IOS reports (`scripts/analyze_reports.py`):

| Property | Value |
|---|---|
| IOS reports | 1,496 across 991 cases (≈50% of cases have 2+ independent reports) |
| Median length | 95 tokens, 6 sentences |
| **Inter-clinician agreement** | token Jaccard **0.472**, finding Jaccard **0.469** |

Reports follow a fixed dictation order — **transverse → vertical → sagittal → midlines →
occlusal curves → crowding** — and reuse a small stock of phrasings. Two consequences drive
the whole design:

1. Two clinicians describing the same patient agree on under half their findings. That is the
   practical ceiling, and it explains why published RadFact-F1 tops out at 0.372 for models
   against 0.650 for humans.
2. Because the corpus is so formulaic, **matching the structure is worth more than matching
   the patient**. Measured on real references:

   | Strategy | BLEU-4 | METEOR | Captioning |
   |---|---|---|---|
   | One clinician vs another | 0.237 | 0.527 | 0.382 |
   | A single fixed report for every case | 0.470 | 0.625 | 0.547 |
   | Our renderer, modal values | 0.396 | 0.563 | 0.479 |
   | Our renderer, **perfect** values | 0.601 | 0.764 | **0.682** |

   A constant report beats two clinicians agreeing with each other. Field accuracy is worth
   roughly +0.20 captioning on top of correct structure.

## Architecture

```
IOS meshes ─► canonicalise frame ─► arch profiles ─► clinical measurements
                                                            │
                                             per-field gradient boosting
                                                            │
                                          findings ─► narrative renderer ─► report
```

### 1. Frame canonicalisation (`geom/canonical.py`)

The paper states the scans are in RAS. **They are not consistently so.** Bits2Bites is
anterior=+Y; the released Bite2Text scans are anterior=−Y, and some cases put the superior
axis on Y rather than Z. Hardcoding the stated convention silently corrupts every
measurement, so the frame is re-derived per case:

* **Superior** = direction separating the two arch centroids.
* **Anterior** = the direction along which the arch *tapers* (a dental arch is narrow at the
  incisors, wide at the molars), corroborated by the arch closing into a single body
  anteriorly and splitting into two arms posteriorly.

The closure cue must be measured on the **mandible only** — the maxillary scan includes the
palate, which fills the central region at every depth. Missing that was the original bug: it
left anterior-detection confidence at a median of 0.00 across Bite2Text, versus ~2.8 after
the fix, and cost 3–10 accuracy points on every downstream field.

Left/right is a true mirror ambiguity that surface geometry cannot resolve. It is settled
empirically against FDI tooth numbers in the Bits2Bites crossbite labels
(`scripts/calibrate_geometry.py`): **+x is the patient's left** (12.5% agreement with the RAS
reading over 32 informative cases, mean margin −25.6°).

### 2. Arch profiles and measurements (`geom/arch.py`, `geom/measures.py`)

Each arch is described in polar coordinates about a pole inside the arch, so every quantity is
a function of arch angle `phi` (0 = anterior midline, sign follows the lateral axis). Per
angular bin we recover the occlusal extreme (cusp tip / incisal edge) and the buccal and
lingual limits of the occlusal band.

The pole is the **midpoint of the ridge's extent, not its median** — a U-shaped point set has
a bimodal lateral distribution, so the median lands on whichever arm carries more vertices and
throws the angular origin off the midline by over a centimetre.

From those profiles: overbite, overjet, midline deviation, curve of Spee, curve of Wilson,
per-side transverse overlap and crossbite extent, cusp interdigitation lag, arch widths and
anterior irregularity. Values outside clinically possible ranges are returned as NaN rather
than as numbers, so the models treat them as missing.

**Validation** — on Bits2Bites, whose 200 cases carry direct occlusal annotations, overbite
separates the bite classes as the textbook says it should: Deep +5.6mm, Normal +2.8mm, Open
+0.2mm; crossbite extent 56° vs 8° for normal. That is what establishes the measurements mean
what they claim.

### 3. Field prediction (`scripts/train_field_models.py`)

Labels come from rule-parsing the clinician narratives (`report/parse.py`, ~98% coverage on
the sagittal fields). Gradient boosting maps measurements to each template field under
patient-level cross-validation:

| Field | CV accuracy | Majority baseline | Lift |
|---|---|---|---|
| constriction | 0.744 | 0.499 | +0.245 |
| overjet | 0.729 | 0.513 | +0.216 |
| crossbite | 0.718 | 0.513 | +0.204 |
| overbite | 0.739 | 0.574 | +0.165 |
| molar_left / molar_right | 0.585 / 0.526 | 0.457 / 0.432 | +0.128 / +0.094 |
| canine_left / canine_right | 0.583 / 0.538 | 0.471 / 0.415 | +0.113 / +0.123 |
| curve of Wilson / Spee | 0.670 / 0.640 | 0.571 / 0.521 | +0.099 / +0.119 |
| crowding_lower | 0.504 | 0.481 | +0.022 |
| **midlines, crowding_upper** | 0.544 / 0.526 | 0.546 / 0.528 | **none → prior used** |

A field is predicted **only** if its model beat the majority baseline. Where geometry cannot
see the finding, the modal value is emitted instead: on a corpus this formulaic, a wrong
specific claim costs RadFact precision while the modal claim is usually right.

The sagittal fields are the weakest (0.53–0.59). Angle classification depends on where an
individual cusp sits relative to an individual groove, which arch-level measurements can only
approximate; closing that gap plausibly requires per-tooth segmentation.

### 4. Renderer (`report/render.py`)

Emits the corpus's six-part narrative in its own idiom. Both sides of the sagittal
relationship are always stated explicitly even when they agree — 42% of the corpus uses the
shorter "bilaterally" form, but the explicit form measures +0.012 captioning higher, because
it shares more 4-grams with the majority phrasing and METEOR-lite rewards token recall.

Generated reports run 98 tokens median against the corpus's 94, so neither BLEU's brevity
penalty nor a length-precision penalty applies.

**Why synthesise rather than retrieve.** An obvious alternative is to emit the training report
whose findings best match the prediction. Measured at the *oracle* level — matching on
ground-truth findings, i.e. the best any such retrieval could do — it loses decisively:

| Strategy (oracle findings, 250 held-out cases) | BLEU-4 | METEOR | Captioning |
|---|---|---|---|
| Findings-matched retrieval of a real report | 0.553 | 0.717 | 0.635 |
| **Our renderer** | 0.611 | 0.775 | **0.693** |

Scoring is against a single reference, which rewards the corpus *mode*; a real report carries
natural variation that costs n-gram overlap without adding correctness.

## Results

> **Reporting convention.** The public leaderboard shows *mean ± std of per-case* scores. The
> evaluator computes both that and a corpus-level aggregate; corpus BLEU-4 runs ~0.04 higher
> because it pools n-gram counts. All figures below are **per-case**, to match the board.

Strict held-out split (`scripts/holdout_validation.py`): 693 train / 296 test, patient-disjoint.
Modal fallbacks and field models are fitted on the training split only — no test case informs
any part of the system.

| Held-out (n=296) | BLEU-4 | METEOR |
|---|---|---|
| prior (no geometry) | 0.4302 ± 0.1411 | 0.6113 ± 0.1301 |
| **geometry (shipped)** | **0.4578 ± 0.1317** | **0.6767 ± 0.1213** |
| oracle (perfect fields) | 0.5555 ± 0.1520 | 0.7594 ± 0.1271 |
| *a real clinician's second report* | *0.2234 ± 0.0773* | *0.5137 ± 0.0846* |

The last row is the calibration anchor. A genuine clinician report, scored against another
clinician's report of the same patient, lands at 0.223 / 0.514 — **inside the public
leaderboard's range** (0.189–0.246 BLEU-4, 0.401–0.457 METEOR, as of 15 Aug 2026). Two things
follow: the measurement pipeline here matches the organisers', and every published entry is
scoring at roughly human-agreement level.

That is the whole thesis of this system. Against a *single* reference, on a corpus this
formulaic, emitting the canonical form beats emitting a different perfectly valid clinical
report. A fluent model that writes like a clinician inherits the clinician's disagreement rate;
a system that writes the corpus mode does not.

### RadFact (the 0.8-weighted clinical half)

Measured with `modal_apps/radfact_judge.py` (Qwen2.5-7B judge, the organisers' prompts and
arithmetic). Because the judge model differs from their `gpt-4o-mini`/GPT-5.2, these are a
**relative** instrument for comparing our own variants, not predicted leaderboard values.

| n=140 | Precision | Recall | RadFact-F1 | Captioning | Final (0.8/0.2) |
|---|---|---|---|---|---|
| geometry (shipped) | 0.436 | 0.485 | 0.459 | 0.580 | **0.483** |
| prior | 0.512 | 0.432 | 0.468 | 0.527 | 0.480 |
| oracle (n=24) | 0.676 | 0.734 | 0.704 | 0.676 | 0.699 |
| *published SOTA (IOS-Qwen, GPT-5.2 judge)* | — | — | *0.372* | — | — |
| *human inter-rater (paper)* | — | — | *0.650* | — | — |

**The prior beats geometry on RadFact-F1 alone** (+0.009), consistently at n=24 and n=140, so
this is a real effect rather than noise. The mechanism is visible in the phrase counts: the
prior's fixed wording is segmented by the judge into 11.0 phrases against geometry's 7.6, and
more of them are individually entailed (5.63 vs 3.46) because generic modal claims are usually
true. But geometry covers **more distinct reference findings** (4.29 vs 3.85 entailed reference
phrases) — it is more informative per case, and the prior partly wins precision through
redundant claims.

Geometry is shipped because it takes the combined score (0.483 vs 0.480) and wins the *live*
leaderboard outright, which ignores RadFact entirely and therefore reduces to captioning
(0.580 vs 0.527). The margin on the clinical metric is small enough that it is worth revisiting
if the organisers' judge segments text differently.

Restricting geometry to only its strongest fields was tested and rejected: thresholding at
0.60/0.65/0.70 out-of-fold accuracy gives captioning 0.558/0.553/0.550, all below full
geometry's 0.580.

## Submission container

`submission/` implements the challenge I/O contract: sockets `3d-lower-teeth-scan`,
`3d-upper-teeth-scan`, `2d-intraoral-photographs` in, `diagnostic-imaging-report.json`
(`{"report": str}`) out. CPU-only — the pipeline is deterministic geometry plus a small
gradient-boosting model, so it needs no accelerator and avoids CUDA image risk.

**It always writes a report.** A missing output scores zero, which is worse than any report we
could otherwise emit, so every stage degrades to the prior-based report rather than raising.
`submission/test_local.py` verifies this against a simulated platform layout including an
empty mesh (the real F5500 failure in the training set), no meshes at all, and the alternative
flat input layout.

## Reproducing

```bash
python scripts/analyze_reports.py                       # corpus statistics
python scripts/extract_geometry_features.py --root data/raw/bite2text --output artifacts/geom/bite2text_features.csv
python scripts/train_field_models.py                    # per-field models + bundle
python scripts/evaluate_system.py                       # out-of-fold end-to-end scores
python submission/test_local.py                         # container contract + robustness
modal run modal_apps/radfact_judge.py --predictions artifacts/eval/predictions_geometry.json \
                                     --references  artifacts/eval/references.json
```
