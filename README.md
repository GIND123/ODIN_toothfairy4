# ODIN 2026 — Bite2Text (Task 2)

Geometry-conditioned orthodontic report generation from registered intraoral scans.

Instead of asking a vision-language model to guess occlusal findings from photographs, this
system **measures** them. The released scans are mutually registered *in occlusion*, so
overbite, overjet, midline deviation, transverse relationships and the occlusal curves are
directly computable from the surfaces; a small model maps those measurements onto the clinical
fields, and a renderer emits them in the corpus's own six-part narrative idiom.

See **[docs/SYSTEM.md](docs/SYSTEM.md)** for the full design, the reverse-engineered scoring
function, and the measured results.

## Headline findings

* **The scoring function is not what it looks like.** On Grand Challenge the evaluator uses
  its own local BLEU-4/METEOR fallbacks, and its "METEOR" is exact-match only with an F-mean
  of `10PR/(R+9P)` — recall weighted 9:1. `src/bite2text/eval/gc_metrics.py` reproduces it.
* **Two clinicians agree on 47% of findings** for the same patient (token Jaccard 0.472,
  finding Jaccard 0.469, n=498 paired reports). That is the real ceiling.
* **The corpus is formulaic enough that a single fixed report scores 0.547 captioning —
  higher than one clinician scores against another (0.382).** Structure matters more than
  patient specificity; correct field values add roughly +0.20 on top.
* **The released scans are not in the frame the paper states.** Bits2Bites is anterior=+Y,
  Bite2Text is anterior=−Y, and some cases put superior on Y. The frame is re-derived per
  case; assuming RAS silently corrupts every measurement.

## Layout

```
src/bite2text/
  geom/          frame canonicalisation, arch profiling, clinical measurements
  report/        narrative parser, template schema, renderer
  eval/          byte-faithful replica of the challenge scorer
  compose.py     inference path: geometry -> fields -> report
scripts/         corpus analysis, feature extraction, training, evaluation
submission/      Grand Challenge container (CPU-only) + robustness tests
modal_apps/      RadFact judge on GPU (vLLM + the organisers' radfact_lite)
docs/SYSTEM.md   design, scoring analysis, measured results
```

## Quick start

```bash
pip install -e ".[dev]"

# Data: place the authorised Bite2Text extraction under data/raw/bite2text/<case>/...
python scripts/analyze_reports.py
python scripts/extract_geometry_features.py --root data/raw/bite2text \
       --output artifacts/geom/bite2text_features.csv
python scripts/train_field_models.py
python scripts/evaluate_system.py

# Container
python submission/test_local.py      # contract + failure-mode tests
cd submission && ./do_save.sh        # builds and exports the .tar.gz for upload
```

## Data note

Bite2Text is account-gated and must be downloaded by an authorised user from the
[official page](https://ditto.ing.unimore.it/bite2text/). No patient data, mesh renders or
report text is committed to this repository.

## Artifact backup

Models, submission archives, evaluation summaries, logs, and plots can be backed up to a
private Hugging Face repository. Copy `.env.example` to the ignored `.env`, set the lowercase
`hf` variable to a fine-grained token, then run:

```bash
pip install huggingface-hub
python scripts/huggingface_artifacts.py upload
python scripts/huggingface_artifacts.py download --output restored_artifacts
```

Set `HF_REPO_ID` in `.env` to override the default repository. The backup intentionally
excludes raw clinical data, photographs, per-case reports/predictions, and identifier-bearing
feature tables.

`Bits2Bites` is a **different** dataset (200 cases, arch pairs with direct occlusal
annotations, no photographs and no reports). It is not the Task 2 training set, but its
labels map onto the same findings over the same input, so it is used here to *validate* that
the geometry measurements mean what they claim and to resolve the left/right frame ambiguity.

## Citation

Lumetti, L., Rizzo, F., Cremonini, F., Candeloro, E., Lombardo, L., Grana, C., & Bolelli, F.
(2026). *Do Multimodal LLMs Understand Intraoral Dental Data? Dataset, Platform, and
Baselines.* ECCV.

Not medical software; not clinical advice.
