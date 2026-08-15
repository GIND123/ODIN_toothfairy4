# ODIN 2026 — Bite2Text starter kit

End-to-end, privacy-conscious tooling for Task 2 of ODIN 2026: index the multimodal dataset, audit its 3D scans, photographs, and bilingual reports, train a leakage-safe retrieval baseline, and validate a submission file.

> This repository contains code only. Bite2Text is account-gated and must be downloaded by an authorized user from the [official page](https://ditto.ing.unimore.it/bite2text/). Do not commit patient data or generated thumbnails.

## Competition snapshot (verified 2026-08-16)

- Public release: **1,000 patient cases** according to the official Bite2Text page.
- Inputs: upper/lower IOS meshes plus standardized RGB intraoral photographs.
- Targets: clinician-authored Italian and English reports, separately for IOS and photographs.
- Hidden test set: the supplied challenge text says **50 cases**; its labels are not public.
- Phase 1 metrics: RadFact Logical precision/recall → F1 (clinical); BLEU-4 and METEOR → mean (captioning); `final = 0.8 * clinical + 0.2 * captioning`.
- Phase 2: up to seven leading methods undergo blinded surgeon pairwise review and Elo-style ranking.
- Missing output is treated as an empty report and scores zero.
- The live leaderboard reportedly omits RadFact Logical scores because of platform limitations; final test ranking is offline.

The prompt supplied with this repository also says “Training 2,000 1,000 patient cases,” which is internally inconsistent. The official dataset page consistently states 1,000 cases, so this project treats 1,000 as the expected public count and surfaces any mismatch in the audit. Rules may change: confirm the [challenge site](https://odin2026.grand-challenge.org/) and [official dataset page](https://ditto.ing.unimore.it/bite2text/) before submitting. The precise Grand Challenge container/output interface was not publicly retrievable during scaffolding, so `config/submission.yaml` is intentionally isolated and must be reconciled with the participant portal.

## Quick start

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# bash: source .venv/bin/activate
pip install -e ".[dev]"

# Place the authorized extraction under data/raw/bite2text (or pass --data-root).
bite2text index --data-root data/raw/bite2text --output artifacts/manifest.csv
bite2text audit --manifest artifacts/manifest.csv --output-dir artifacts/audit
bite2text train-baseline --manifest artifacts/manifest.csv --output artifacts/baseline.joblib
bite2text predict --manifest artifacts/test_manifest.csv --model artifacts/baseline.joblib --output artifacts/submission.json
bite2text validate-submission --input artifacts/submission.json --manifest artifacts/test_manifest.csv
```

To exercise the entire workflow without protected data:

```bash
python scripts/make_fixture.py --output data/fixture
bite2text all --data-root data/fixture --output-dir artifacts/fixture_run
pytest
```

The generated `artifacts/.../audit/report.html` is self-contained and contains aggregate charts only; it never embeds clinical photographs, mesh renders, report text, or case identifiers.

## Expected extraction layout

The indexer supports either modality-first directories (the official description) or patient-first directories. Filenames must share a stable patient token; configure extraction regexes in `config/dataset.yaml` when the downloaded naming convention differs.

```text
data/raw/bite2text/
├── ios/
├── intraoral-photo/
├── reports_ios_it/
├── reports_ios_en/
├── reports_intraoral-photo_it/
└── reports_intraoral-photo_en/
```

The manifest stores relative paths and a salted SHA-256 pseudonym, not the raw patient token. See [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md), [docs/MODELING.md](docs/MODELING.md), and [docs/SUBMISSION.md](docs/SUBMISSION.md).

## Reproducibility and safety

- Patient-level splits only; never split individual images or jaws.
- Duplicate hashes and near-duplicate text groups are reported before modeling.
- Missing/corrupt files, image properties, mesh topology, language/report statistics, and cross-modal coverage are audited.
- Seeds and configuration snapshots are written beside artifacts.
- Reports are aggregate-only by default because the source is sensitive clinical data.
- The included baseline is a pipeline smoke test, not medical software and not clinical advice.

## Citation

Dataset authors request citation of: Lumetti, L., Rizzo, F., Cremonini, F., Candeloro, E., Luca, L., Grana, C., & Bolelli, F. (2026). *Do Multimodal LLMs Understand Intraoral Dental Data? Dataset, Platform, and Baselines.* ECCV. Verify the final BibTeX on the official page before publication.

