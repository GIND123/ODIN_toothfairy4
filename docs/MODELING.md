# Modeling blueprint

The included retrieval baseline validates plumbing: it builds TF-IDF features from available source reports and retrieves a training English target report. It cannot exploit 2D/3D pixels and is not intended to be competitive.

A competition system should use patient-grouped validation and ablate:

- 2D encoder over standardized views with view-aware aggregation;
- 3D point/mesh encoder over upper and lower jaws with scale/orientation normalization;
- gated cross-modal fusion with missing-modality masks;
- a clinically constrained report decoder or structured finding extraction followed by deterministic verbalization;
- conservative confidence/abstention and hallucination checks.

Select checkpoints primarily with a locally reproducible clinical factuality proxy, then BLEU-4/METEOR. Keep a frozen validation cohort; tune no decisions on the public leaderboard. Document preprocessing, prompts, model/license provenance, seeds, hardware, and per-case latency.

