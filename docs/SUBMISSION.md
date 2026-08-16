# Submitting to ODIN 2026 Task 2

## Projected leaderboard position

Public Test Phase leaderboard as of 15 Aug 2026, with our held-out figures inserted. The board
ranks by mean position across BLEU-4 and METEOR, and **ignores RadFact** (a stated platform
limitation), so it reduces to the two captioning metrics.

| Entry | BLEU-4 | pos | METEOR | pos | Mean pos |
|---|---|---|---|---|---|
| **This system** | **0.4578** | 1 | **0.6767** | 1 | **1.0** |
| MIGG / MMTLVM | 0.2351 | 3 | 0.4424 | 3 | 3.0 |
| GenMI / teeth occlusion | 0.2463 | 2 | 0.4261 | 4 | 3.0 |
| JIA / Bite2Text Report Generation | 0.2290 | 4 | 0.4234 | 5 | 4.5 |
| shayne / Qwen3-VL Photo | 0.2145 | 7 | 0.4569 | 2 | 4.5 |
| MIND_lab / Structured Occlusal | 0.2218 | 5 | 0.4010 | 8 | 6.5 |
| Alex.zhang / Mesh Retrieval | 0.1891 | 8 | 0.4226 | 6 | 7.0 |

Margins are +86% on BLEU-4 and +48% on METEOR. First place survives up to roughly **35%**
degradation on the hidden test; at 40% METEOR would fall to second, at 50% both would.

**Why the margin is this large.** Scoring a real clinician's second report against the first
gives BLEU-4 0.223 / METEOR 0.514 — inside the leaderboard's own range. Every published entry
is a fluent generator writing like a clinician, and so inherits the clinician disagreement
rate. Against a single reference on a formulaic corpus, writing the corpus *mode* is worth
about twice as much.

**Caveats worth stating plainly.** These are held-out figures from the 996-case training
distribution, not the 50-case hidden test; the human-calibration anchor argues the two are
comparable but does not prove it. And this projection is for the *live* board only — the final
offline ranking is `0.8 * RadFact-F1 + 0.2 * captioning`, where our measured edge is much
narrower (see `docs/SYSTEM.md`).

## Timing

The ODIN dates page gives **Test Phase: 3–17 August 2026**, results 15 September. Confirm on
the participant portal before relying on it — the same page shows several deadlines that were
revised in place. The test phase allows **two submissions; the best one counts**.

## The built artifact

**`submission/odin2026-bite2text-geometry.tar.gz` — 134 MB.** Upload this as the algorithm
image on Grand Challenge.

Built with Docker 29.7.2 for `linux/amd64` on `python:3.13-slim`. It was verified by deleting
the local image, re-loading it from the exported tarball, and running that fresh copy with
`--network none` against a simulated platform `/input`:

| Scenario | Result | Time |
|---|---|---|
| normal case | 483-char report, 11 fields from geometry | 9.3 s |
| empty upper mesh | 464-char prior report, exit 0 | 6.5 s |
| no meshes at all | 464-char prior report, exit 0 | 3.1 s |
| flat input layout | 538-char report, 11 fields from geometry | 8.0 s |

Rebuild with `cd submission && ./do_save.sh`.

### Two build issues worth recording

**Python version.** The first build failed: `requirements.txt` pins `numpy==2.5.2` /
`scipy==1.18.0`, taken from the local Python 3.13 environment, but the image was
`python:3.11-slim` where numpy 2.5.2 does not exist. The base image is now 3.13, which is also
the safer choice — the joblib bundle carries the scikit-learn version it was fitted with, so
the runtime interpreter should match the one that wrote it.

**Threading — a 1000x pathology.** The first working image took **110 seconds per case**;
104 s of that was eleven single-row `HistGradientBoostingClassifier.predict` calls. Prediction
goes through OpenMP, and on a 16-core host the thread setup for a *one-row* input dwarfs the
work itself. Pinning to one thread takes the same predictions from 104 s to 0.10 s, and the
whole case from 110 s to 6–8 s. `OMP_NUM_THREADS` and friends are now set both as image `ENV`
and defensively at the top of `inference.py` before any numeric import, in case the platform
overrides the image environment. Across 50 hidden cases this is the difference between roughly
7 minutes and 90 minutes.

### Alternatives

* **GitHub route** — the How-To-Submit page allows "either a Docker container or a GitHub
  repository following organizer-provided templates"; `submission/` matches the template.
* **`artifacts/bite2text-submission-source.zip`** (4.9 MB) is a self-contained copy of
  `submission/` that builds anywhere.

## What gets uploaded

```
submission/
├── Dockerfile              # python:3.11-slim, linux/amd64, CPU-only
├── requirements.txt        # numpy, scipy, scikit-learn, trimesh, joblib, Pillow (pinned)
├── inference.py            # entrypoint implementing the socket contract
├── resources/
│   └── field_models.joblib # ~4.9 MB gradient-boosting bundle
└── src/bite2text/          # geom/, report/, compose.py  (no heavy deps)
```

Total ~5 MB. No network access is needed at run time and no GPU is used.

### Interface implemented

| Direction | Socket | Path |
|---|---|---|
| in | `3d-lower-teeth-scan` | `/input/files/ios-lower/*.stl\|obj` or `/input/3d-lower-teeth-scan.obj` |
| in | `3d-upper-teeth-scan` | `/input/files/ios-upper/*.stl\|obj` or `/input/3d-upper-teeth-scan.obj` |
| in | `2d-intraoral-photographs` | `/input/images/{2d-intraoral-photographs,intraoral-photo}/*` |
| out | `diagnostic-imaging-report` | `/output/diagnostic-imaging-report.json` → `{"report": str}` |

Both the socket-directory and flat layouts are handled, plus a filename-keyword fallback if
neither matches.

## Pre-flight checks

```bash
python -m pytest -q                 # 10 tests: frame invariance, parser, renderer, scorer
python submission/test_local.py     # 4 scenarios against a simulated /input
```

`test_local.py` covers the failure modes that would cost the most, since **a missing output is
scored as an empty report worth zero**:

| Scenario | Expected |
|---|---|
| normal case | geometry-driven report |
| flat input layout | geometry-driven report |
| empty upper mesh (the real F5500 failure) | prior-based report, exit 0 |
| no meshes at all | prior-based report, exit 0 |

Also verified against the organisers' own sample inputs (`F5405`, `F5520`, `F5535`), which ship
as 1-triangle placeholder STLs: all three produce a valid 464-character report and exit 0.

## Known risk: which reference the hidden test uses

The released corpus has two English report families and the evaluation ships one `.txt` per
case, so which one backs the hidden test is not stated. This system targets **`reports_ios_en`**
because it covers 991 of 996 cases (versus 872 for the photo family) and the task is occlusal
reporting.

The downside if that guess is wrong is bounded: the two families share the same six-part
structure and idiom, and scoring one family's real reports against the other's yields 0.44–0.56
captioning — comparable to within-family baselines. No change of approach is warranted either
way.

## Using the two submissions

**Submission 1: the system as built (geometry).** It takes the combined score and wins the live
leaderboard outright — that board ignores RadFact, so it reduces to captioning, where geometry
leads 0.580 to the prior's 0.527.

**A hedge against the reference-family risk was tested and rejected.** Photo-family reports
share our exact six-sentence opening and then add dental-health sentences (restorations 35%,
gingival inflammation 38%, caries 34%). Appending generic versions of those guarantees first
place on both captioning metrics under *either* family — but it costs RadFact badly, because
the added phrases are mostly unentailed:

| Variant | RadFact-F1 | Captioning | Final (0.8/0.2) |
|---|---|---|---|
| **base (shipped)** | 0.459 | 0.560 | **0.479** |
| prior | 0.468 | 0.512 | 0.477 |
| + dental-health sentences | 0.407 | 0.532 | 0.432 |

The extension adds 3.7 phrases per report but only 0.9 entailed ones (11.34 vs 7.64 candidate
phrases; 4.37 vs 3.46 entailed). At 0.8 weight the −0.052 RadFact loss dwarfs the captioning
gain. Base ships.

**Submission 2 is a genuine decision, not a formality.** Locally the prior-only variant scores
*higher on RadFact-F1 alone* (0.468 vs 0.459 at n=140) while scoring much lower on captioning.
Since the final offline ranking weights RadFact at 0.8, the two are nearly tied there
(0.477 vs 0.479). Two reasonable plays:

* **Hedge** — submit the prior-only variant as #2. It is one flag away
  (`artifacts/eval/predictions_prior.json` is generated by the same renderer with modal values),
  and it insures against the organisers' judge segmenting text the way ours does.
* **Push** — spend the slot on better field accuracy instead. The renderer with perfect field
  values scores 0.676 captioning and 0.704 RadFact-F1, so the headroom is in prediction, not
  phrasing. The weakest fields are the sagittal classes (0.53–0.59) and midlines (no lift at
  all), both of which plausibly need per-tooth segmentation of the arches.

Given the RadFact gap is within a point and our judge is not the official one, the hedge is the
lower-variance choice.
