# Data audit protocol

The audit runs before any split or model fitting. It produces `manifest.csv`, `summary.json`, machine-readable issue tables, aggregate PNG charts, and a self-contained HTML report.

Checks include:

1. **Inventory and linkage:** case count, files per modality, required upper/lower IOS presence, photo counts, report coverage, orphan files, and unexpected extensions.
2. **Integrity:** byte size, streaming SHA-256, image decoding and RGB/shape metadata, mesh loading, vertex/face counts, bounds, watertightness, and empty/degenerate assets.
3. **Text:** encoding/read errors, characters/words/lines, empty reports, rough language hints, exact duplicate content, and suspiciously short reports. Raw text is never copied into audit artifacts.
4. **Leakage:** exact binary duplicates across patients, exact normalized-report duplicates, and patient-level grouping requirements. Near-duplicate image/mesh analysis is deliberately opt-in because clinical images must not leave the authorized environment.
5. **Distribution:** per-modality missingness, image resolution/aspect ratio, mesh complexity, report length, and case completeness.

## Identifier handling

Set `BITE2TEXT_ID_SALT` to a private stable value. The manifest pseudonymizes discovered case tokens with salted SHA-256. Without it, a development-only default is used and a warning is recorded. Keep any mapping and raw logs out of version control.

## Interpretation

A “pass” means structural fitness for this pipeline, not clinical correctness. Review severe outliers with an authorized clinician in the governed data environment. Never publish individual images, meshes, case identifiers, or verbatim reports.

