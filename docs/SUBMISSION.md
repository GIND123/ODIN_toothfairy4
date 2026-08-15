# Submission checklist

1. Re-read the authenticated “How to Submit” page and reconcile `config/submission.yaml` with its exact filename, archive, JSON schema, container interfaces, timeout, and hardware limits.
2. Produce exactly one non-empty English report for every hidden case and no additional cases.
3. Run `bite2text validate-submission` against the test manifest.
4. Verify UTF-8 encoding, deterministic inference, offline execution, model-weight availability, and a clean container build.
5. Inspect logs for identifiers or protected clinical content before packaging.
6. Record the git commit, image digest, config, seed, and validation output.

The validator enforces the repository’s configurable schema; it does not certify compliance with rules unavailable outside the participant portal.

