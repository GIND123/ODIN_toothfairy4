#!/usr/bin/env bash
# Export the built image as the .tar.gz Grand Challenge expects.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
IMAGE=${IMAGE:-odin2026-bite2text-geometry}
OUT="$SCRIPT_DIR/${IMAGE}.tar.gz"
"$SCRIPT_DIR/do_build.sh"
docker save "$IMAGE" | gzip -c > "$OUT"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
