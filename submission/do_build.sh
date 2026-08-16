#!/usr/bin/env bash
# Build the Bite2Text submission image. Run from the submission/ directory.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
IMAGE=${IMAGE:-odin2026-bite2text-geometry}
docker build --platform=linux/amd64 -t "$IMAGE" "$SCRIPT_DIR"
echo "Built $IMAGE"
