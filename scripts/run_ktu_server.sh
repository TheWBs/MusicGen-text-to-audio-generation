#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MUSICGEN_MODEL_ID="${MUSICGEN_MODEL_ID:-facebook/musicgen-small}"
export MUSICGEN_OUTPUT_DIR="${MUSICGEN_OUTPUT_DIR:-outputs}"
export MUSICGEN_MAX_DURATION_SECONDS="${MUSICGEN_MAX_DURATION_SECONDS:-20}"

python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"
