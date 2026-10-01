#!/usr/bin/env bash
# Jeff v1.2 0.8B on this Mac with the MLX backend, every adapter in models/adapters loaded beside the base.
# Download them first: uv run jeffref fetch --base --tasks guard,triage,support-intents,tools,ground
# Requests pick the adapter by name in the "model" field. Adapters added later: download them, then
#   curl -X POST http://127.0.0.1:8765/v1/adapters/reload
set -euo pipefail
cd "$(dirname "$0")/.."
export JEFF_BACKEND=mlx JEFF_CHECKPOINT=models/jeff-v1.2-0.8b JEFF_ADAPTERS=models/adapters PORT=8765
export JEFF_MLX_CACHE_LIMIT_GB=${JEFF_MLX_CACHE_LIMIT_GB:-1}
exec uv run --no-sync python -m jeffref.serve_jeff
