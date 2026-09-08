#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python || ! -d web/node_modules ]]; then
  echo "Install dependencies first: uv sync && npm --prefix web ci"
  exit 1
fi
exec .venv/bin/python scripts/dev.py
