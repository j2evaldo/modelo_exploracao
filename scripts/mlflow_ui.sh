#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/mlflow" ui \
  --backend-store-uri "sqlite:///$ROOT/mlflow.db" \
  --host 127.0.0.1 --port "${PORT:-5000}" "$@"
