#!/usr/bin/env bash
# stop_servers.sh — Stop all three vllm-mlx servers started by setup_models.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.server_pids"

[[ -f "$PID_FILE" ]] || { echo "No PID file found at $PID_FILE — servers may not be running."; exit 0; }

# shellcheck disable=SC1090
source "$PID_FILE"

for var in ORCH_PID BASE_PID NLI_PID; do
  pid="${!var:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $var (pid $pid)…"
    kill "$pid"
  else
    echo "$var (pid ${pid:-?}) is not running — skipping"
  fi
done

rm -f "$PID_FILE"
echo "Done."
