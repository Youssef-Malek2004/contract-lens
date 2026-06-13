#!/usr/bin/env bash
# setup_models.sh — Full vllm-mlx setup for ContractLens.
#
# Order of operations:
#   1. Start Orchestrator server    (Qwen3-4B,           port 8001)  ← background
#   2. Start Base model server      (Qwen3-1.7B,          port 8003)  ← background
#   3. Merge LoRA adapter + convert to MLX 4-bit                      ← foreground, blocks
#   4. Start NLI server             (merged 1.7B,         port 8002)  ← background
#   5. Wait for all three servers to report ready
#   6. Run test_vllm_endpoints.py
#
# Prerequisites:
#   conda env 'genai-ms2'  — Python deps (transformers, peft, mlx-lm, etc.)
#   conda env 'serve-lm'   — vllm-mlx  (run ServeLM/setup.sh once to provision)
#
# Usage:
#   cd contract-lens
#   ./setup_models.sh
#   ./setup_models.sh --skip-merge   # if mlx-nli-1.7b-4bit/ already exists
#
# Servers stay running after this script exits.
# PIDs are written to .server_pids for stop_servers.sh.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVELM_DIR="$SCRIPT_DIR/../../ServeLM"
LOGS_DIR="$SCRIPT_DIR/logs"
NLI_MLX_DIR="$SCRIPT_DIR/mlx-nli-1.7b-4bit"
PID_FILE="$SCRIPT_DIR/.server_pids"

# ── Model IDs ─────────────────────────────────────────────────────────────────
ORCH_MODEL="mlx-community/Qwen3-4b-4bit"
BASE_MODEL="mlx-community/Qwen3-1.7b-4bit"
NLI_MODEL_ID="Youssef-Malek/contractnli-qwen3-1.7b-mlx-4bit"  # registered API name

# ── Ports ─────────────────────────────────────────────────────────────────────
ORCH_PORT=8001
NLI_PORT=8002
BASE_PORT=8003

# ── Args ──────────────────────────────────────────────────────────────────────
SKIP_MERGE=0
for arg in "$@"; do
  [[ "$arg" == "--skip-merge" ]] && SKIP_MERGE=1
done

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_model() {
  local url="$1" model_id="$2" name="$3"
  local max_wait=180 elapsed=0
  printf "[$(date '+%H:%M:%S')] Waiting for %-32s" "$name …"
  while [[ $elapsed -lt $max_wait ]]; do
    if python3 - "$url" "$model_id" <<'PYEOF' 2>/dev/null
import sys, json, urllib.request
url, model_id = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"{url}/models", timeout=2) as r:
        data = json.loads(r.read())
    served = [m.get("id") for m in data.get("data", [])]
    sys.exit(0 if model_id in served else 1)
except Exception:
    sys.exit(1)
PYEOF
    then
      echo " ready."
      return 0
    fi
    printf "."
    sleep 3
    elapsed=$((elapsed + 3))
  done
  echo " TIMED OUT after ${max_wait}s"
  log "  Check log: $LOGS_DIR/server_${port}.log"
  return 1
}

# ── Preflight ─────────────────────────────────────────────────────────────────
[[ -d "$SERVELM_DIR" ]] || die "ServeLM directory not found at $SERVELM_DIR"
[[ -f "$SERVELM_DIR/serve.sh" ]] || die "serve.sh not found in $SERVELM_DIR"

command -v conda &>/dev/null || die "conda not found — install Miniconda/Anaconda first"
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate genai-ms2 || die "conda env 'genai-ms2' not found — run the setup from README first"

mkdir -p "$LOGS_DIR"

echo "================================================================"
echo "  ContractLens — Model Setup"
echo "  Logs: $LOGS_DIR/"
echo "================================================================"

# ── Step 1: Orchestrator ──────────────────────────────────────────────────────
log "Starting Orchestrator (Qwen3-4B) on port $ORCH_PORT → logs/server_${ORCH_PORT}.log"
MODEL="$ORCH_MODEL" PORT="$ORCH_PORT" bash "$SERVELM_DIR/serve.sh" \
  --reasoning-parser qwen3 \
  &> "$LOGS_DIR/server_${ORCH_PORT}.log" &
ORCH_PID=$!
disown $ORCH_PID

# ── Step 2: Base model ────────────────────────────────────────────────────────
log "Starting Base model (Qwen3-1.7B) on port $BASE_PORT → logs/server_${BASE_PORT}.log"
MODEL="$BASE_MODEL" PORT="$BASE_PORT" bash "$SERVELM_DIR/serve.sh" \
  --reasoning-parser qwen3 \
  &> "$LOGS_DIR/server_${BASE_PORT}.log" &
BASE_PID=$!
disown $BASE_PID

# ── Step 3: Merge + convert NLI model ────────────────────────────────────────
if [[ -d "$NLI_MLX_DIR" ]]; then
  log "Merged model already at '$NLI_MLX_DIR' — skipping merge"
  log "  Delete it and re-run to force a re-merge"
elif [[ $SKIP_MERGE -eq 1 ]]; then
  die "--skip-merge passed but '$NLI_MLX_DIR' does not exist — run without --skip-merge first"
else
  log "Ensuring mlx-lm is up to date (Qwen3 requires a recent version)…"
  pip install --upgrade mlx-lm --quiet
  log "Merging LoRA adapter and converting to MLX 4-bit…"
  python "$SCRIPT_DIR/merge_adapter.py" \
    --convert \
    --mlx-output-dir "$NLI_MLX_DIR"
fi

[[ -d "$NLI_MLX_DIR" ]] || die "MLX model not found at '$NLI_MLX_DIR'"

# ── Step 4: NLI server ────────────────────────────────────────────────────────
# --served-model-name overrides the registered API model ID so it matches
# NLI_VLLM_ID in src/loaders/_constants.py regardless of the local path.
log "Starting NLI Core (merged 1.7B) on port $NLI_PORT → logs/server_${NLI_PORT}.log"
# No --reasoning-parser: the NLI model was fine-tuned without thinking and has
# never seen <think> tokens. The parser would route all output to reasoning_content
# indefinitely. Without it, all generated tokens go directly to delta.content.
MODEL="$NLI_MLX_DIR" PORT="$NLI_PORT" bash "$SERVELM_DIR/serve.sh" \
  --served-model-name "$NLI_MODEL_ID" \
  &> "$LOGS_DIR/server_${NLI_PORT}.log" &
NLI_PID=$!
disown $NLI_PID

# ── Save PIDs ─────────────────────────────────────────────────────────────────
cat > "$PID_FILE" <<EOF
ORCH_PID=$ORCH_PID
BASE_PID=$BASE_PID
NLI_PID=$NLI_PID
EOF
log "PIDs saved to $PID_FILE — run ./stop_servers.sh to stop all three"

# ── Step 5: Wait for all servers ──────────────────────────────────────────────
echo ""
log "Waiting for servers to finish loading models…"
wait_for_model "http://localhost:${ORCH_PORT}/v1" "$ORCH_MODEL"   "Orchestrator  :${ORCH_PORT}"
wait_for_model "http://localhost:${BASE_PORT}/v1" "$BASE_MODEL"   "Base model    :${BASE_PORT}"
wait_for_model "http://localhost:${NLI_PORT}/v1"  "$NLI_MODEL_ID" "NLI Core      :${NLI_PORT}"

# ── Step 6: Run tests ─────────────────────────────────────────────────────────
echo ""
log "Running endpoint tests…"
echo "================================================================"
cd "$SCRIPT_DIR/.." && python -m unittest tests.test_vllm_endpoints -v
echo "================================================================"

echo ""
log "All done. Three servers are running in the background."
echo ""
echo "  Orchestrator  :${ORCH_PORT}  (${ORCH_MODEL})"
echo "  NLI Core      :${NLI_PORT}  (${NLI_MODEL_ID})"
echo "  Base model    :${BASE_PORT}  (${BASE_MODEL})"
echo ""
echo "  Logs : $LOGS_DIR/"
echo "  Stop : ./stop_servers.sh"
