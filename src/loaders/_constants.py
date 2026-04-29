# HuggingFace model IDs
ORCHESTRATOR_ID  = "Qwen/Qwen3-4B"
NLI_BASE_ID      = "Qwen/Qwen3-1.7B"
NLI_ADAPTER_ID   = "Youssef-Malek/contractnli-vast-ai-qwen3-1.7b"

# vllm-mlx model IDs (MLX 4-bit quantized)
# NLI_VLLM_MODEL_ID requires merge_adapter.py to be run first — see AGENTS.md
ORCHESTRATOR_VLLM_ID = "mlx-community/Qwen3-4b-4bit"
NLI_VLLM_ID          = "Youssef-Malek/contractnli-qwen3-1.7b-mlx-4bit"
BASE_VLLM_ID         = "mlx-community/Qwen3-1.7b-4bit"

# Default vllm server URLs (one process per model, each on its own port)
ORCHESTRATOR_VLLM_URL = "http://localhost:8001/v1"
NLI_VLLM_URL          = "http://localhost:8002/v1"
BASE_VLLM_URL         = "http://localhost:8003/v1"
