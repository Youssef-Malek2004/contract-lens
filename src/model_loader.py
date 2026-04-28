"""
Central model loading for all agents.
Handles device detection and quantization automatically.
All agents call load_model() or load_peft_model() — never handle device/quant directly.

Device priority:  MPS (Apple Silicon) → CUDA (Nvidia) → CPU
Quantization:     CUDA → BitsAndBytesConfig NF4 4-bit
                  MPS  → dtype=torch.float16
                  CPU  → dtype=torch.float16 (fallback only)

Remote mode:
    load_orchestrator(remote=True) does NOT download or load weights.
    It expects an OpenAI-compatible vllm-mlx server already running at
    VLLM_BASE_URL serving VLLM_ORCHESTRATOR_MODEL_ID. See
    ../serving-local-models/serve-qwen3.sh.
    NLI (PEFT) cannot be served this way — adapters are LoRA on a
    non-quantized base, vllm-mlx serves a pre-quantized merged model.
"""
import json
import urllib.error
import urllib.request

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Model IDs ────────────────────────────────────────────────────────────────
ORCHESTRATOR_ID = "Qwen/Qwen3-4B"    # pure transformer, full MPS speed, thinking + tool-calling
NLI_BASE_ID     = "Qwen/Qwen3-1.7B"
NLI_ADAPTER_ID  = "Youssef-Malek/contractnli-vast-ai-qwen3-1.7b"

# ── vllm-mlx remote serving ──────────────────────────────────────────────────
VLLM_BASE_URL                = "http://localhost:8001/v1"
VLLM_ORCHESTRATOR_MODEL_ID   = "mlx-community/Qwen3-4b-4bit"  # lowercase 'b' is intentional


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _quantization_config(device: str):
    """
    CUDA → BitsAndBytesConfig NF4 4-bit (~1 GB for 1.7B, ~2.5 GB for 4B).
           CUDA has native int4 GEMM kernels — genuinely faster and smaller.

    MPS  → None (float16, ~3.5 GB for 1.7B, ~8 GB for 4B).
           MPS has no native int4 kernels — quantization adds startup cost
           with no inference speedup. float16 is the right choice on MPS.

    CPU  → None (float16, fallback only).
    """
    if device == "cuda":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None  # MPS + CPU: float16 via dtype kwarg — fast load, fast inference


def load_model(
    model_id: str,
    device: str = None,
) -> tuple:
    """
    Load a standard (non-PEFT) causal LM with the right quantization for the device.
    Returns (model, tokenizer). Model is in eval mode.

    Usage:
        model, tokenizer = load_model(ORCHESTRATOR_ID)
        model, tokenizer = load_model("Qwen/Qwen3-1.7B", device="cuda")
    """
    if device is None:
        device = get_device()

    quant_cfg = _quantization_config(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if isinstance(quant_cfg, BitsAndBytesConfig):
        # bitsandbytes handles device placement via device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map=device,
            low_cpu_mem_usage=True,
        )
    elif quant_cfg is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)

    model.eval()
    return model, tokenizer


def load_peft_model(
    base_model_id: str,
    adapter_id: str,
    device: str = None,
) -> tuple:
    """
    Load a base model then apply a LoRA adapter on top.
    Tokenizer is loaded from the adapter repo (has the patched chat template).
    Returns (peft_model, tokenizer). Model is in eval mode, adapter ON.

    Toggle adapter at call site:
        with model.disable_adapter():   # adapter OFF — base model behaviour
            output = model.generate(...)
        # adapter back ON after context exit

    Usage:
        model, tokenizer = load_peft_model(NLI_BASE_ID, NLI_ADAPTER_ID)
    """
    if device is None:
        device = get_device()

    quant_cfg = _quantization_config(device)

    if isinstance(quant_cfg, BitsAndBytesConfig):
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant_cfg,
            device_map=device,
            low_cpu_mem_usage=True,
        )
    elif quant_cfg is not None:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant_cfg,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)

    # Adapter repo has the patched chat template — load tokenizer from there.
    # is_quantized=True tells PEFT not to re-apply the bitsandbytes quantization
    # config saved in adapter_config.json (which was written by unsloth on CUDA).
    tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    model = PeftModel.from_pretrained(base, adapter_id, is_quantized=(device == "cuda"))
    model.eval()
    return model, tokenizer


# ── Convenience loaders ───────────────────────────────────────────────────────

def load_orchestrator(device: str = None, remote: bool = False) -> tuple:
    """
    Load Qwen3-4B (orchestrator + conversation agent model).
    Use enable_thinking=True in apply_chat_template at generate time.

    remote=True  → no weights are downloaded or loaded into memory. Returns
                   (RemoteOrchestrator, tokenizer) where RemoteOrchestrator
                   talks to a vllm-mlx server at VLLM_BASE_URL. Raises
                   RuntimeError if the server is unreachable or not serving
                   VLLM_ORCHESTRATOR_MODEL_ID.
    remote=False → original local load via load_model().
    """
    if remote:
        return _load_remote_orchestrator()
    return load_model(ORCHESTRATOR_ID, device)


# ── Remote orchestrator (vllm-mlx) ────────────────────────────────────────────

class RemoteOrchestrator:
    """
    Thin client over an OpenAI-compatible vllm-mlx server.

    Mimics just enough of the local model interface that ConversationAgent
    needs: a streaming chat endpoint. Tokenization/templating happens
    server-side, so callers pass the OpenAI-style messages list directly.
    """

    def __init__(self, base_url: str, model_id: str) -> None:
        self.base_url = base_url
        self.model_id = model_id

    def chat_stream(
        self,
        messages: list,
        max_new_tokens: int,
        enable_thinking: bool = True,
        timeout: float = 300.0,
    ):
        """
        Yield text chunks from the server's streaming response.

        vllm-mlx is launched with the Qwen3 reasoning parser, so the server
        splits thinking from the answer into delta.reasoning_content vs
        delta.content. We re-wrap reasoning in <think>...</think> on the way
        out so the output stream looks identical to the local model
        (which emits raw <think> tags) and downstream _strip_think() works
        unchanged on the joined return value.
        """
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": True,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        in_think = False
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                rc = delta.get("reasoning_content")
                if rc:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield rc
                content = delta.get("content")
                if content:
                    if in_think:
                        yield "</think>\n\n"
                        in_think = False
                    yield content
        if in_think:
            yield "</think>\n\n"


def _check_vllm_server(base_url: str, model_id: str, timeout: float = 2.0) -> None:
    """
    Confirm the vllm-mlx server is reachable and serving model_id.
    Raises RuntimeError with an actionable message otherwise.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        raise RuntimeError(
            f"Cannot reach vllm-mlx server at {base_url}: {e}\n"
            f"Either start it (e.g. ../serving-local-models/serve-qwen3.sh) "
            f"or drop --remote to load the orchestrator locally via load_model()."
        ) from e

    served = {m.get("id") for m in json.loads(body).get("data", [])}
    if model_id not in served:
        raise RuntimeError(
            f"vllm-mlx server is up at {base_url} but '{model_id}' is not served. "
            f"Available: {sorted(served)}. Restart the server with that model, "
            f"or drop --remote to load locally."
        )


def _load_remote_orchestrator() -> tuple:
    """
    Verify the vllm-mlx server is up and return (RemoteOrchestrator, tokenizer).
    Tokenizer is loaded locally (small, ~10 MB) so the agent can still do
    truncation and prefill-token counts. Model weights are NOT downloaded.
    """
    _check_vllm_server(VLLM_BASE_URL, VLLM_ORCHESTRATOR_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(ORCHESTRATOR_ID)
    client = RemoteOrchestrator(VLLM_BASE_URL, VLLM_ORCHESTRATOR_MODEL_ID)
    return client, tokenizer


def load_nli_model(device: str = None) -> tuple:
    """
    Load fine-tuned Qwen3-1.7B + LoRA adapter (NLI Core).
    Adapter ON by default. Use model.disable_adapter() for base behaviour.
    enable_thinking=False at generate time (model was trained without thinking).
    """
    return load_peft_model(NLI_BASE_ID, NLI_ADAPTER_ID, device)
