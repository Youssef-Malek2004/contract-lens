"""
Central model loading for all agents.
Handles device detection and quantization automatically.
All agents call load_model() or load_peft_model() — never handle device/quant directly.

Device priority:  MPS (Apple Silicon) → CUDA (Nvidia) → CPU
Quantization:     CUDA → BitsAndBytesConfig NF4 4-bit
                  MPS  → QuantoConfig int4  (optimum-quanto, device-agnostic)
                  CPU  → QuantoConfig int4  (slow, fallback only)
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Model IDs ────────────────────────────────────────────────────────────────
ORCHESTRATOR_ID = "Qwen/Qwen3-4B"    # pure transformer, full MPS speed, thinking + tool-calling
NLI_BASE_ID     = "Qwen/Qwen3-1.7B"
NLI_ADAPTER_ID  = "Youssef-Malek/contractnli-vast-ai-qwen3-1.7b"


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
           MPS has no native int4 kernels. QuantoConfig dequantizes every
           forward pass → slower inference + 2-5 min startup cost. Not worth it.

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

def load_orchestrator(device: str = None) -> tuple:
    """
    Load Qwen3-4B (orchestrator + conversation agent model).
    Use enable_thinking=True in apply_chat_template at generate time.
    """
    return load_model(ORCHESTRATOR_ID, device)


def load_nli_model(device: str = None) -> tuple:
    """
    Load fine-tuned Qwen3-1.7B + LoRA adapter (NLI Core).
    Adapter ON by default. Use model.disable_adapter() for base behaviour.
    enable_thinking=False at generate time (model was trained without thinking).
    """
    return load_peft_model(NLI_BASE_ID, NLI_ADAPTER_ID, device)
