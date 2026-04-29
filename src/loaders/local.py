"""
Local model loader — loads weights directly into process memory.
Works on Windows, Linux, Mac (MPS/CUDA/CPU). No vllm dependency.

Quantization is automatic:
  CUDA → BitsAndBytesConfig NF4 4-bit   (native int4 GEMM kernels)
  MPS  → dtype=torch.float16            (MPS has no native int4 GEMM)
  CPU  → dtype=torch.float16            (fallback only)
"""
from __future__ import annotations

import threading
from typing import Iterator

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from .interface import ModelHandle, ModelLoader
from ._constants import ORCHESTRATOR_ID, NLI_BASE_ID, NLI_ADAPTER_ID


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _quant_config(device: str) -> BitsAndBytesConfig | None:
    if device == "cuda":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None  # MPS + CPU: float16 via dtype kwarg


class LocalModelHandle(ModelHandle):
    def __init__(self, model_id: str, model, tokenizer, device: str) -> None:
        self._model_id = model_id
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tokenizer(self):
        return self._tokenizer

    def stream(
        self,
        messages: list[dict],
        max_new_tokens: int,
        enable_thinking: bool = True,
    ) -> Iterator[str]:
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300.0,
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=lambda: self._model.generate(**gen_kwargs))
        thread.start()
        yield from streamer
        thread.join()


class LocalLoader(ModelLoader):
    """
    Loads all three models locally using transformers + peft.
    Pass device=None for automatic detection (MPS → CUDA → CPU).
    """

    def __init__(self, device: str = None) -> None:
        self.device = device or get_device()

    def _load_causal_lm(self, model_id: str, tokenizer_id: str = None) -> LocalModelHandle:
        quant = _quant_config(self.device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
        if isinstance(quant, BitsAndBytesConfig):
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
        model.eval()
        return LocalModelHandle(model_id, model, tokenizer, self.device)

    def load_orchestrator(self) -> LocalModelHandle:
        return self._load_causal_lm(ORCHESTRATOR_ID)

    def load_nli_model(self) -> LocalModelHandle:
        """
        Loads fine-tuned Qwen3-1.7B with LoRA adapter ON.
        Tokenizer is loaded from the adapter repo — it has the patched chat template.
        Use enable_thinking=False at generate time (model was SFT-trained without thinking).
        """
        quant = _quant_config(self.device)
        if isinstance(quant, BitsAndBytesConfig):
            base = AutoModelForCausalLM.from_pretrained(
                NLI_BASE_ID,
                quantization_config=quant,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                NLI_BASE_ID,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(NLI_ADAPTER_ID)
        model = PeftModel.from_pretrained(
            base, NLI_ADAPTER_ID, is_quantized=(self.device == "cuda")
        )
        model.eval()
        return LocalModelHandle(NLI_ADAPTER_ID, model, tokenizer, self.device)

    def load_base_model(self) -> LocalModelHandle:
        """Loads base Qwen3-1.7B without any adapter. Use enable_thinking=True."""
        return self._load_causal_lm(NLI_BASE_ID)
