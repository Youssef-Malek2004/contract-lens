from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterator


class ModelHandle(ABC):
    """
    Unified interface for a loaded model regardless of backend (local or vllm).

    Subclasses implement stream(). generate() is provided here as a template
    method: it collects the full stream and strips <think> blocks.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """HuggingFace model ID or local path — for logging."""
        ...

    @property
    @abstractmethod
    def tokenizer(self):
        """Tokenizer — for prompt building and token counting."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        max_new_tokens: int,
        enable_thinking: bool = True,
    ) -> Iterator[str]:
        """
        Yield raw text chunks. <think>…</think> blocks are included when
        enable_thinking=True so callers can display or strip them.
        """
        ...

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
        enable_thinking: bool = True,
    ) -> str:
        """Collect the full stream and return it with <think> blocks stripped."""
        raw = "".join(self.stream(messages, max_new_tokens, enable_thinking))
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


class ModelLoader(ABC):
    """
    Factory that produces a ModelHandle for each of the three model roles.

    Implementations: LocalLoader (src/loaders/local.py)
                     VllmLoader  (src/loaders/vllm.py)
    """

    @abstractmethod
    def load_orchestrator(self) -> ModelHandle:
        """Qwen3-4B — conversation agent. Use enable_thinking=True."""
        ...

    @abstractmethod
    def load_nli_model(self) -> ModelHandle:
        """Fine-tuned Qwen3-1.7B — NLI classification. Use enable_thinking=False."""
        ...

    @abstractmethod
    def load_base_model(self) -> ModelHandle:
        """Base Qwen3-1.7B — hypothesis agents, confidence + quotes. Use enable_thinking=True."""
        ...
