"""
vllm loader — all three models are served by separate vllm-mlx processes.
Only tokenizers are downloaded locally (~10 MB each); weights stay on the server.

Default ports:
  8001 — Orchestrator  (mlx-community/Qwen3-4b-4bit)
  8002 — NLI Core      (your merged + MLX-converted fine-tuned 1.7B)
  8003 — Base 1.7B     (mlx-community/Qwen3-1.7b-4bit)

Each load_*() call verifies the server is reachable and serving the expected
model ID before returning. If a server is down or the model ID doesn't match,
a RuntimeError is raised with a human-readable message.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from transformers import AutoTokenizer

from ._constants import (
    NLI_ADAPTER_ID,
    BASE_VLLM_ID,
    BASE_VLLM_URL,
    NLI_VLLM_ID,
    NLI_VLLM_URL,
    ORCHESTRATOR_VLLM_ID,
    ORCHESTRATOR_VLLM_URL,
)
from .interface import ModelHandle, ModelLoader


@dataclass
class VllmConfig:
    """Server URLs and model IDs for all three endpoints. Override any field as needed."""
    orchestrator_url:      str = ORCHESTRATOR_VLLM_URL
    orchestrator_model_id: str = ORCHESTRATOR_VLLM_ID
    nli_url:               str = NLI_VLLM_URL
    nli_model_id:          str = NLI_VLLM_ID
    base_url:              str = BASE_VLLM_URL
    base_model_id:         str = BASE_VLLM_ID


class VllmModelHandle(ModelHandle):
    """
    HTTP client over an OpenAI-compatible vllm-mlx endpoint.

    The vllm-mlx Qwen3 reasoning parser splits thinking into
    delta.reasoning_content and the answer into delta.content.
    stream() re-wraps reasoning in <think> tags so the output
    is identical to local inference and _strip_think() works unchanged.
    """

    def __init__(self, model_id: str, base_url: str, tokenizer) -> None:
        self._model_id = model_id
        self._base_url = base_url
        self._tokenizer = tokenizer

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
        timeout: float = 300.0,
    ) -> Iterator[str]:
        payload = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": True,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
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
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
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


class VllmLoader(ModelLoader):
    """
    Produces VllmModelHandle instances by connecting to running vllm-mlx servers.
    Each load_*() verifies the target server is up and serving the correct model ID.
    """

    def __init__(self, config: VllmConfig = None) -> None:
        self.config = config or VllmConfig()

    def _verify(self, base_url: str, model_id: str, timeout: float = 2.0) -> None:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
                body = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Cannot reach vllm server at {base_url}: {exc}\n"
                f"Start the server serving '{model_id}' before loading."
            ) from exc
        served = {m.get("id") for m in json.loads(body).get("data", [])}
        if model_id not in served:
            raise RuntimeError(
                f"Server at {base_url} is up but '{model_id}' is not being served.\n"
                f"Available models: {sorted(served)}"
            )

    def _build(
        self,
        base_url: str,
        model_id: str,
        tokenizer_id: str = None,
    ) -> VllmModelHandle:
        self._verify(base_url, model_id)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
        return VllmModelHandle(model_id, base_url, tokenizer)

    def load_orchestrator(self) -> VllmModelHandle:
        c = self.config
        return self._build(c.orchestrator_url, c.orchestrator_model_id)

    def load_nli_model(self) -> VllmModelHandle:
        c = self.config
        # Tokenizer from the adapter repo — has the patched Qwen3 chat template.
        return self._build(c.nli_url, c.nli_model_id, tokenizer_id=NLI_ADAPTER_ID)

    def load_base_model(self) -> VllmModelHandle:
        c = self.config
        return self._build(c.base_url, c.base_model_id)
