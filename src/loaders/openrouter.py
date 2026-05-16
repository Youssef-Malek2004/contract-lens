"""
OpenRouter loader — connects to OpenRouter's OpenAI-compatible API.

Reads OPENROUTER_API_KEY from the environment (or .env if python-dotenv is
installed). Only tokenizers are downloaded locally so prompt building and
token counting still work.

NLI is intentionally unsupported here — the fine-tuned contractnli adapter is
local-only, so load_nli_model() raises. Use LocalLoader for that role.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from transformers import AutoTokenizer

from ._constants import (
    NLI_BASE_ID,
    OPENROUTER_BASE_MODEL_ID,
    OPENROUTER_BASE_URL,
    OPENROUTER_ORCHESTRATOR_ID,
    ORCHESTRATOR_ID,
)
from .interface import ModelHandle, ModelLoader


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class OpenRouterConfig:
    """OpenRouter endpoint + model IDs. api_key falls back to OPENROUTER_API_KEY."""
    api_key:               str = None
    base_url:              str = OPENROUTER_BASE_URL
    orchestrator_model_id: str = OPENROUTER_ORCHESTRATOR_ID
    base_model_id:         str = OPENROUTER_BASE_MODEL_ID
    http_referer:          str = None  # optional ranking header
    x_title:               str = None  # optional ranking header


class OpenRouterModelHandle(ModelHandle):
    """
    HTTP client over OpenRouter's OpenAI-compatible chat/completions endpoint.

    Reasoning content arrives in delta.reasoning (some providers use
    delta.reasoning_content); both are re-wrapped in <think> tags so the
    output is identical to the local and vllm backends.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        tokenizer,
        http_referer: str = None,
        x_title: str = None,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url
        self._api_key = api_key
        self._tokenizer = tokenizer
        self._http_referer = http_referer
        self._x_title = x_title

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
        }
        if not enable_thinking:
            payload["reasoning"] = {"exclude": True}

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._x_title:
            headers["X-Title"] = self._x_title

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
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
                rc = delta.get("reasoning") or delta.get("reasoning_content")
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


class OpenRouterLoader(ModelLoader):
    """
    Produces OpenRouterModelHandle instances against OpenRouter's API.

    Tokenizers come from the matching HF repos (ORCHESTRATOR_ID, NLI_BASE_ID)
    so chat-template formatting and token counting work the same as the other
    backends. NLI is not supported — the fine-tuned adapter is local-only.
    """

    def __init__(self, config: OpenRouterConfig = None) -> None:
        self.config = config or OpenRouterConfig()
        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to .env (see .env.example) or export it before loading the OpenRouter backend."
            )
        self._api_key = api_key

    def _build(self, model_id: str, tokenizer_id: str) -> OpenRouterModelHandle:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        return OpenRouterModelHandle(
            model_id=model_id,
            base_url=self.config.base_url,
            api_key=self._api_key,
            tokenizer=tokenizer,
            http_referer=self.config.http_referer,
            x_title=self.config.x_title,
        )

    def load_orchestrator(self) -> OpenRouterModelHandle:
        return self._build(self.config.orchestrator_model_id, ORCHESTRATOR_ID)

    def load_nli_model(self) -> OpenRouterModelHandle:
        raise NotImplementedError(
            "OpenRouterLoader does not serve the NLI model — the fine-tuned "
            "contractnli adapter is local-only. Use LocalLoader for NLI."
        )

    def load_base_model(self) -> OpenRouterModelHandle:
        return self._build(self.config.base_model_id, NLI_BASE_ID)
