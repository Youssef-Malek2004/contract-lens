#!/usr/bin/env python3
"""
test_vllm_endpoints.py
Unit tests for all three vllm-mlx endpoints.

Each test is independent — a server being down skips only that test.
Run with all three servers active for a full green suite.

Start servers (from ServeLM/):
    ./serve-qwen3.sh            # port 8001 — Orchestrator
    ./serve-nli.sh              # port 8002 — NLI Core (needs merged model)
    ./serve-base.sh             # port 8003 — Base 1.7B

Usage:
    conda activate genai-ms2
    python tests/test_vllm_endpoints.py
    python -m unittest tests.test_vllm_endpoints -v
"""
import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import VllmConfig, VllmLoader
from src.loaders._constants import (
    BASE_VLLM_ID,
    BASE_VLLM_URL,
    NLI_VLLM_ID,
    NLI_VLLM_URL,
    ORCHESTRATOR_VLLM_ID,
    ORCHESTRATOR_VLLM_URL,
)

NLI_SYSTEM_PROMPT = (
    "You are a contract NLI system. Given contract spans and hypotheses, "
    "classify each hypothesis as ENTAILED, CONTRADICTED, or NOT_MENTIONED. "
    "Return ONLY a JSON array."
)
NLI_USER_PROMPT = (
    'Contract Spans:\n[0] "The Receiving Party shall not use Confidential Information '
    'for any purpose other than evaluation."\n\n'
    "Hypotheses:\n"
    "H04: Receiving Party shall not use any Confidential Information for any "
    "purpose other than the purposes stated in Agreement.\n\n"
    "Return ONLY a JSON array. No other text."
)


def _server_live(base_url: str, model_id: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            data = json.loads(resp.read())
        return any(m.get("id") == model_id for m in data.get("data", []))
    except Exception:
        return False


class TestOrchestratorEndpoint(unittest.TestCase):
    """Qwen3-4B at localhost:8001"""

    def setUp(self):
        if not _server_live(ORCHESTRATOR_VLLM_URL, ORCHESTRATOR_VLLM_ID):
            self.skipTest(
                f"Orchestrator server not running at {ORCHESTRATOR_VLLM_URL} "
                f"serving '{ORCHESTRATOR_VLLM_ID}'"
            )
        config = VllmConfig()
        self.handle = VllmLoader(config).load_orchestrator()

    def test_returns_non_empty_response(self):
        response = self.handle.generate(
            [{"role": "user", "content": "What is a Non-Disclosure Agreement? One sentence."}],
            max_new_tokens=2000,
            enable_thinking=True,
        )
        self.assertGreater(len(response), 0)

    def test_think_blocks_stripped_by_generate(self):
        response = self.handle.generate(
            [{"role": "user", "content": "What does NDA stand for?"}],
            max_new_tokens=2000,
            enable_thinking=True,
        )
        self.assertNotIn("<think>", response)
        self.assertNotIn("</think>", response)

    def test_stream_yields_chunks(self):
        chunks = list(self.handle.stream(
            [{"role": "user", "content": "Define confidentiality in one word."}],
            max_new_tokens=2000,
            enable_thinking=False,
        ))
        self.assertGreater(len(chunks), 0)


class TestNliModelEndpoint(unittest.TestCase):
    """Fine-tuned Qwen3-1.7B at localhost:8002"""

    def setUp(self):
        if not _server_live(NLI_VLLM_URL, NLI_VLLM_ID):
            self.skipTest(
                f"NLI server not running at {NLI_VLLM_URL} serving '{NLI_VLLM_ID}'.\n"
                f"Run merge_adapter.py then start serve-nli.sh first."
            )
        config = VllmConfig()
        self.handle = VllmLoader(config).load_nli_model()

    def test_returns_json_array(self):
        response = self.handle.generate(
            [
                {"role": "system", "content": NLI_SYSTEM_PROMPT},
                {"role": "user",   "content": NLI_USER_PROMPT},
            ],
            max_new_tokens=1500,
            enable_thinking=False,
        )
        self.assertIn("[", response, "NLI model did not return a JSON array")
        self.assertIn("]", response)

    def test_contains_hypothesis_id(self):
        response = self.handle.generate(
            [
                {"role": "system", "content": NLI_SYSTEM_PROMPT},
                {"role": "user",   "content": NLI_USER_PROMPT},
            ],
            max_new_tokens=1500,
            enable_thinking=False,
        )
        self.assertIn("H04", response)

    def test_contains_valid_label(self):
        response = self.handle.generate(
            [
                {"role": "system", "content": NLI_SYSTEM_PROMPT},
                {"role": "user",   "content": NLI_USER_PROMPT},
            ],
            max_new_tokens=1500,
            enable_thinking=False,
        )
        valid_labels = ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED")
        self.assertTrue(
            any(lbl in response for lbl in valid_labels),
            f"Response contains none of {valid_labels}:\n{response}",
        )

    def test_no_think_blocks_when_thinking_disabled(self):
        response = self.handle.generate(
            [
                {"role": "system", "content": NLI_SYSTEM_PROMPT},
                {"role": "user",   "content": NLI_USER_PROMPT},
            ],
            max_new_tokens=1500,
            enable_thinking=False,
        )
        self.assertNotIn("<think>", response)


class TestBaseModelEndpoint(unittest.TestCase):
    """Base Qwen3-1.7B at localhost:8003"""

    def setUp(self):
        if not _server_live(BASE_VLLM_URL, BASE_VLLM_ID):
            self.skipTest(
                f"Base model server not running at {BASE_VLLM_URL} "
                f"serving '{BASE_VLLM_ID}'"
            )
        config = VllmConfig()
        self.handle = VllmLoader(config).load_base_model()

    def test_returns_non_empty_response(self):
        # enable_thinking=False: Qwen3-1.7B is verbose in reasoning and exhausts
        # 2000+ tokens on thinking alone; for endpoint verification we just need
        # to confirm the server returns a non-empty response.
        response = self.handle.generate(
            [{"role": "user", "content": "Summarise in one sentence: keep secrets secret."}],
            max_new_tokens=500,
            enable_thinking=False,
        )
        self.assertGreater(len(response), 0)

    def test_think_blocks_stripped_by_generate(self):
        response = self.handle.generate(
            [{"role": "user", "content": "What does confidential mean?"}],
            max_new_tokens=2000,
            enable_thinking=True,
        )
        self.assertNotIn("<think>", response)
        self.assertNotIn("</think>", response)

    def test_stream_yields_chunks(self):
        chunks = list(self.handle.stream(
            [{"role": "user", "content": "Say the word: confidential."}],
            max_new_tokens=2000,
            enable_thinking=False,
        ))
        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
