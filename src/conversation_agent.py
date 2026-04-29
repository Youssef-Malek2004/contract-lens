"""
src/conversation_agent.py

Conversation Agent for ContractLens.
Drives Qwen3-4B (orchestrator model) with RAG context injection and
persistent multi-turn conversation history.

Architecture spec: architecture/architecture.yaml — agents.conversation_agent
Critical constraint: retrieved spans are reasoning aids only; all evidence
must cite spans from the analyzed contract, never from the training corpus.
"""

import importlib
import itertools
import json
import os
import re
import threading
import time
import warnings
from datetime import datetime, timezone
from typing import Callable, List

from src.loaders import ModelHandle, get_device, get_loader
from src.types import RetrievedSpan

# ── Constants ─────────────────────────────────────────────────────────────────

HISTORY_FILE = "conversation_history.json"
CONTEXT_WINDOW       = 20_000   # total budget (input + output) we let the orchestrator use
MAX_CONTRACT_TOKENS  = 14_000   # ceiling on the contract block alone — system + RAG + history + question + 1k output still fit in CONTEXT_WINDOW
MIN_NEW_TOKENS       = 256      # floor: refuse to generate if the prompt leaves less than this
SAFETY_MARGIN        = 256      # slack for chat-template tokens & client/server tokenizer drift
MAX_HISTORY_TURNS    = 5        # trim oldest pairs if session grows long


def _compute_max_new_tokens(input_len: int) -> int:
    """Output is whatever's left after the prompt — fills CONTEXT_WINDOW."""
    budget = CONTEXT_WINDOW - input_len - SAFETY_MARGIN
    if budget < MIN_NEW_TOKENS:
        raise ValueError(
            f"Prompt is too large: {input_len} input tokens leaves only {budget} "
            f"for output (need ≥ {MIN_NEW_TOKENS}). Lower MAX_CONTRACT_TOKENS or "
            f"raise CONTEXT_WINDOW."
        )
    return budget

CONVERSATION_SYSTEM_PROMPT = (
    "You are a contract analysis assistant for NDA review.\n\n"
    "You will receive:\n"
    "1. A [RETRIEVAL CONTEXT] block with example spans from a training corpus.\n"
    "   These are reasoning aids ONLY. Do NOT cite them as evidence, quote them,\n"
    "   or attribute any claim to them. They come from other contracts entirely.\n\n"
    "2. A CONTRACT block with the full text of the NDA being analyzed.\n"
    "   All evidence, quotes, and citations MUST come exclusively from this contract.\n"
    "   If you quote text, it must be a verbatim substring of this contract.\n\n"
    "3. A QUESTION from the user.\n\n"
    "Answer clearly and concisely. Cite specific language from the analyzed contract "
    "to support your answer. If the contract does not address the question, say so explicitly."
)

# ── Spinner (mirrors quick_infer.py) ─────────────────────────────────────────

class _Spinner:
    """Animates a braille spinner in a background thread during model prefill."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for ch in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            print(f"\r{ch} {self.message}", end="", flush=True)
            time.sleep(0.08)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        print(f"\r{' ' * (len(self.message) + 4)}\r", end="", flush=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks before displaying or storing a response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _stub_retrieve(
    query: str,
    top_k: int = 5,
    hypothesis_id: str = None,
    label_filter: str = None,
) -> List[RetrievedSpan]:
    """Fallback when the RAG module is not yet built. Returns empty list."""
    return []


def _load_rag_fn(mode: str) -> Callable:
    """
    Dynamically import the retrieve() function for the requested mode.
    Falls back to _stub_retrieve with a warning if the module doesn't exist yet
    (Members 2 & 3 may still be working on it).
    """
    module_name = "src.rag_vector" if mode == "vector" else "src.rag_graph"
    try:
        mod = importlib.import_module(module_name)
        return mod.retrieve
    except ImportError:
        warnings.warn(
            f"{module_name} not found — RAG context will be empty. "
            "Build the index first: python 03_build_index.py --mode " + mode,
            stacklevel=2,
        )
        return _stub_retrieve


def _load_contract(path: str, idx: int):
    """Load contract text and id from a ContractNLI JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    docs = data["documents"]
    if idx >= len(docs):
        raise IndexError(
            f"--idx {idx} is out of range — file has {len(docs)} documents (0-{len(docs)-1})"
        )
    doc = docs[idx]
    return doc["text"], str(doc["id"])


def _truncate_contract(text: str, max_tokens: int, tokenizer) -> str:
    """
    Truncate contract text to max_tokens to stay inside the model's context window.
    Largest test.json contract is ~10,500 tokens; p90 is ~5,500.
    """
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    truncated = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
    return truncated + "\n...[CONTRACT TRUNCATED]"


def _format_rag_context(spans: List[RetrievedSpan]) -> str:
    """
    Format retrieved spans as the [RETRIEVAL CONTEXT] block defined in
    architecture.yaml retrieval.agent_consumption.prompt_injection_format.
    Returns "" when spans is empty so the block is omitted entirely.
    """
    if not spans:
        return ""
    lines = [
        "[RETRIEVAL CONTEXT — external memory from training corpus]",
        "[For reasoning reference only. Never cite as evidence.]",
        "[All evidence must come from the analyzed contract below.]",
        "",
    ]
    for i, span in enumerate(spans, 1):
        ann = span.get("hypothesis_annotations", {})
        ann_str = ", ".join(f"{h}={lbl}" for h, lbl in ann.items()) if ann else "no annotations"
        lines.append(
            f"Example {i} | Source: {span['doc_id']}, span_{span['span_idx']} | {ann_str}"
        )
        lines.append(f'"{span["text"]}"')
        lines.append("")
    lines.append("[END RETRIEVAL CONTEXT]")
    return "\n".join(lines)


def _load_history(path: str) -> list:
    """Load conversation history from JSON file. Returns [] if absent or corrupted."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        warnings.warn(f"Could not read {path} — starting with fresh history.")
        return []


def _save_history(path: str, history: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _rotate_if_needed(history: list, contract_path: str, idx: int) -> list:
    """
    Clear history when the user switches to a different contract.
    History entries carry contract_idx and contract_path for this check.
    """
    if not history:
        return []
    last = history[-1]
    if last.get("contract_idx") != idx or last.get("contract_path") != contract_path:
        print(f"[ConversationAgent] New contract session (idx={idx}). Clearing history.")
        return []
    return history


def _build_messages(
    history: list,
    rag_block: str,
    contract_text: str,
    question: str,
) -> list:
    """
    Assemble the messages list passed to tokenizer.apply_chat_template.

    Turn 1 (no history):
        [system] [user: RAG_BLOCK + CONTRACT + QUESTION]

    Turn N (history present):
        [system] [prior user/assistant pairs...] [user: RAG_BLOCK + CONTRACT + QUESTION]

    The full contract and RAG block are included in every new user turn so the
    model has grounding regardless of how much of the earlier context fits in
    attention. Prior turns are trimmed to MAX_HISTORY_TURNS pairs if the
    session grows long.
    """
    messages = [{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}]

    # Replay prior turns — strip tracking fields, keep only role + content
    prior = [{"role": e["role"], "content": e["content"]} for e in history]
    # Trim to MAX_HISTORY_TURNS pairs (each pair = 1 user + 1 assistant message)
    max_prior_messages = MAX_HISTORY_TURNS * 2
    if len(prior) > max_prior_messages:
        prior = prior[-max_prior_messages:]
    messages.extend(prior)

    # Build the new user turn
    parts = []
    if rag_block:
        parts.append(rag_block)
    parts.append(f"CONTRACT:\n{contract_text}")
    parts.append(f"QUESTION: {question}")
    messages.append({"role": "user", "content": "\n\n".join(parts)})

    return messages


# ── ConversationAgent ─────────────────────────────────────────────────────────

class ConversationAgent:
    """
    Conversational response layer driven by Qwen3-4B (orchestrator model).

    Loads the model once on __init__. Each call to run_turn():
      1. Loads the requested contract
      2. Loads/rotates conversation history
      3. Retrieves RAG context via the active branch (vector or graph)
      4. Builds the prompt and generates a response with streaming
      5. Persists the exchange to conversation_history.json
    """

    def __init__(self, retrieval_mode: str = "vector", remote: bool = False) -> None:
        self.device = get_device()
        mode = "vllm" if remote else "local"
        if remote:
            print("[ConversationAgent] Mode: remote (vllm-mlx @ localhost:8001)")
        else:
            print(f"[ConversationAgent] Device: {self.device}")
        print("[ConversationAgent] Loading Qwen3-4B (orchestrator)…")
        self._model: ModelHandle = get_loader(mode, device=self.device).load_orchestrator()
        self.retrieve_fn = _load_rag_fn(retrieval_mode)
        self.retrieval_mode = retrieval_mode
        self.history_path = HISTORY_FILE
        print("[ConversationAgent] Ready.")

    def run_turn(self, contract_path: str, idx: int, user_prompt: str) -> str:
        """
        Execute one conversational turn.

        Args:
            contract_path: Path to ContractNLI JSON file (e.g. data/test.json)
            idx:           Zero-based document index within that file
            user_prompt:   User's natural language question

        Returns:
            Cleaned response string (think blocks stripped).
        """
        # 1. Load contract
        contract_text, doc_id = _load_contract(contract_path, idx)
        contract_text = _truncate_contract(contract_text, MAX_CONTRACT_TOKENS, self._model.tokenizer)
        print(f"[ConversationAgent] Contract: {doc_id}  ({len(contract_text)} chars)")

        # 2. Load and rotate history
        history = _rotate_if_needed(
            _load_history(self.history_path), contract_path, idx
        )

        # 3. Retrieve RAG context
        rag_spans = self.retrieve_fn(query=user_prompt, top_k=5)
        rag_block = _format_rag_context(rag_spans)
        if rag_spans:
            print(f"[ConversationAgent] RAG: {len(rag_spans)} spans retrieved ({self.retrieval_mode})")
            for i, span in enumerate(rag_spans, 1):
                ann = span.get("hypothesis_annotations", {}) or {}
                ann_str = ", ".join(f"{h}={lbl}" for h, lbl in ann.items()) or "no-annotations"
                preview = span["text"].replace("\n", " ")
                if len(preview) > 140:
                    preview = preview[:140] + "…"
                print(f"  [{i}] doc={span['doc_id']} span={span['span_idx']} score={span.get('score')} | {ann_str}")
                print(f"      \"{preview}\"")
        else:
            print(f"[ConversationAgent] RAG: no context (index not built or stub active)")

        # 4. Build messages
        messages = _build_messages(history, rag_block, contract_text, user_prompt)

        # 5. Generate response
        response = self._generate(messages)

        # 6. Persist exchange
        now = datetime.now(timezone.utc).isoformat()
        history.append({
            "role": "user",
            "content": messages[-1]["content"],
            "timestamp": now,
            "contract_idx": idx,
            "contract_path": contract_path,
        })
        history.append({
            "role": "assistant",
            "content": response,
            "timestamp": now,
            "contract_idx": idx,
            "contract_path": contract_path,
        })
        _save_history(self.history_path, history)
        print(f"[ConversationAgent] History saved ({len(history)} entries) → {self.history_path}")

        return response

    def _generate(self, messages: list) -> str:
        """
        Stream a response via the model handle (local or vllm — same code path).
        Computes token budget locally, drives the spinner until the first chunk
        arrives, then streams the rest to stdout. Returns think-stripped text.
        """
        tokenizer = self._model.tokenizer
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_len = len(tokenizer.encode(text))
        max_new = _compute_max_new_tokens(input_len)

        stream = self._model.stream(messages, max_new, enable_thinking=True)

        with _Spinner(f"Reading {input_len} input tokens (budget {max_new} out)…"):
            try:
                first_chunk = next(stream)
            except StopIteration:
                first_chunk = ""

        print("Answer ▶ ", end="", flush=True)
        output = first_chunk
        if first_chunk:
            print(first_chunk, end="", flush=True)
        for chunk in stream:
            print(chunk, end="", flush=True)
            output += chunk
        print()

        return _strip_think(output.strip())
