"""
src/dispatcher.py

Full-analysis dispatcher for ContractLens.

Flow implemented here:
  orchestrator -> run_full_analysis tool -> Dispatcher
  Dispatcher fans out 17 hypothesis jobs.
  For each hypothesis:
    1. retrieve external examples from the active RAG backend
    2. send analyzed contract + hypothesis + external examples to one
       OpenRouter Qwen 3.5 9B hypothesis agent
    3. normalize the single-hypothesis JSON into a HypothesisTrace

Important boundary:
  The hypothesis agent is not told that a RAG system exists. Retrieved items are
  described only as optional "background examples". Final evidence is forced to
  come only from the analyzed contract.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from src.constants import HYPOTHESES, LABEL_TO_STATUS
# Keep these local to avoid importing the full model-loader package when the
# dispatcher is used in lightweight/test environments.
N_PARALLEL_AGENTS = 5
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_BASE_MODEL_ID = "qwen/qwen3.5-9b"
from src.runtrace import (
    build_contract_runtrace,
    utc_now_iso,
    write_contract_runtrace,
)
from src.runtrace_recorder import get_active_recorder
from src.tools import retrieve
from src.types import HypothesisTrace, RetrievedSpan

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


LABELS = {"ENTAILED", "CONTRADICTED", "NOT_MENTIONED"}
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 180.0


@dataclass
class DispatcherConfig:
    model_id: str = OPENROUTER_BASE_MODEL_ID   # qwen/qwen3.5-9b
    base_url: str = OPENROUTER_BASE_URL
    api_key: Optional[str] = None
    top_k: int = DEFAULT_TOP_K
    max_tokens: int = 1500
    temperature: float = 0.0
    concurrency: int = N_PARALLEL_AGENTS
    timeout: float = DEFAULT_TIMEOUT

    def resolved_api_key(self) -> str:
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env or export it before running /analyze."
            )
        return key


class FullAnalysisDispatcher:
    """Fan-out dispatcher that runs exactly one agent per hypothesis."""

    def __init__(self, contract: dict, session_id: str | None = None, config: DispatcherConfig | None = None) -> None:
        self.contract = contract
        self.session_id = session_id
        self.config = config or DispatcherConfig()
        self.contract_text = str(contract.get("text") or "")
        self.contract_id = f"doc_{contract.get('id', 'unknown')}"
        self.numbered_spans = _numbered_contract_spans(contract)

    async def run_full_analysis(self, contract_id: str | None = None, retrieval_mode: str = "vector") -> dict:
        """
        Entry point used by src.tools.run_full_analysis.

        `contract_id` is accepted for the tool schema, but the dispatcher uses
        the contract object captured from the active SessionState.
        """
        started_at = utc_now_iso()
        wall_start = time.perf_counter()
        sem = asyncio.Semaphore(max(1, int(self.config.concurrency)))

        async def _bounded(h_id: str, h_text: str) -> HypothesisTrace:
            async with sem:
                return await self._run_one_hypothesis(h_id, h_text, retrieval_mode)

        tasks = [_bounded(h_id, h_text) for h_id, h_text in HYPOTHESES.items()]
        traces = list(await asyncio.gather(*tasks))
        traces.sort(key=lambda t: t.get("hypothesis_id", ""))

        elapsed_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        metrics = _compute_metrics(traces, elapsed_ms)
        recorder = get_active_recorder()
        tool_calls = recorder.snapshot() if recorder else []

        runtrace = build_contract_runtrace(
            run_id=f"full_analysis_{self.contract_id}_{int(time.time())}",
            contract={
                "contract_id": self.contract_id,
                "source_id": self.contract.get("id"),
                "file_name": self.contract.get("file_name"),
                "span_count": len(self.contract.get("spans") or []),
                "text_chars": len(self.contract_text),
            },
            retrieval_mode=retrieval_mode,
            playbook={"hypotheses": HYPOTHESES, "labels": sorted(LABELS)},
            hypothesis_traces=traces,
            metrics=metrics,
            started_at=started_at,
            ended_at=utc_now_iso(),
            tool_calls=tool_calls,
            parameters={
                "dispatcher": "17_parallel_hypothesis_agents",
                "model": self.config.model_id,
                "top_k_per_hypothesis": self.config.top_k,
                "concurrency": self.config.concurrency,
            },
            retrieval_context={
                "external_memory": "retrieved per hypothesis and hidden from final evidence",
                "contract_evidence": "numbered spans from analyzed contract only",
            },
            session_id=self.session_id,
        )
        path = write_contract_runtrace(runtrace)

        return {
            "status": "completed",
            "contract_id": self.contract_id,
            "requested_contract_id": contract_id,
            "retrieval_mode": retrieval_mode,
            "model": self.config.model_id,
            "hypothesis_traces": traces,
            "metrics": metrics,
            "runtrace_path": str(path),
        }

    async def _run_one_hypothesis(self, h_id: str, hypothesis: str, retrieval_mode: str) -> HypothesisTrace:
        started = utc_now_iso()
        t0 = time.perf_counter()
        retrieved: list[RetrievedSpan] = []
        try:
            query = _build_retrieval_query(h_id, hypothesis, self.contract_text)
            retrieved = retrieve(
                query=query,
                mode=retrieval_mode,
                top_k=self.config.top_k,
                hypothesis_id=h_id,
                agent_id=h_id,
            )
            messages = _build_agent_messages(
                contract_spans=self.numbered_spans,
                hypothesis_id=h_id,
                hypothesis=hypothesis,
                background_examples=retrieved,
            )
            raw = await asyncio.to_thread(self._call_openrouter, messages)
            parsed = _parse_agent_json(raw)
            trace = _trace_from_prediction(
                h_id=h_id,
                hypothesis=hypothesis,
                prediction=parsed,
                contract_text=self.contract_text,
                contract_spans=self.contract.get("spans") or [],
                retrieved=retrieved,
                started_at=started,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                model_id=self.config.model_id,
            )
            return trace
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return _error_trace(
                h_id=h_id,
                hypothesis=hypothesis,
                error=exc,
                retrieved=retrieved,
                started_at=started,
                latency_ms=latency_ms,
                model_id=self.config.model_id,
            )

    def _call_openrouter(self, messages: list[dict]) -> str:
        system_message = {
            "role": "system",
            "content": (
                "You are a strict JSON API. "
                "Return ONLY valid JSON. "
                "Do not use markdown. "
                "Do not use code fences. "
                "Do not explain. "
                "Do not include <think> tags."
            ),
        }

        payload = {
            "model": self.config.model_id,
            "messages": [system_message] + messages,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "reasoning": {
                "effort": "none",
                "exclude": True
            },
        }

        req = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.resolved_api_key()}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {err}") from exc

        return ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""

def make_pipeline(contract: dict, session_id: str | None = None, config: DispatcherConfig | None = None):
    """Return an async pipeline function compatible with ToolContext.pipeline."""

    default_dispatcher = FullAnalysisDispatcher(
        contract=contract,
        session_id=session_id,
        config=config,
    )

    async def _pipeline(
        contract_id: str,
        retrieval_mode: str = "vector",
        contract_override: dict | None = None,
    ) -> dict:
        if contract_override is not None:
            dispatcher = FullAnalysisDispatcher(
                contract=contract_override,
                session_id=session_id,
                config=config,
            )
            return await dispatcher.run_full_analysis(
                contract_id=str(contract_override.get("id", "dynamic_contract")),
                retrieval_mode=retrieval_mode,
            )

        return await default_dispatcher.run_full_analysis(
            contract_id=contract_id,
            retrieval_mode=retrieval_mode,
        )

    return _pipeline


def _contract_from_latest_user_prompt() -> dict | None:
    """
    If the latest user message contains a pasted contract, convert it into
    the same dict shape the dispatcher already expects.
    """
    try:
        from src.tools import get_tool_context
        ctx = get_tool_context()
    except Exception:
        return None

    history = getattr(ctx.session, "conversation_history", []) or []

    latest_user_message = None
    for turn in reversed(history):
        if turn.get("role") == "user":
            latest_user_message = turn.get("content") or ""
            break

    if not latest_user_message:
        return None

    contract_text = _extract_contract_text(latest_user_message)

    if not contract_text:
        return None

    return {
        "id": "prompt_contract",
        "file_name": "contract_from_prompt.txt",
        "text": contract_text,
        "spans": [[0, len(contract_text)]],
    }


def _extract_contract_text(message: str) -> str | None:
    """
    Supported prompt formats:

    CONTRACT:
    ...contract text...

    or:

    CONTRACT:
    ...contract text...
    END CONTRACT
    """
    msg = message.strip()

    match = re.search(
        r"CONTRACT\s*:\s*(.*?)(?:\n\s*END\s+CONTRACT\s*$|$)",
        msg,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None

    contract_text = match.group(1).strip()

    if len(contract_text) < 100:
        return None

    return contract_text

def _numbered_contract_spans(contract: dict) -> list[dict]:
    text = str(contract.get("text") or "")
    spans = contract.get("spans") or []
    numbered = []
    for i, span in enumerate(spans):
        try:
            start, end = int(span[0]), int(span[1])
            span_text = text[start:end]
        except Exception:
            start, end, span_text = None, None, str(span)
        numbered.append({"span_id": i, "start": start, "end": end, "text": span_text})
    if not numbered and text:
        # Fallback for non-ContractNLI inputs.
        numbered.append({"span_id": 0, "start": 0, "end": len(text), "text": text})
    return numbered


def _build_retrieval_query(h_id: str, hypothesis: str, contract_text: str) -> str:
    # Contract words help find similar clauses; hypothesis keeps retrieval focused.
    preview = re.sub(r"\s+", " ", contract_text[:1200]).strip()
    return f"{h_id}: {hypothesis}\nContract excerpt: {preview}"


def _build_agent_messages(
    *,
    contract_spans: list[dict],
    hypothesis_id: str,
    hypothesis: str,
    background_examples: Iterable[RetrievedSpan],
) -> list[dict]:
    span_lines = []
    for s in contract_spans:
        txt = re.sub(r"\s+", " ", str(s.get("text") or "")).strip()
        if txt:
            span_lines.append(f"[{s['span_id']}] {txt}")

    examples = []
    for i, span in enumerate(background_examples, 1):
        txt = re.sub(r"\s+", " ", str(span.get("text") or "")).strip()
        if txt:
            examples.append(f"Example {i}: {txt}")

    contract_spans_text = "\n".join(span_lines)
    external_context = "\n".join(examples[:5]) if examples else "No additional examples."

    prompt = f"""
    /no_think
Analyze exactly ONE hypothesis against the contract.

You must return ONLY this JSON object:

{{
  "hypothesis_id": "{hypothesis_id}",
  "label": "ENTAILED|CONTRADICTED|NOT_MENTIONED",
  "confidence": 0.0,
  "evidence_span_ids": [],
  "verbatim_quote": null,
  "groundedness_check": false,
  "quote_integrity_check": true,
  "reason": "one short sentence"
}}

Rules:
- Output valid JSON only.
- No markdown.
- No explanation.
- No text before or after JSON.
- The label must be exactly one of: ENTAILED, CONTRADICTED, NOT_MENTIONED.
- evidence_span_ids must be a list of span numbers from the contract only.
- If there is no direct contract evidence, use [] and verbatim_quote=null.
- Additional reference material is only background. Do not cite it as evidence.
- Evidence must come only from Contract spans.

Contract spans:
{contract_spans_text}

Hypothesis:
{hypothesis}

Additional reference material:
{external_context}
"""

    return [
        {
            "role": "user",
            "content": prompt,
        }
    ]
def _parse_agent_json(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Agent did not return JSON: {text[:300]}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Agent JSON must be an object")
    return data

def _repair_quote(
    quote: str | None,
    evidence_ids: list[int],
    contract_text: str,
    contract_spans: list,
) -> str | None:
    if not quote:
        return None

    try:
        # If the model already gave an exact quote, keep it.
        if quote in contract_text:
            return quote

        cleaned_quote = re.sub(r"\s+", " ", quote).strip()

        for evidence_id in evidence_ids:
            if not (0 <= evidence_id < len(contract_spans)):
                continue

            span = contract_spans[evidence_id]

            # Case 1: span is [start, end] or (start, end)
            if isinstance(span, (list, tuple)) and len(span) >= 2:
                start = int(span[0])
                end = int(span[1])
                raw_span_text = contract_text[start:end]

            # Case 2: span is {"text": "..."}
            elif isinstance(span, dict):
                raw_span_text = str(span.get("text") or "")

            # Unknown span format: skip safely
            else:
                continue

            if not raw_span_text:
                continue

            # Split the RAW span text into raw sentences.
            raw_sentences = re.split(r"(?<=[.!?])\s+", raw_span_text)

            for raw_sentence in raw_sentences:
                raw_sentence = raw_sentence.strip()
                normalized_sentence = re.sub(r"\s+", " ", raw_sentence).strip()

                if not raw_sentence:
                    continue

                # If the model quote is inside this sentence after normalization,
                # return the RAW sentence from the contract.
                if cleaned_quote in normalized_sentence:
                    return raw_sentence

                # If the model copied only the beginning of the sentence,
                # return the full RAW sentence.
                if len(cleaned_quote) >= 40 and cleaned_quote[:40] in normalized_sentence:
                    return raw_sentence

        # If repair failed, return the original quote.
        return quote

    except Exception:
        # Very important: never let quote repair break the whole prediction.
        return quote
    


def _trace_from_prediction(
    *,
    h_id: str,
    hypothesis: str,
    prediction: dict,
    contract_text: str,
    contract_spans: list,
    retrieved: list[RetrievedSpan],
    started_at: str,
    latency_ms: float,
    model_id: str,
) -> HypothesisTrace:
    label = str(prediction.get("label") or "NOT_MENTIONED").upper().strip()
    if label not in LABELS:
        label = "NOT_MENTIONED"

    confidence = _clamp_float(prediction.get("confidence", 0.5))
    evidence_ids = _clean_evidence_ids(prediction.get("evidence_span_ids", []), len(contract_spans))
    quote = str(prediction.get("verbatim_quote") or "").strip() or None

    if label == "NOT_MENTIONED":
        evidence_ids = []
        quote = None
    else:
        quote = _repair_quote(
            quote=quote,
            evidence_ids=evidence_ids,
            contract_text=contract_text,
            contract_spans=contract_spans,
        )

    grounded = all(0 <= i < len(contract_spans) for i in evidence_ids)
    quote_ok = True if not quote else quote in contract_text

    failures = []
    if label != "NOT_MENTIONED" and not evidence_ids:
        failures.append({
            "validator": "groundedness_check",
            "message": "Evidence-required label has no span ids.",
            "related_hypothesis_id": h_id,
        })
        grounded = False

    if not quote_ok:
        failures.append({
            "validator": "quote_integrity_check",
            "message": "verbatim_quote is not an exact substring of the analyzed contract.",
            "related_hypothesis_id": h_id,
        })

    return {
        "hypothesis_id": h_id,
        "label": label,
        "confidence": confidence,
        "evidence_spans": evidence_ids,
        "verbatim_quote": quote,
        "groundedness_check": grounded,
        "quote_integrity_check": quote_ok,
        "playbook_result": {
            "status": LABEL_TO_STATUS.get(label, "missing"),
            "reason": str(prediction.get("reason") or "")[:500],
        },
        "agent_metadata": {
            "agent_id": h_id,
            "model": model_id,
            "hypothesis": hypothesis,
            "external_examples_seen": len(retrieved),
        },
        "validation_failures": failures,
        "tool_calls": [],
        "latency_ms": latency_ms,
        "started_at": started_at,
        "ended_at": utc_now_iso(),
    }

def _error_trace(
    *,
    h_id: str,
    hypothesis: str,
    error: Exception,
    retrieved: list[RetrievedSpan],
    started_at: str,
    latency_ms: float,
    model_id: str,
) -> HypothesisTrace:
    return {
        "hypothesis_id": h_id,
        "label": "NOT_MENTIONED",
        "confidence": 0.0,
        "evidence_spans": [],
        "verbatim_quote": None,
        "groundedness_check": False,
        "quote_integrity_check": True,
        "playbook_result": {"status": "error", "reason": str(error)[:500]},
        "agent_metadata": {
            "agent_id": h_id,
            "model": model_id,
            "hypothesis": hypothesis,
            "external_examples_seen": len(retrieved),
        },
        "validation_failures": [{"validator": "schema", "message": str(error)[:500], "related_hypothesis_id": h_id}],
        "tool_calls": [],
        "latency_ms": latency_ms,
        "started_at": started_at,
        "ended_at": utc_now_iso(),
    }


def _clean_evidence_ids(value: Any, span_count: int) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out = []
    for item in value:
        try:
            i = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= i < span_count and i not in out:
            out.append(i)
    return out


def _clamp_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.5
    return round(max(0.0, min(1.0, x)), 4)


def _compute_metrics(traces: list[HypothesisTrace], elapsed_ms: float) -> dict:
    total = len(traces)
    grounded = sum(1 for t in traces if t.get("groundedness_check"))
    quote_ok = sum(1 for t in traces if t.get("quote_integrity_check"))
    by_label: dict[str, int] = {label: 0 for label in sorted(LABELS)}
    for t in traces:
        by_label[str(t.get("label"))] = by_label.get(str(t.get("label")), 0) + 1
    return {
        "hypothesis_count": total,
        "elapsed_ms": elapsed_ms,
        "avg_latency_ms": round(sum(float(t.get("latency_ms", 0.0)) for t in traces) / max(total, 1), 2),
        "groundedness_pass_rate": round(grounded / max(total, 1), 4),
        "quote_integrity_pass_rate": round(quote_ok / max(total, 1), 4),
        "labels": by_label,
    }
