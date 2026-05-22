"""M3 aggregation, validation, playbook application, and RunTrace writing."""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

from src.constants import HYPOTHESES
from src.playbook_loader import load_playbook
from src.preprocessor import build_chunks
from src.runtrace import build_contract_runtrace, utc_now_iso, write_contract_runtrace
from src.runtrace_recorder import get_active_recorder
from src.types import HypothesisTrace, ToolCallRecord


LABELS = ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED")
OPENROUTER_BASE_MODEL_ID = "qwen/qwen3.5-9b"
N_PARALLEL_AGENTS = 5


def finalize_contract_analysis(
    *,
    contract: dict,
    retrieval_mode: str,
    hypothesis_traces: List[HypothesisTrace],
    started_at: str,
    ended_at: Optional[str] = None,
    session_id: Optional[str] = None,
    parameters: Optional[dict] = None,
    approval_events: Optional[list] = None,
    output_dir: Path = Path("runs_ms3"),
) -> dict:
    """Validate traces, attach playbook results, write the contract RunTrace."""
    ended_at = ended_at or utc_now_iso()
    contract_text = str(contract.get("text") or "")
    chunks = _schema_chunks(contract)
    playbook = load_playbook()
    recorder = get_active_recorder()
    tool_calls = _sanitize_tool_calls(recorder.snapshot() if recorder else [])

    validated: List[HypothesisTrace] = []
    for trace in sorted(hypothesis_traces, key=lambda t: t.get("hypothesis_id", "")):
        validated.append(
            _validate_trace(
                trace=trace,
                contract_text=contract_text,
                span_count=len(contract.get("spans") or []),
                playbook=playbook,
                tool_calls=tool_calls,
                retrieval_mode=retrieval_mode,
            )
        )

    metrics = _metrics(validated, started_at, ended_at)
    contract_payload = {
        "contract_id": _contract_id(contract),
        "source_type": "txt",
        "source_name": contract.get("file_name") or contract.get("source_name") or "unknown",
        "language": "en",
        "hash_sha256": hashlib.sha256(contract_text.encode("utf-8")).hexdigest(),
        "chunks": chunks,
    }
    runtrace = build_contract_runtrace(
        run_id=f"ms3_{contract_payload['contract_id']}_{started_at.replace(':', '').replace('.', '')}",
        contract=contract_payload,
        retrieval_mode=retrieval_mode,
        playbook=playbook.metadata(),
        hypothesis_traces=validated,
        metrics=metrics,
        started_at=started_at,
        ended_at=ended_at,
        tool_calls=tool_calls,
        approval_events=approval_events or [],
        parameters={
            "hypothesis_model": OPENROUTER_BASE_MODEL_ID,
            "n_parallel_agents": N_PARALLEL_AGENTS,
            "temperature": 0.0,
            **(parameters or {}),
        },
        retrieval_context={
            "external_memory": [],
            "contract_evidence": _contract_evidence(validated, chunks),
        },
        run_validations=_run_validations(validated),
        session_id=session_id,
    )
    path = write_contract_runtrace(runtrace, output_dir=output_dir)
    return {
        "status": "completed",
        "contract_id": contract_payload["contract_id"],
        "retrieval_mode": retrieval_mode,
        "hypothesis_traces": validated,
        "metrics": metrics,
        "runtrace_path": str(path),
    }


def _contract_id(contract: dict) -> str:
    raw = str(contract.get("contract_id") or contract.get("id") or "unknown")
    return raw if raw.startswith("doc_") else f"doc_{raw}"


def _schema_chunks(contract: dict) -> list[dict]:
    if contract.get("spans") and contract.get("text"):
        return [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "span": c["span"],
            }
            for c in build_chunks(contract)
        ]
    text = str(contract.get("text") or "")
    return [{"chunk_id": "chunk_000", "text": text, "span": {"char_start": 0, "char_end": len(text)}}]


def _validate_trace(
    *,
    trace: HypothesisTrace,
    contract_text: str,
    span_count: int,
    playbook,
    tool_calls: List[ToolCallRecord],
    retrieval_mode: str,
) -> HypothesisTrace:
    h_id = trace.get("hypothesis_id", "")
    label = trace.get("label", "NOT_MENTIONED")
    if label not in LABELS:
        label = "NOT_MENTIONED"
    evidence = _clean_evidence(trace.get("evidence_spans", []), span_count)
    quote = trace.get("verbatim_quote")
    if label == "NOT_MENTIONED":
        evidence = []
        quote = None

    failures = list(trace.get("validation_failures") or [])
    grounded = len(evidence) == len(trace.get("evidence_spans", []) or [])
    if label != "NOT_MENTIONED" and not evidence:
        grounded = False
        failures.append({
            "validator": "groundedness_check",
            "message": "Evidence-required label has no valid contract span ids.",
            "related_hypothesis_id": h_id,
        })
    quote_ok = True if not quote else str(quote) in contract_text
    if not quote_ok:
        failures.append({
            "validator": "quote_integrity_check",
            "message": "verbatim_quote is not an exact substring of the analyzed contract.",
            "related_hypothesis_id": h_id,
        })

    agent_calls = [c for c in tool_calls if c.get("agent_id") == h_id]
    return {
        **trace,
        "hypothesis_id": h_id,
        "hypothesis_text": trace.get("hypothesis_text") or HYPOTHESES.get(h_id, ""),
        "label": label,
        "confidence": max(0.0, min(1.0, float(trace.get("confidence", 0.0) or 0.0))),
        "evidence_spans": evidence,
        "verbatim_quote": quote,
        "groundedness_check": grounded,
        "quote_integrity_check": quote_ok,
        "playbook_result": playbook.apply(h_id, label, trace.get("confidence")),
        "agent_metadata": {
            "agent_id": h_id,
            "rag_query": f"{h_id}: {HYPOTHESES.get(h_id, '')}",
            "rag_hits": _rag_hits(agent_calls),
            "rag_mode": retrieval_mode,
        },
        "validation_failures": failures,
        "tool_calls": agent_calls,
    }


def _clean_evidence(value: Iterable, span_count: int) -> list[int]:
    out = []
    for item in value or []:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < span_count and idx not in out:
            out.append(idx)
    return out


def _rag_hits(agent_calls: List[ToolCallRecord]) -> int:
    for call in agent_calls:
        if call.get("name") == "retrieve":
            output = call.get("output")
            if isinstance(output, list):
                return len(output)
    return 0


def _sanitize_tool_calls(calls: List[ToolCallRecord]) -> List[ToolCallRecord]:
    sanitized = []
    for call in calls:
        item = dict(call)
        output = item.get("output")
        if isinstance(output, list):
            item["output"] = [
                {**x, "corpus_split": x.get("corpus_split", "train.json")}
                if isinstance(x, dict) and {"text", "doc_id", "span_idx", "score", "hypothesis_annotations"} <= set(x)
                else x
                for x in output
            ]
        sanitized.append(item)
    return sanitized


def _contract_evidence(traces: List[HypothesisTrace], chunks: list[dict]) -> list[dict]:
    by_original = {}
    for chunk in chunks:
        try:
            idx = int(chunk["chunk_id"].split("_")[1])
        except Exception:
            continue
        by_original[idx] = chunk
    items = []
    seen = set()
    for trace in traces:
        for idx in trace.get("evidence_spans", []) or []:
            chunk = by_original.get(idx)
            if not chunk or chunk["chunk_id"] in seen:
                continue
            seen.add(chunk["chunk_id"])
            items.append({
                "chunk_id": chunk["chunk_id"],
                "quote": chunk["text"],
                "relevance_score": 1.0,
                "evidence_source": "contract",
                "span": chunk["span"],
            })
    return items


def _metrics(traces: List[HypothesisTrace], started_at: str, ended_at: str) -> dict:
    total = len(traces)
    grounded = sum(1 for t in traces if t.get("groundedness_check"))
    quote_ok = sum(1 for t in traces if t.get("quote_integrity_check"))
    labels = Counter(t.get("label", "NOT_MENTIONED") for t in traces)
    return {
        "hypothesis_count": total,
        "correct_count": 0,
        "compliant_count": grounded,
        "quote_integrity_count": quote_ok,
        "contract_accuracy": 0.0,
        "groundedness_rate": round(grounded / max(total, 1), 6),
        "quote_integrity_rate": round(quote_ok / max(total, 1), 6),
        "contract_latency_ms": round(sum(float(t.get("latency_ms", 0.0) or 0.0) for t in traces), 3),
        "label_counts": {label: int(labels.get(label, 0)) for label in LABELS},
    }


def _run_validations(traces: List[HypothesisTrace]) -> list[dict]:
    failures = sum(len(t.get("validation_failures", []) or []) for t in traces)
    return [
        {
            "validator_id": "hypothesis_count_check",
            "status": "PASS" if len(traces) == 17 else "FAIL",
            "message": f"Expected 17 hypothesis traces, got {len(traces)}.",
        },
        {
            "validator_id": "trace_validation_failures",
            "status": "PASS" if failures == 0 else "WARN",
            "message": f"{failures} non-fatal validation failure(s) recorded.",
        },
    ]
