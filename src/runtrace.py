"""
src/runtrace.py

RunTrace v3 builders + writers. Two flavours share the v3 envelope:

  - contract_analysis   → runs_ms3/runtrace_doc_<contract_id>.json
                          (one per `run_full_analysis` invocation)
  - conversation_session → runs_ms3/session_<session_id>.json
                          (one per REPL session)

Members 2/3 write contract RunTraces via `write_contract_runtrace()`.
Member 4 writes session RunTraces via `write_session_runtrace()`.
No one else opens RunTrace files directly — keep the schema centralised.

All writes are atomic (tempfile + os.replace) so a crash mid-write never
leaves a partial JSON file behind.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.types import (
    ApprovalEvent,
    ContractAnalysisRunTrace,
    ConversationSessionRunTrace,
    ConversationTurn,
    HypothesisTrace,
    RetrievalModeSwitchEvent,
    ToolCallRecord,
)


SCHEMA_VERSION = "3.0-ms3"
DEFAULT_OUTPUT_DIR = Path("runs_ms3")


# ── Time + ID helpers ───────────────────────────────────────────────────────


def utc_now_iso() -> str:
    """ISO8601 UTC with millisecond precision, ending in 'Z'."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def make_session_id(contract_id: str, when: Optional[datetime] = None) -> str:
    """
    Format: `<contract_id>_<ISO8601>` per MS3 plan §6.4.

    Colons in the timestamp are dropped so the id is safe to use as a filename
    on case-sensitive filesystems that disallow ':' (Windows/SMB shares).
    """
    when = when or datetime.now(timezone.utc)
    iso = when.isoformat(timespec="seconds").replace("+00:00", "Z")
    return f"{contract_id}_{iso.replace(':', '')}"


# ── Builders (assemble v3 envelopes from per-stage outputs) ─────────────────


def build_contract_runtrace(
    *,
    run_id: str,
    contract: dict,
    retrieval_mode: str,
    playbook: dict,
    hypothesis_traces: List[HypothesisTrace],
    metrics: dict,
    started_at: str,
    ended_at: Optional[str] = None,
    tool_calls: Optional[List[ToolCallRecord]] = None,
    approval_events: Optional[List[ApprovalEvent]] = None,
    parameters: Optional[dict] = None,
    retrieval_context: Optional[dict] = None,
    run_validations: Optional[list] = None,
    conversation_history: Optional[List[ConversationTurn]] = None,
    session_id: Optional[str] = None,
) -> ContractAnalysisRunTrace:
    """
    Assemble a contract-analysis RunTrace dict ready to hand to
    `write_contract_runtrace`. Fills sensible defaults for empty optional
    arrays so downstream readers never trip over missing keys.
    """
    payload: ContractAnalysisRunTrace = {
        "schema_version": SCHEMA_VERSION,
        "runtrace_type": "contract_analysis",
        "run": {
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": ended_at or utc_now_iso(),
            "framework": "agentic_pipeline",
            "parameters": parameters or {},
        },
        "contract": contract,
        "retrieval_mode": retrieval_mode,
        "retrieval_context": retrieval_context or {"external_memory": [], "contract_evidence": []},
        "playbook": playbook,
        "hypothesis_traces": list(hypothesis_traces),
        "metrics": metrics,
        "run_validations": list(run_validations or []),
        "conversation_history": list(conversation_history or []),
        "tool_calls": list(tool_calls or []),
        "approval_events": list(approval_events or []),
    }
    if session_id:
        payload["session_id"] = session_id
    return payload


def build_session_runtrace(
    *,
    session_id: str,
    contract_id: str,
    retrieval_mode: str,
    started_at: str,
    conversation_history: List[ConversationTurn],
    tool_calls: List[ToolCallRecord],
    ended_at: Optional[str] = None,
    approval_events: Optional[List[ApprovalEvent]] = None,
    retrieval_mode_switches: Optional[List[RetrievalModeSwitchEvent]] = None,
    referenced_contract_runtraces: Optional[List[str]] = None,
) -> ConversationSessionRunTrace:
    """
    Assemble a conversation-session RunTrace dict ready to hand to
    `write_session_runtrace`. Designed for incremental writes — M4's TUI can
    rebuild + rewrite this after every turn.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "runtrace_type": "conversation_session",
        "session_id": session_id,
        "contract_id": contract_id,
        "started_at": started_at,
        "ended_at": ended_at or utc_now_iso(),
        "retrieval_mode": retrieval_mode,
        "retrieval_mode_switches": list(retrieval_mode_switches or []),
        "conversation_history": list(conversation_history),
        "tool_calls": list(tool_calls),
        "approval_events": list(approval_events or []),
        "referenced_contract_runtraces": list(referenced_contract_runtraces or []),
    }


# ── Writers ─────────────────────────────────────────────────────────────────


def write_contract_runtrace(
    runtrace: ContractAnalysisRunTrace,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Persist a contract-analysis RunTrace.
    Filename: `runtrace_doc_<contract_id>.json`.
    """
    runtrace.setdefault("schema_version", SCHEMA_VERSION)
    runtrace.setdefault("runtrace_type", "contract_analysis")
    contract = runtrace.get("contract") or {}
    contract_id = contract.get("contract_id")
    if not contract_id:
        raise ValueError("contract.contract_id is required to name the RunTrace file")
    path = Path(output_dir) / f"runtrace_doc_{contract_id}.json"
    _atomic_write_json(path, runtrace)
    return path


def write_session_runtrace(
    runtrace: ConversationSessionRunTrace,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Persist a conversation-session RunTrace.
    Filename: `session_<session_id>.json`. Safe to call repeatedly to
    snapshot the session — each call atomically overwrites the prior file.
    """
    runtrace.setdefault("schema_version", SCHEMA_VERSION)
    runtrace.setdefault("runtrace_type", "conversation_session")
    session_id = runtrace.get("session_id")
    if not session_id:
        raise ValueError("session_id is required to name the session RunTrace file")
    path = Path(output_dir) / f"session_{session_id}.json"
    _atomic_write_json(path, runtrace)
    return path


# ── Atomic write ────────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
