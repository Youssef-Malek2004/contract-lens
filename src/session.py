"""
src/session.py

Per-REPL-session state. One SessionState is created at boot and carried
through the entire conversation loop. It is the single source of truth for:

  - which contract is loaded and its id
  - the active RAG branch (vector / graph)
  - developer-mode flag
  - conversation history (persisted into the session RunTrace)
  - approval events, RAG-mode switches, and contract-runtrace back-references
  - the cached run_full_analysis result (so M1's lookup_hypothesis tool can
    read prior analysis without re-running the 17 agents)

M1's lookup_hypothesis tool reads `session.cached_analysis`.
M4's agent.py owns the lifecycle (create → mutate → write RunTrace).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.runtrace import make_session_id, utc_now_iso
from src.types import (
    ApprovalEvent,
    ConversationTurn,
    RetrievalModeSwitchEvent,
)


@dataclass
class SessionState:
    """All mutable state for one REPL session."""

    contract_id: str
    contract: dict
    session_id: str
    started_at: str

    active_mode: str = "vector"
    dev_mode: bool = False

    conversation_history: List[ConversationTurn] = field(default_factory=list)
    approval_events: List[ApprovalEvent] = field(default_factory=list)
    mode_switches: List[RetrievalModeSwitchEvent] = field(default_factory=list)
    contract_runtraces: List[str] = field(default_factory=list)

    cached_analysis: Optional[Dict[str, Any]] = None

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        contract_id: str,
        contract: dict,
        dev_mode: bool = False,
    ) -> "SessionState":
        now = datetime.now(timezone.utc)
        return cls(
            contract_id=contract_id,
            contract=contract,
            session_id=make_session_id(contract_id, when=now),
            started_at=utc_now_iso(),
            dev_mode=dev_mode,
        )

    # ── Mutation helpers ─────────────────────────────────────────────────────

    def add_turn(
        self,
        role: str,
        content: str,
        citations: Optional[list] = None,
    ) -> None:
        turn: ConversationTurn = {
            "role": role,
            "content": content,
            "timestamp": utc_now_iso(),
        }
        if citations:
            turn["citations"] = citations
        self.conversation_history.append(turn)

    def switch_rag_mode(self, to_mode: str) -> None:
        """Record a /vector-rag or /graph-rag switch event."""
        if to_mode == self.active_mode:
            return
        event: RetrievalModeSwitchEvent = {
            "from_mode": self.active_mode,
            "to_mode": to_mode,
            "timestamp": utc_now_iso(),
        }
        self.mode_switches.append(event)
        self.active_mode = to_mode

    def record_approval(self, event: ApprovalEvent) -> None:
        self.approval_events.append(event)

    def add_contract_runtrace_ref(self, filename: str) -> None:
        """Called by agent.py after run_full_analysis produces a contract RunTrace."""
        if filename not in self.contract_runtraces:
            self.contract_runtraces.append(filename)

    def reset_history(self) -> None:
        """Clear conversation history (slash command /reset)."""
        self.conversation_history.clear()
