"""
Shared TypedDicts used across all agents and RAG modules.
Import from here — never redefine these shapes locally.

MS3 extensions (additions to MS2):
  - ToolCallRecord         — one tool invocation, used by every agent
  - ApprovalEvent          — user Y/N gate for heavy tools (run_full_analysis)
  - RetrievalModeSwitchEvent — /vector-rag ⇄ /graph-rag toggle log
  - ValidationFailure      — non-fatal flag attached to a HypothesisTrace
  - ConversationTurn       — one user/assistant exchange in session history
  - ContractAnalysisRunTrace, ConversationSessionRunTrace
                           — top-level v3 envelopes (one per flavour)
  - HypothesisTrace gains optional v3 fields (validation_failures, tool_calls,
    latency_ms, started_at, ended_at). These are populated by M3's aggregator
    and M2's worker — MS2 callers that don't populate them remain valid.
"""
from typing import Any, Dict, List, Optional, TypedDict


class RetrievedSpan(TypedDict):
    """
    One result from retrieve(). Identical shape from both vector RAG and
    GraphRAG — callers never need to know which branch produced it.
    """
    text: str                           # verbatim span text from training contract
    doc_id: str                         # training contract identifier
    span_idx: int                       # original index from build_chunks()
    score: float                        # cosine sim (vector) or concept score (graph)
    hypothesis_annotations: Dict[str, str]  # {"H06": "ENTAILED", ...} — gold labels
                                            # empty dict if span never cited as evidence


class HypothesisTask(TypedDict):
    """
    One item in the dispatcher's task queue. Self-contained — a hypothesis
    worker needs nothing else to complete its job.
    """
    hypothesis_id: str                  # "H01"…"H17"
    hypothesis_text: str                # full hypothesis statement
    label: str                          # ENTAILED | CONTRADICTED | NOT_MENTIONED
    evidence_spans: List[int]           # span indices from NLI Core output
    contract_chunks: List[dict]         # full analyzed contract chunks (for quote extraction)
    rag_context: List[RetrievedSpan]    # pre-fetched training examples for this hypothesis+label


class PlaybookResult(TypedDict):
    severity: str    # LOW | MEDIUM | HIGH
    action: str      # ACCEPT | CLARIFY | ESCALATE | NEGOTIATE
    rationale: str   # template-filled string from playbook rule


class AgentMetadata(TypedDict):
    agent_id: str    # same as hypothesis_id — which logical agent produced this
    rag_query: str   # query string sent to retrieve() for this task
    rag_hits: int    # number of RetrievedSpans in rag_context
    rag_mode: str    # "vector" | "graph"


# ── MS3 additions ───────────────────────────────────────────────────────────


class ToolCallRecord(TypedDict, total=False):
    """
    One tool invocation by one agent. Written by `runtrace_recorder` and
    accumulated into both the contract RunTrace (per-hypothesis-agent calls)
    and the session RunTrace (orchestrator calls across the conversation).

    Required at write time: agent_id, name, arguments, count, started_at,
    duration_ms. `output` is set when the call completes successfully;
    `error` is set instead if it raised.
    """
    agent_id: str          # "orchestrator" | "H01"…"H17"
    name: str              # "retrieve" | "lookup_hypothesis" | "run_full_analysis" | "get_span"
    arguments: dict
    output: Any
    count: int             # per-(agent_id, name) call ordinal, 1-indexed
    started_at: str        # ISO8601 UTC, ms precision
    duration_ms: float
    error: str


class ApprovalEvent(TypedDict):
    """User Y/N gate recorded when a heavy tool is invoked."""
    tool: str              # tool name being gated, e.g. "run_full_analysis"
    arguments: dict        # arguments the orchestrator wanted to pass
    approved: bool         # True = Y, False = N
    approved_by: str       # "user"
    timestamp: str         # ISO8601 UTC


class RetrievalModeSwitchEvent(TypedDict):
    """Logged when the user runs /vector-rag or /graph-rag mid-session."""
    from_mode: str         # "vector" | "graph"
    to_mode: str
    timestamp: str


class ValidationFailure(TypedDict, total=False):
    """
    Non-fatal flag attached by M3's aggregator. The trace is *kept*; the
    pipeline does NOT auto-retry (resolved decision §6.2).
    """
    validator: str         # "groundedness_check" | "quote_integrity_check" | "schema"
    message: str
    related_hypothesis_id: str


class ConversationTurn(TypedDict, total=False):
    """One exchange in the REPL session. Persisted to the session RunTrace."""
    role: str              # "user" | "assistant" | "system"
    content: str
    timestamp: str         # ISO8601 UTC
    citations: List[dict]  # optional citation block payload


class HypothesisTrace(TypedDict, total=False):
    """
    Output of one hypothesis worker. Becomes one entry in the contract
    RunTrace `hypothesis_traces` array.

    MS2 required fields (hypothesis_id, label, confidence, evidence_spans,
    verbatim_quote, groundedness_check, quote_integrity_check, playbook_result,
    agent_metadata) are still expected by readers; v3 adds optional fields
    (validation_failures, tool_calls, latency_ms, started_at, ended_at) that
    M3 and M2 populate. TypedDict total=False keeps the contract permissive so
    a partially-filled worker output (e.g. before M3 has run validation) still
    type-checks — readers must check key presence.
    """
    hypothesis_id: str
    label: str
    confidence: float               # 0.0–1.0
    evidence_spans: List[int]       # span indices from the analyzed contract only
    verbatim_quote: Optional[str]   # exact substring of analyzed contract; None if NOT_MENTIONED
    groundedness_check: bool        # all cited span indices valid in analyzed contract
    quote_integrity_check: bool     # verbatim_quote appears as exact substring in contract text
    playbook_result: PlaybookResult
    agent_metadata: AgentMetadata

    # ── MS3 additions ──
    validation_failures: List[ValidationFailure]
    tool_calls: List[ToolCallRecord]
    latency_ms: float
    started_at: str
    ended_at: str


class ContractAnalysisRunTrace(TypedDict, total=False):
    """
    Top-level v3 envelope written per contract analysis. Persisted via
    `src.runtrace.write_contract_runtrace`.
    """
    schema_version: str                            # "3.0-ms3"
    runtrace_type: str                             # "contract_analysis"
    run: dict
    contract: dict
    retrieval_mode: str                            # "vector" | "graph"
    retrieval_context: dict
    playbook: dict
    hypothesis_traces: List[HypothesisTrace]
    metrics: dict
    run_validations: List[dict]
    conversation_history: List[ConversationTurn]
    tool_calls: List[ToolCallRecord]
    approval_events: List[ApprovalEvent]
    session_id: str                                # links back to the session that triggered it


class ConversationSessionRunTrace(TypedDict, total=False):
    """
    Top-level v3 envelope written per REPL session. Persisted via
    `src.runtrace.write_session_runtrace`.
    """
    schema_version: str                            # "3.0-ms3"
    runtrace_type: str                             # "conversation_session"
    session_id: str                                # "<contract_id>_<ISO8601>"
    contract_id: str
    started_at: str
    ended_at: str
    retrieval_mode: str                            # active branch at session end
    retrieval_mode_switches: List[RetrievalModeSwitchEvent]
    conversation_history: List[ConversationTurn]
    tool_calls: List[ToolCallRecord]
    approval_events: List[ApprovalEvent]
    referenced_contract_runtraces: List[str]       # filenames of contract RunTraces produced this session
