"""
Shared TypedDicts used across all agents and RAG modules.
Import from here — never redefine these shapes locally.
"""
from typing import Dict, List, Optional, TypedDict


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
    action: str      # ACCEPT | CLARIFY | ESCALATE
    rationale: str   # template-filled string from playbook rule


class AgentMetadata(TypedDict):
    agent_id: str    # same as hypothesis_id — which logical agent produced this
    rag_query: str   # query string sent to retrieve() for this task
    rag_hits: int    # number of RetrievedSpans in rag_context
    rag_mode: str    # "vector" | "graph"


class HypothesisTrace(TypedDict):
    """
    Output of one hypothesis worker. Becomes one entry in the RunTrace
    hypothesis_traces array. Extends MS1 trace with agent + RAG metadata.
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
