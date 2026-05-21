"""
src/tools.py

The three tool handlers the orchestrator's function-calling protocol
dispatches to (MS3 plan §2.4). Every invocation is wrapped by the active
RunTraceRecorder so the audit chain required by MS3 req (h) is complete.

Wiring — the handlers read their dependencies via a ContextVar-bound
ToolContext rather than module-level globals or threaded arguments:

    from src.tools import set_tool_context, ToolContext
    from src import rag_vector, rag_graph

    set_tool_context(ToolContext(
        session=session,           # M4's SessionState (duck-typed)
        rag_vector=rag_vector,     # module exposing .retrieve(query, top_k, ...)
        rag_graph=rag_graph,
        pipeline=run_full_analysis,            # M2's async entry function
        approval_gate=tui.request_approval,    # M4's async Y/N prompt
    ))

The handlers are also importable directly — e.g. M2's hypothesis workers
call `retrieve(query, mode="vector", hypothesis_id="H06", agent_id="H06")`
and get the same audit logging without going through the orchestrator.
"""
from __future__ import annotations

import contextvars
import inspect
from dataclasses import dataclass
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Protocol,
)

from src.runtrace_recorder import record_tool_call, recorded_tool
from src.runtrace import utc_now_iso
from src.types import (
    ApprovalEvent, HypothesisTrace, RetrievedSpan,
)


# ── Protocols (avoids hard imports / cycles with M2 + M4) ───────────────────


class RagRetriever(Protocol):
    """Shape both src.rag_vector and src.rag_graph already satisfy."""
    def retrieve(  # noqa: D401
        self,
        query: str,
        top_k: int = 5,
        hypothesis_id: Optional[str] = None,
        label_filter: Optional[str] = None,
    ) -> List[RetrievedSpan]: ...


class SessionLike(Protocol):
    """Minimal shape M4's SessionState must expose for the tools to work."""
    cached_traces: Dict[str, HypothesisTrace]
    approval_events: List[ApprovalEvent]
    retrieval_mode: str  # "vector" | "graph"


PipelineFn      = Callable[..., Awaitable[dict]]   # M2's run_full_analysis
ApprovalGateFn  = Callable[[str, dict], Awaitable[bool]]


# ── Tool context (bind once at session boot) ────────────────────────────────


@dataclass
class ToolContext:
    session: SessionLike
    rag_vector: Optional[RagRetriever] = None
    rag_graph: Optional[RagRetriever] = None
    pipeline: Optional[PipelineFn] = None
    approval_gate: Optional[ApprovalGateFn] = None


_tool_ctx: contextvars.ContextVar[Optional[ToolContext]] = (
    contextvars.ContextVar("contract_lens_tool_ctx", default=None)
)


def set_tool_context(ctx: Optional[ToolContext]) -> None:
    _tool_ctx.set(ctx)


def get_tool_context() -> ToolContext:
    ctx = _tool_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "No ToolContext bound. Call set_tool_context(ToolContext(...)) "
            "before invoking any tool."
        )
    return ctx


# ── Tool handlers ───────────────────────────────────────────────────────────


@recorded_tool("retrieve")
def retrieve(
    query: str,
    mode: str = "vector",
    top_k: int = 5,
    hypothesis_id: Optional[str] = None,
    label_filter: Optional[str] = None,
    *,
    agent_id: str = "orchestrator",
) -> List[RetrievedSpan]:
    """
    External-memory lookup over the training corpus.

    `mode` picks 'vector' (FAISS) vs 'graph' (networkx ConceptGraph).
    `hypothesis_id` + `label_filter` are forwarded for re-ranking — exactly
    the signature M2's dispatcher uses for per-task pre-fetch.

    Results are reasoning aids only — the orchestrator + workers MUST NOT
    cite them as evidence; final evidence comes from the analyzed contract.
    """
    ctx = get_tool_context()
    retriever = ctx.rag_vector if mode == "vector" else ctx.rag_graph
    if retriever is None:
        raise RuntimeError(f"No {mode!r} retriever bound on the tool context")
    return retriever.retrieve(
        query,
        top_k=top_k,
        hypothesis_id=hypothesis_id,
        label_filter=label_filter,
    )


@recorded_tool("lookup_hypothesis")
def lookup_hypothesis(
    h_id: str,
    *,
    agent_id: str = "orchestrator",
) -> Optional[HypothesisTrace]:
    """
    Read a cached per-hypothesis result from a prior run_full_analysis in
    this session. Returns None if no analysis has run for `h_id` yet —
    the orchestrator should then either run the analysis or tell the user.
    """
    ctx = get_tool_context()
    return ctx.session.cached_traces.get(h_id)


async def run_full_analysis(
    contract_id: str,
    retrieval_mode: Optional[str] = None,
    contract_override: Optional[dict] = None,
    *,
    agent_id: str = "orchestrator",
) -> dict:
    """
    The heavy tool. Two stages: (1) request user approval via the bound
    gate; (2) on approval, await M2's pipeline. The approval event is
    recorded on the session regardless of outcome; the tool call itself is
    recorded by the active RunTraceRecorder via the surrounding context
    manager.

    NOT decorated with @recorded_tool because the gate happens *before* the
    pipeline runs — we want one ToolCallRecord covering the whole flow
    (including a declined approval) and the timing to reflect that.

    Bypassing the gate: leave `approval_gate=None` on the ToolContext
    (M5's --eval mode does this).
    """
    ctx = get_tool_context()
    mode = retrieval_mode or ctx.session.retrieval_mode
    arguments = {
    "contract_id": contract_id,
    "retrieval_mode": mode,
    "has_contract_override": contract_override is not None,
}

    with record_tool_call("run_full_analysis", arguments, agent_id=agent_id) as call:
        approved = True
        if ctx.approval_gate is not None:
            approved = await ctx.approval_gate("run_full_analysis", arguments)

        ctx.session.approval_events.append({
            "tool": "run_full_analysis",
            "arguments": arguments,
            "approved": approved,
            "approved_by": "user",
            "timestamp": utc_now_iso(),
        })

        if not approved:
            result = {
                "status": "declined",
                "reason": "User declined to run the full 17-hypothesis analysis.",
            }
            call.set_output(result)
            return result

        if ctx.pipeline is None:
            raise RuntimeError("No pipeline bound on the tool context")

        result = await _maybe_await(ctx.pipeline(contract_id, mode, contract_override=contract_override))

        # Stash hypothesis traces so subsequent lookup_hypothesis calls hit cache.
        for trace in (result or {}).get("hypothesis_traces", []) or []:
            h_id = trace.get("hypothesis_id")
            if h_id:
                ctx.session.cached_traces[h_id] = trace

        call.set_output(result)
        return result


async def _maybe_await(value: Any) -> Any:
    """Accept both sync and async pipeline implementations from M2."""
    if inspect.isawaitable(value):
        return await value
    return value


# ── Function-calling schemas exposed to the orchestrator ────────────────────


TOOL_SCHEMAS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Retrieve example spans from the training corpus to inform "
                "your reasoning. These are reasoning aids only — never cite "
                "them as evidence and never quote them as the contract's text."
            ),
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["vector", "graph"],
                        "description": "Retrieval backend; defaults to the session's active mode.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                    "hypothesis_id": {
                        "type": "string",
                        "pattern": "^H(0[1-9]|1[0-7])$",
                        "description": "Re-rank to prefer spans annotated for this hypothesis.",
                    },
                    "label_filter": {
                        "type": "string",
                        "enum": ["ENTAILED", "CONTRADICTED", "NOT_MENTIONED"],
                        "description": "Used with hypothesis_id to prefer that label.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_hypothesis",
            "description": (
                "Read the cached per-hypothesis result (label, evidence_spans, "
                "verbatim_quote, confidence) from a prior run_full_analysis in "
                "this session. Returns null if the analysis hasn't run yet."
            ),
            "parameters": {
                "type": "object",
                "required": ["h_id"],
                "properties": {
                    "h_id": {
                        "type": "string",
                        "pattern": "^H(0[1-9]|1[0-7])$",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_full_analysis",
            "description": (
                "Run the full 17-hypothesis analysis on the current contract. "
                "Heavy operation — REQUIRES USER APPROVAL. Only call this when "
                "the user explicitly asks for a full review, types /analyze, or "
                "the question genuinely cannot be answered without all 17 labels."
            ),
            "parameters": {
                "type": "object",
                "required": ["contract_id"],
                "properties": {
                    "contract_id":    {"type": "string"},
                    "retrieval_mode": {"type": "string", "enum": ["vector", "graph"]},
                },
            },
        },
    },
]


TOOL_HANDLERS: Dict[str, Callable[..., Any]] = {
    "retrieve":          retrieve,
    "lookup_hypothesis": lookup_hypothesis,
    "run_full_analysis": run_full_analysis,
}
