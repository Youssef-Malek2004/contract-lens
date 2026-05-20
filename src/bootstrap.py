"""
src/bootstrap.py

One-line wiring for the orchestrator side of the runtime.

Centralises the boot sequence so the demo runner (`scripts/demo_m1.py`)
and M4's `agent.py` both bind the same way:

  - construct a RunTraceRecorder and mark it active
  - build a ToolContext (real or stub retrievers, M2's pipeline, gate)
  - bind the ToolContext

The caller still owns the session object (M4's SessionState) — bootstrap
only sets up the orchestrator-side wiring.

Typical use:

    from src.bootstrap import setup_runtime
    from src.approval import ApprovalGate

    gate = ApprovalGate()
    recorder = setup_runtime(
        session=session,
        pipeline=run_full_analysis,     # M2's async entry function
        approval_gate=gate.request,
        use_real_rag=True,
    )
    orchestrator = Orchestrator()
"""
from __future__ import annotations

from typing import Awaitable, Callable, Optional

from src.runtrace_recorder import RunTraceRecorder, set_active_recorder
from src.tools import (
    ApprovalGateFn,
    PipelineFn,
    RagRetriever,
    SessionLike,
    ToolContext,
    set_tool_context,
)


def setup_runtime(
    *,
    session: SessionLike,
    pipeline: Optional[PipelineFn] = None,
    approval_gate: Optional[ApprovalGateFn] = None,
    rag_vector: Optional[RagRetriever] = None,
    rag_graph: Optional[RagRetriever] = None,
    use_real_rag: bool = True,
    recorder: Optional[RunTraceRecorder] = None,
) -> RunTraceRecorder:
    """
    Bind the recorder + tool context for the current asyncio/sync context.

    Returns the active RunTraceRecorder so the caller can snapshot events
    after each turn (M4 uses this for the session RunTrace).

    - Pass explicit `rag_vector` / `rag_graph` adapters to stub retrieval
      (useful for tests + the demo when indexes aren't built).
    - Leave `pipeline=None` only if you don't intend to call
      `run_full_analysis` — the tool handler will raise if invoked.
    - Pass `approval_gate=None` to bypass approval (eval mode).
    """
    if recorder is None:
        recorder = RunTraceRecorder()
    set_active_recorder(recorder)

    if use_real_rag and rag_vector is None and rag_graph is None:
        # Lazy import — keeps test envs without faiss/networkx from breaking.
        try:
            from src import rag_vector as _rv, rag_graph as _rg
            rag_vector, rag_graph = _rv, _rg
        except ImportError:
            pass

    set_tool_context(ToolContext(
        session=session,
        rag_vector=rag_vector,
        rag_graph=rag_graph,
        pipeline=pipeline,
        approval_gate=approval_gate,
    ))
    return recorder
