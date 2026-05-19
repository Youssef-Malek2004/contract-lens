# Milestone 3 — Work Split (5 Members)

**Due:** 22 May 2026
**Branch:** `ms-3`
**Requirements doc:** [`docs/MILESTONE3_PLAN.md`](MILESTONE3_PLAN.md) — read this first
**Architecture spec:** [`architecture/architecture.yaml`](../architecture/architecture.yaml) — living doc

---

## Deliverables Checklist (from MS3 PDF)

| #   | Deliverable                                                                                   | Owner   |
| --- | --------------------------------------------------------------------------------------------- | ------- |
| a   | Codebase on new `ms-3` branch                                                                 | All     |
| b   | Combined evaluation CSV (MS1 + MS3 columns, same 123-NDA split)                               | M5      |
| c   | Zip of all RunTraces from the MS3 evaluation                                                  | M5      |
| d   | Contributions markdown (`docs/CONTRIBUTIONS.md`)                                              | M5 (collects from all) |

---

## Shared Foundations (must exist before others can integrate)

These land on **day 1** so the rest of the team can code against stable shapes.

| Foundation                                  | Owner | Where                                            |
| ------------------------------------------- | ----- | ------------------------------------------------ |
| `src/types.py` extensions (incl. `ToolCallRecord`, `HypothesisTrace` v3, session metadata) | M1    | extends existing MS2 file                        |
| Tool-call recorder mechanism                | M1    | `src/runtrace_recorder.py` (new)                 |
| RunTrace v3 schema + writer                 | M1    | `schemas/runtrace_schema.json` + `src/runtrace.py` (new) |
| OpenRouter constants                        | M1    | `src/loaders/_constants.py`                      |

Everyone else imports `RetrievedSpan`, `HypothesisTask`, `HypothesisTrace`, `ToolCallRecord` from `src/types.py` and writes audit events through the recorder.

---

## Integration Cheatsheet (M1 foundations landed)

The four foundation files are live on `ms-3`. Use them like this — no other patterns are supported.

### What landed

| File | What it gives you |
| --- | --- |
| `src/loaders/_constants.py` | `OPENROUTER_ORCHESTRATOR_ID = "qwen/qwen3.5-27b"`, `OPENROUTER_BASE_MODEL_ID = "qwen/qwen3.5-9b"`, `N_PARALLEL_AGENTS = 5` |
| `src/types.py` | `ToolCallRecord`, `ApprovalEvent`, `RetrievalModeSwitchEvent`, `ValidationFailure`, `ConversationTurn`, `ContractAnalysisRunTrace`, `ConversationSessionRunTrace`; `HypothesisTrace` has v3 optional fields (`validation_failures`, `tool_calls`, `latency_ms`, `started_at`, `ended_at`) |
| `src/runtrace_recorder.py` | `RunTraceRecorder`, `record_tool_call(...)` ctx, `@recorded_tool(...)` decorator (sync + async), `set_active_recorder(...)` / `use_recorder(...)` (ContextVar-backed) |
| `src/runtrace.py` | `build_contract_runtrace(...)`, `build_session_runtrace(...)`, `write_contract_runtrace(...)`, `write_session_runtrace(...)`, `make_session_id(contract_id)`, `utc_now_iso()` |
| `schemas/runtrace_schema.json` | v3.0-ms3 — `oneOf` discriminator on `runtrace_type` between `contract_analysis` and `conversation_session` |

### Recorder — every tool call must go through this

**Context-manager form** (when you call a tool inline):

```python
from src.runtrace_recorder import record_tool_call

with record_tool_call("retrieve",
                     {"query": q, "mode": mode, "top_k": 5},
                     agent_id="H06") as call:
    result = rag_vector.retrieve(q, mode=mode, top_k=5)
    call.set_output(result)
```

**Decorator form** (when you own the tool handler):

```python
from src.runtrace_recorder import recorded_tool

@recorded_tool("get_span")
def get_span(idx: int, *, agent_id: str = "orchestrator", contract):
    return contract["chunks"][idx]["text"]
```

- `agent_id` is `"orchestrator"` for M1/M4 calls, `"H01"`…`"H17"` for M2's hypothesis-worker calls.
- The decorator reads `agent_id` from a kwarg of the same name by default — pass it through in workers.
- Works on `async def` callables too — concurrent workers each see the recorder via `ContextVar`.

### Recorder lifecycle (M4 owns this)

```python
from src.runtrace_recorder import RunTraceRecorder, set_active_recorder

# At REPL boot (agent.py):
recorder = RunTraceRecorder()
set_active_recorder(recorder)
# …all tool calls in any module now flow into `recorder`…
events = recorder.snapshot()   # used to assemble the session RunTrace
```

If no recorder is active (e.g. unit tests), `record_tool_call` is a no-op — calls still run normally.

### `ToolCallRecord` shape (what gets written)

```json
{
  "agent_id": "H06",
  "name": "retrieve",
  "arguments": {"query": "consultants third-party", "mode": "vector", "top_k": 5},
  "output":    {"hits": 5, "top_score": 0.87},
  "count": 1,
  "started_at": "2026-05-20T12:34:56.789Z",
  "duration_ms": 142.301
}
```

`count` is the per-`(agent_id, name)` ordinal — the recorder assigns it; you don't.

### Writers — no one opens RunTrace files directly

**M3's aggregator** (per `run_full_analysis` invocation):

```python
from src.runtrace import build_contract_runtrace, write_contract_runtrace, utc_now_iso

rt = build_contract_runtrace(
    run_id=f"run_{contract['contract_id']}",
    contract=contract,
    retrieval_mode=mode,
    playbook=playbook_meta,
    hypothesis_traces=validated_traces,   # 17 of them
    metrics=metrics,
    tool_calls=recorder.snapshot(),
    approval_events=approval_events,
    started_at=started_iso,
    ended_at=utc_now_iso(),
    session_id=session_id,                # optional back-ref
)
path = write_contract_runtrace(rt)        # → runs_ms3/runtrace_doc_<contract_id>.json
```

**M4's TUI** (on every turn + on `/exit`):

```python
from src.runtrace import build_session_runtrace, write_session_runtrace, make_session_id

session_id = make_session_id(contract_id)   # "<contract_id>_<ISO8601>"
rt = build_session_runtrace(
    session_id=session_id,
    contract_id=contract_id,
    retrieval_mode=session.active_mode,
    started_at=session.started_at,
    conversation_history=session.history,
    tool_calls=recorder.snapshot(),
    approval_events=session.approval_events,
    retrieval_mode_switches=session.mode_switches,
    referenced_contract_runtraces=session.contract_runtraces,
)
write_session_runtrace(rt)                  # → runs_ms3/session_<session_id>.json
```

Both writers are atomic (tempfile + `os.replace`) — safe to call every turn.

### Pipeline entry signature (M2 must expose)

```python
# src/run_full_analysis.py
async def run_full_analysis(contract_id: str, retrieval_mode: str) -> dict:
    """Returns a summary dict the orchestrator surfaces back to the user."""
```

M1's `tools.py` will `await` this from the `run_full_analysis` tool handler.

### Approval-gate event contract (M1 ↔ M4, day 5)

M1 emits an `ApprovalEvent` payload; M4's TUI renders the prompt, blocks on Y/N, returns the result. M1 records it; M4 also appends it to the session RunTrace via `approval_events=...` on `build_session_runtrace`.

```json
{
  "tool": "run_full_analysis",
  "arguments": {"contract_id": "doc_001", "retrieval_mode": "vector"},
  "approved": true,
  "approved_by": "user",
  "timestamp": "2026-05-20T12:34:56.789Z"
}
```

### Don'ts

- ✗ Don't add fields to `HypothesisTrace` / `ToolCallRecord` without M1 sign-off — the schema validator will reject them.
- ✗ Don't `open()` / `json.dump` into `runs_ms3/` directly — go through `write_*_runtrace`.
- ✗ Don't log tool calls by hand — `record_tool_call` / `@recorded_tool` is the only audit path.
- ✗ Don't re-instantiate `OpenRouterLoader` ad-hoc with hardcoded model IDs — read from `src/loaders/_constants.py`.

---

## Member 1 — Orchestrator + Tools + Audit Infrastructure

**Slice:** the brain + everything tool-related + the audit data contract.

**Files to create / update:**

```
src/orchestrator.py            ← Qwen3.5-27B orchestrator agent, intent routing, SSE streaming
src/tools.py                   ← retrieve · lookup_hypothesis · run_full_analysis handlers
src/runtrace_recorder.py       ← cross-cutting tool-call recorder (used by everyone)
src/runtrace.py                ← Contract + Session RunTrace builders/writers
src/types.py                   ← extend with ToolCallRecord, validation_failures, session types
schemas/runtrace_schema.json   ← v3 (contract_analysis + conversation_session flavours)
src/loaders/_constants.py      ← OPENROUTER_ORCHESTRATOR_ID = "qwen/qwen3.5-27b"
                                  OPENROUTER_BASE_MODEL_ID    = "qwen/qwen3.5-9b"
```

**Scope:**

- Orchestrator agent: load orchestrator model via `OpenRouterLoader`, system prompt explaining the three tools + when to converse vs tool-call, OpenAI-style function-calling protocol, conversation-history injection.
- **SSE streaming** of orchestrator response into a Python iterator the TUI consumes.
- **Three tool handlers** as Python callables, each wrapped by the tool-call recorder:
  - `retrieve(query, mode, top_k, hypothesis_id?, label_filter?)` — delegates to existing `src/rag_vector.py` / `src/rag_graph.py`.
  - `lookup_hypothesis(h_id)` — reads cached HypothesisTrace from this session's prior `run_full_analysis` result.
  - `run_full_analysis(contract_id, retrieval_mode)` — **the integration point with Member 2 + 3**. Calls into their pipeline entry function. Surfaces approval requirement to the TUI (Member 4) before firing.
- **Approval gate**: emits a structured approval-request event the TUI consumes; blocks until the TUI returns Y/N.
- **Tool-call recorder** (`src/runtrace_recorder.py`): a thin decorator / context manager every agent uses to log `{agent_id, name, arguments, output, count, started_at, duration_ms}`. One global recorder per active session — Members 2/3/4 push events into it.
- **RunTrace writer** (`src/runtrace.py`): two writers — `write_contract_runtrace()` (called by aggregator), `write_session_runtrace()` (called by TUI on exit / periodically). Both share the v3 envelope; everyone calls into here, nobody else opens runtrace files directly.
- **`schemas/runtrace_schema.json` v3**: contract_analysis + conversation_session shapes, full `tool_calls` and `approval_events` definitions. Must validate the JSON the writer produces.

**Depends on:** nothing (foundation slot — lands first).
**Blocks:** Members 2, 3, 4, 5.

**Coordination:**

- Publish `ToolCallRecord` shape + recorder API on day 1 so M2/M3/M4 can integrate.
- Publish `run_full_analysis(contract_id, retrieval_mode) → dict` signature so M2/M3 know the entry function to expose.

---

## Member 2 — `run_full_analysis` Pipeline · LLM Side

**Slice:** the half of the heavy pipeline that touches LLMs.

**Files to create / update:**

```
src/dispatcher.py              ← build 17 HypothesisTasks + per-task RAG pre-fetch
src/hypothesis_agent.py        ← qwen3.5-9b worker w/ retrieve + get_span tools
src/run_full_analysis.py       ← orchestrates dispatcher → agents → aggregator (M3)
```

**Scope:**

- **Dispatcher** (`src/dispatcher.py`):
  - Pure-Python fan-out. Takes the analyzed contract + active retrieval mode.
  - For each of the 17 hypotheses: call `retrieve(hypothesis_text, mode, hypothesis_id=Hn, top_k=5)` to pre-fetch RAG context.
  - Emit 17 `HypothesisTask` dicts (no inference here).
- **Hypothesis Agent** (`src/hypothesis_agent.py`):
  - One async worker function. Takes a `HypothesisTask`, calls qwen3.5-9b via `OpenRouterLoader`.
  - Worker has its **own** tool surface: `retrieve` (additional context) + `get_span(idx)` (read span text from analyzed contract). Tools wrapped by M1's recorder with `agent_id=Hn`.
  - Prompts model to emit JSON: `{label, evidence_spans, confidence, verbatim_quote}`.
  - Returns a partially-filled `HypothesisTrace` (validation fields filled by M3's aggregator).
- **Concurrency** (`src/run_full_analysis.py`):
  - Top-level entry function: `run_full_analysis(contract_id, retrieval_mode) → dict`. This is exactly the signature M1 calls.
  - Uses `asyncio.gather` with a `Semaphore(N_PARALLEL_AGENTS)` (default 5) to dispatch the 17 workers concurrently.
  - Hands the collected list of `HypothesisTrace`s to M3's aggregator.
  - Returns aggregator output to M1.

**Depends on:** M1's `src/types.py`, recorder, and `OpenRouterLoader` ID constants.
**Coordinates with:** M3 on the `HypothesisTask` → `HypothesisTrace` boundary.
**Blocks:** M5 (eval can't run until pipeline is end-to-end).

---

## Member 3 — `run_full_analysis` Pipeline · Validation Side

**Slice:** the half of the heavy pipeline that does pure-Python validation + playbook.

**Files to create / update:**

```
src/aggregator.py              ← validate · apply playbook · compute metrics · flag failures
src/playbook_loader.py         ← load + index playbook.yaml (unedited, MS1)
```

**Scope:**

- **Playbook loader** (`src/playbook_loader.py`):
  - Read `playbook.yaml` unedited (MS3 req d).
  - Build a lookup table `{hypothesis_id: {label: rule}}`.
  - No LLM. Pure Python.
- **Aggregator** (`src/aggregator.py`):
  - Receives the list of 17 partial `HypothesisTrace`s from M2's pipeline.
  - **Per-trace validation** (no LLM):
    - `groundedness_check`: every `evidence_spans[i]` is a valid index in the analyzed contract.
    - `quote_integrity_check`: `verbatim_quote` appears as an exact substring of the contract text.
    - schema validation against M1's RunTrace v3 schema.
  - On any validation failure → write the reason into `validation_failures: [...]` on the trace. **No re-run** (resolved decision §6.2 in the plan).
  - **Apply playbook**: for each trace, look up the rule by `(hypothesis_id, label)` and attach `playbook_result: {severity, action, rationale}`.
  - **Contract-level metrics**: label accuracy (n/a until gold labels are joined — M5 does the join), groundedness rate, quote-integrity rate, total latency. M5 computes the MS1-comparable metrics from the written RunTraces; M3 only writes per-contract values.
  - Calls **M1's `write_contract_runtrace()`** to persist (does not open files directly).
  - Returns a summary dict to M2's `run_full_analysis()` for surfacing back to the orchestrator.

**Depends on:** M1's types + RunTrace writer + schema; M2's pipeline orchestration calling shape.
**Coordinates with:** M2 on the `HypothesisTrace` partial→complete boundary.
**Blocks:** M5.

---

## Member 4 — TUI / REPL / CLI

**Slice:** the face — everything the user sees and types.

**Files to create / update:**

```
agent.py                       ← REPL entry point (rewrite for MS3)
src/tui.py                     ← rendering, slash commands, approval prompts
src/session.py                 ← per-session state: history, active RAG mode, session_id
```

**Scope:**

- **REPL** (`agent.py`):
  - Persistent shell. CLI flags: `--contract`, `--idx`, `--dev`/`--verbose`, `--prompt` (one-shot), `--eval` (M5's batch entry).
  - On boot: derive `session_id = "<contract_id>_<ISO8601>"`. Open `runs_ms3/session_<id>.json` via M1's session writer (incremental writes).
- **TUI rendering** (`src/tui.py`):
  - **User mode** (default): status icons during tool execution (`retrieving…`, `analyzing H06…`, etc.), streamed answer, citation block with `[contract]` vs `[external]` separated.
  - **Developer mode** (`--dev` or `/dev`): everything in user mode **plus** orchestrator `<think>` stream, raw tool-call JSON, per-hypothesis worker output as it streams, per-stage timings.
  - **Streaming consumer**: reads M1's orchestrator SSE iterator and renders token-by-token.
  - **Per-hypothesis progress** during `run_full_analysis`: live list of `[H01] ✓ ENTAILED (1.1s)` lines as workers complete (subscribe to M2/M3's progress events — coordinate API).
- **Slash commands** (`src/tui.py` handler):
  - `/dev` · `/user` · `/verbose` — mode toggle
  - `/vector-rag` · `/graph-rag` — switch active RAG branch (mutates `src/session.py` state; logged to session RunTrace)
  - `/analyze` — force `run_full_analysis` request (still goes through approval)
  - `/reset` — clear conversation history for current contract
  - `/exit` — quit (finalises session RunTrace)
- **Approval prompt UI**: when M1 emits an approval-request event, render the Claude-Code-style prompt, block on Y/N, return result. Log to `approval_events` in session RunTrace via M1's writer.
- **Session state** (`src/session.py`): conversation_history (persisted to `conversation_history.json` for backward compat with MS2 + into session RunTrace), active retrieval mode, dev/user flag, cached `run_full_analysis` results for `lookup_hypothesis` (so M1's tool can read them).

**Depends on:** M1's orchestrator iterator + recorder + session-RunTrace writer.
**Coordinates with:** M2/M3 on progress-event format (one short message per hypothesis completion).
**Blocks:** M5 only loosely (eval runs in `--eval` non-REPL mode).

---

## Member 5 — Evaluation + Submission Packaging

**Slice:** the closer — turn the working system into the deliverables.

**Files to create / update:**

```
pipeline/06_eval_ms3.py        ← batch all 123 test NDAs through run_full_analysis
evaluation.csv                  ← combined MS1 + MS3 columns, regenerated
runs_ms3/                       ← per-contract RunTraces from MS3 evaluation
runs_ms3.zip                    ← submission artifact
docs/CONTRIBUTIONS.md           ← collected from all members
```

**Scope:**

- **MS3 evaluation runner** (`pipeline/06_eval_ms3.py`):
  - Headless batch mode — bypasses the TUI. Iterates `data/test.json` (123 NDAs).
  - For each contract: call `src.run_full_analysis.run_full_analysis(contract_id, retrieval_mode)` directly (no orchestrator routing — heavy tool is invoked unconditionally).
  - Reads back the per-contract RunTrace from `runs_ms3/runtrace_doc_NNN.json` via M1's schema.
  - Joins each `HypothesisTrace` with the gold label from `data/test.json` to compute `label_accuracy`.
  - Aggregates: **Label Accuracy, Groundedness, Quote Integrity Pass Rate, Avg Latency** — the same four MS1 used.
- **Combined CSV** (`evaluation.csv`):
  - One row per contract (or per hypothesis instance — match MS1's existing shape exactly).
  - Columns: `contract_id`, `hypothesis_id` (or rolled-up), MS1 columns, MS3 columns side-by-side.
  - Reproduces or imports MS1's column from the existing `evaluation.csv` (don't re-run MS1 if values are already trustworthy — but verify).
- **Submission packaging**: `zip -r runs_ms3.zip runs_ms3/` for deliverable (c).
- **`docs/CONTRIBUTIONS.md`**: collect 2–3 lines from each member, finalise on submission day.

**Depends on:** end-to-end pipeline (M1 + M2 + M3) working; can use stubs for the eval-runner scaffolding while M1–M3 finalise.
**Independent of:** M4's TUI (eval is headless).

---

## Dependency Graph

```
Day 1 — foundations land
└── M1: types.py · runtrace_recorder · runtrace writer · schema · OpenRouter constants
         │
         ├── Day 2–4 (parallel, all depend on M1 foundations)
         │   ├── M1 (continued): orchestrator.py + tools.py + approval gate + SSE streaming
         │   ├── M2: dispatcher · hypothesis_agent · run_full_analysis orchestration
         │   ├── M3: aggregator · playbook_loader · validation
         │   └── M4: TUI · REPL · slash commands · session state
         │           ↑
         │           depends loosely on M1's orchestrator iterator
         │
         ├── Day 5 — integration
         │   └── M1 ↔ M2/M3: wire run_full_analysis tool to pipeline entry
         │   └── M1 ↔ M4:    wire orchestrator iterator + approval gate to TUI
         │   └── M2/M3 ↔ M4: wire per-hypothesis progress events
         │
         └── Day 5–7 — closer
             └── M5: eval runner · evaluation.csv · ZIP · CONTRIBUTIONS.md
                     (eval scaffolding from day 2; final run after integration)
```

**Critical path:** M1 → (M2 + M3) → M5.
**M4 in parallel** with M2/M3 once M1's orchestrator iterator + recorder lands.

---

## Shared Files — Coordination Required

| File                              | Primary Owner | Touches                                                      |
| --------------------------------- | ------------- | ------------------------------------------------------------ |
| `src/types.py`                    | M1            | All — import only, no edits without M1 sign-off              |
| `src/runtrace_recorder.py`        | M1            | M2/M3/M4 import the decorator/context manager                |
| `src/runtrace.py`                 | M1            | M3 calls `write_contract_runtrace()`, M4 calls `write_session_runtrace()` |
| `schemas/runtrace_schema.json`    | M1            | M5 validates against it                                      |
| `src/loaders/_constants.py`       | M1            | M2 reads `OPENROUTER_BASE_MODEL_ID`                          |
| `src/run_full_analysis.py`        | M2            | M1 calls it from the `run_full_analysis` tool handler        |
| `src/session.py`                  | M4            | M1's `lookup_hypothesis` tool reads cached traces from here  |
| `agent.py`                        | M4            | M5's `--eval` flag bypasses the REPL via a separate code path |
| `docs/CONTRIBUTIONS.md`           | M5 (collects) | Everyone contributes 2–3 lines                               |
| `evaluation.csv`                  | M5            | Sole owner                                                   |
| `runs_ms3/`                       | M3 (writes) · M5 (reads) | M3's aggregator writes via M1's writer; M5 reads for metrics |

---

## Final File Structure After MS3

```
contract-lens/
├── agent.py                       ← M4 (rewritten)
├── playbook.yaml                  (MS1 — unedited per req d)
│
├── src/
│   ├── constants.py               (MS1)
│   ├── preprocessor.py            (MS1)
│   ├── types.py                   ← M1 (extended for MS3)
│   ├── rag_vector.py              (MS2 — unchanged)
│   ├── rag_graph.py               (MS2 — unchanged)
│   ├── orchestrator.py            ← M1
│   ├── tools.py                   ← M1
│   ├── runtrace_recorder.py       ← M1
│   ├── runtrace.py                ← M1
│   ├── dispatcher.py              ← M2
│   ├── hypothesis_agent.py        ← M2
│   ├── run_full_analysis.py       ← M2
│   ├── aggregator.py              ← M3
│   ├── playbook_loader.py         ← M3
│   ├── tui.py                     ← M4
│   ├── session.py                 ← M4
│   ├── conversation_agent.py      (MS2 — kept for reference / may be folded into orchestrator)
│   └── loaders/                   (MS2 — constants updated by M1)
│
├── schemas/
│   ├── playbook_schema.json       (MS2 — unedited)
│   └── runtrace_schema.json       ← M1 (v3.0-ms3)
│
├── pipeline/
│   ├── 01_preprocess.py           (MS1)
│   ├── 02_finetune.sh             (MS1)
│   ├── 03_build_index.py          (MS2)
│   ├── 05_eval_runtrace.py        (MS1 — kept for MS1 column reproduction if needed)
│   ├── 05b_debug_single.py        (MS1)
│   └── 06_eval_ms3.py             ← M5
│
├── runs/                          (MS1 outputs — unchanged)
├── runs_ms3/                      ← M3 writes · M5 reads
├── evaluation.csv                 ← M5 (combined MS1 + MS3)
├── runs_ms3.zip                   ← M5 (submission artifact)
│
└── docs/
    ├── MILESTONE2_PLAN.md         (MS2)
    ├── MILESTONE3_PLAN.md         (this milestone — requirements)
    ├── MILESTONE3_WORKSPLIT.md    (this doc)
    └── CONTRIBUTIONS.md           ← M5 collects
```

---

## Hard Constraints (carried from `MILESTONE3_PLAN.md` §7)

| Constraint           | Detail                                                                   |
| -------------------- | ------------------------------------------------------------------------ |
| Same model family    | Qwen only                                                                |
| No fine-tuned model  | Forbidden by MS3 req (f) — adapter kept in repo for MS1 reproduction only |
| Inference backend    | OpenRouter only                                                          |
| Retrieval corpus     | `data/train.json` only — never index `data/test.json`                    |
| Evidence grounding   | RAG = reasoning aid; cited evidence must come from the analyzed contract |
| Playbook             | Used unedited                                                            |
| Tool-call audit      | Every agent's every tool invocation logged to RunTrace                   |
| Test split           | Same 123 NDAs as MS1                                                     |
| Heavy tool gate      | `run_full_analysis` requires explicit user approval (skipped in M5's `--eval`) |
| Worker concurrency   | `N_PARALLEL_AGENTS = 5` (env-tunable)                                    |
| Aggregator policy    | Flag failures; no auto-retry                                             |
| RAG selection        | One branch active per session — toggle via `/vector-rag` ⇄ `/graph-rag`  |
| Session ID           | `<contract_id>_<ISO8601-timestamp>`                                      |
| Streaming            | OpenRouter SSE for orchestrator response                                 |
