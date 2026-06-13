# Milestone 3 — Preliminary Plan (Requirements)

**Due:** 22 May 2026
**Branch:** `ms-3`
**Architecture spec:** `architecture/architecture.yaml`
**Scope:** requirements only — no work split, no deep implementation detail.

---

## 1. Source Requirements (from `Milestone_3.pdf`)


| Ref            | Requirement                                                                                                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| a              | **17-hypothesis analysis mode** — given a contract, output each hypothesis + label + evidence in JSON                                                                                   |
| b              | **Conversation mode (legal assistant)** — upload a contract and ask free-form questions; answers must be insightful, correct, and cited, with external sources clearly marked when used |
| c              | **Auditability** via RunTrace schema — one RunTrace per contract **and** one RunTrace per conversation session                                                                          |
| d              | **Playbook integration** — `playbook.yaml` from MS1 used unedited                                                                                                                       |
| e              | **CLI** with flags + the ability to **converse** (not single-shot)                                                                                                                      |
| f              | **No fine-tuned model** — same family allowed (Qwen)                                                                                                                                    |
| g              | **Vector RAG and GraphRAG both present** — usage left to design                                                                                                                         |
| h              | **RunTrace must include tool calls** (name, args, output, count) **per agent**                                                                                                          |
| § eval         | Run on the same 123-NDA test split from MS1; collect the same metrics                                                                                                                   |
| § agentic      | Must have multiple agents with **feedback, memory, and tool calls**                                                                                                                     |
| § deliverables | New `ms-3` branch · combined evaluation CSV (MS1 + MS3) · zip of all RunTraces · contributions markdown                                                                                 |


---

## 2. System Design Decisions (resolved in discussion)

### 2.1 Entry point — no mode flag

A single CLI / REPL. The user never picks "analyze" vs "converse". The **orchestrator decides** per turn whether to:

- answer conversationally (the default, no tool call needed), or
- invoke a tool (`retrieve`, `lookup_hypothesis`, or `run_full_analysis`).

### 2.2 `run_full_analysis` requires user approval

Heavy tools surface a confirmation prompt in the TUI before firing — Claude-Code-style.
Example: *"I'd like to run the full 17-hypothesis analysis on this contract (~Xs). Proceed? [Y/n]"*
Approval is recorded in the session RunTrace as part of the tool call.

### 2.3 Inference — OpenRouter only


| Role                           | Model                                          |
| ------------------------------ | ---------------------------------------------- |
| Orchestrator                   | `qwen/qwen3.5-27b` (OpenRouter)                |
| Hypothesis agents (×17)        | `qwen/qwen3.5-9b` (OpenRouter)                 |
| Conversation answer generation | same orchestrator model                        |
| Embeddings (RAG)               | local `sentence-transformers/all-MiniLM-L6-v2` |


No local Qwen weights. No fine-tuned adapter in the loop. `src/loaders/_constants.py` updated; `LocalLoader` and `VllmLoader` remain in the codebase but are not used in the MS3 pipeline.

### 2.4 Tool surface

**Orchestrator tools:**


| Tool                | Purpose                                                              | Notes                      |
| ------------------- | -------------------------------------------------------------------- | -------------------------- |
| `retrieve`          | RAG lookup (vector / graph / hybrid)                                 | Used freely per turn       |
| `lookup_hypothesis` | Read cached per-hypothesis result from prior `run_full_analysis`     | Fast — no inference        |
| `run_full_analysis` | Trigger the 17-hypothesis fan-out + playbook + per-contract RunTrace | **Requires user approval** |


Dropped vs MS2: `run_nli_core`, `answer_conversationally`, `dispatch_hypothesis_tasks`, `run_hypothesis_workers`, `aggregate_results`.
The dispatch / workers / aggregate steps now live **inside** `run_full_analysis` as internal sub-stages, not exposed to the orchestrator.

**Hypothesis-agent tools** (each of the 17 workers can use these):


| Tool       | Purpose                                                                    |
| ---------- | -------------------------------------------------------------------------- |
| `retrieve` | Additional RAG context if the worker's pre-fetched context is insufficient |
| `get_span` | Fetch full text of a span by index from the analyzed contract              |


### 2.5 Agent inventory (satisfies "multiple agents w/ feedback, memory, tool calls")


| Agent                | Type                      | Memory                                                              | Tools                                          | Feedback                                                                                                                                                      |
| -------------------- | ------------------------- | ------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Orchestrator         | LLM (27B)                 | Conversation history + cached analysis                              | retrieve, lookup_hypothesis, run_full_analysis | Synthesises and reports tool results to user                                                                                                                  |
| Hypothesis agent ×17 | LLM (9B)                  | Task dict + per-agent RAG context                                   | retrieve, get_span                             | Output validated by Aggregator; failures are flagged in the trace (no auto re-run)                                                                            |
| Dispatcher           | Python                    | —                                                                   | (calls retrieve internally)                    | Builds task queue; runs up to **5 hypothesis agents in parallel** (`N_PARALLEL_AGENTS=5`)                                                                     |
| Aggregator           | Python                    | —                                                                   | —                                              | Validates each hypothesis trace (groundedness, quote integrity), applies playbook, **flags failed traces** in the RunTrace — does not auto-retry              |
| Conversation layer   | LLM (27B, = orchestrator) | Persistent history (`conversation_history.json`) + session RunTrace | (shares orchestrator tools)                    | Cites contract vs external sources distinctly                                                                                                                 |


### 2.6 RAG strategy

Both indexes stay. **No hybrid scoring.** Exactly one branch is active per session and is selected via REPL toggle:

| Command       | Effect                                                                              |
| ------------- | ----------------------------------------------------------------------------------- |
| `/vector-rag` | Switch the session to FAISS vector retrieval (default at REPL startup).             |
| `/graph-rag`  | Switch the session to networkx GraphRAG retrieval.                                  |

The active branch is propagated into every `retrieve` tool call (`mode` argument) and recorded in the session RunTrace + each contract RunTrace's `run_metadata.retrieval_mode`. Switching mid-session is allowed; the switch event is logged.

### 2.7 Playbook integration

`playbook.yaml` is loaded unedited (per requirement d). Applied by the Aggregator post-hoc: for each `HypothesisTrace`, look up the matching playbook rule and attach `{severity, action, rationale}`.

### 2.8 RunTrace v3 schema

Two RunTrace flavours:

**Contract RunTrace** (`runs_ms3/runtrace_doc_NNN.json`) — per-contract analysis output.

- All MS2 fields, plus:
- `schema_version: "3.0-ms3"`
- `runtrace_type: "contract_analysis"`
- `tool_calls: [{agent_id, name, arguments, output, count, timestamp}]`
- `approval_events: [{tool, approved_by, timestamp}]`

**Session RunTrace** (`runs_ms3/session_<contract_id>_<timestamp>.json`) — per-conversation auditability.

- `runtrace_type: "conversation_session"`
- `session_id` (format: `<contract_id>_<ISO8601-timestamp>`), `contract_id`
- `retrieval_mode` — active RAG branch, plus a log of any mid-session `/vector-rag` ↔ `/graph-rag` switches
- Full `conversation_history`
- All `tool_calls` made across the session
- References to any `contract_analysis` RunTraces produced during the session

### 2.9 TUI experience (Claude-Code-style)

Persistent REPL on `python agent.py --contract ... --idx N`.

**Two display modes:**


| Mode                         | What the user sees                                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **User mode** (default)      | Status icons during tool execution · final answer streamed · citation block (contract vs external clearly separated) · approval prompts for heavy tools                      |
| **Developer mode** (verbose) | Everything in user mode **plus** orchestrator `<think>` reasoning · raw tool-call JSON (name, args, output) · per-hypothesis worker output as it streams · timings per stage |


Toggle via slash command in REPL: `/dev`, `/user`, `/verbose` (alias). Also settable from CLI: `--verbose` / `--dev` boots into developer mode.

User-mode example:

```
> what's the deal with consultants under this NDA?

  retrieving similar clauses (vector)…  5 spans
  drafting…

ANSWER
──────
Yes, with restrictions. The Receiving Party may disclose to consultants
bound by equivalent confidentiality terms — see §4.2.

Sources
  [contract] §4.2, §4.5
  [external] doc_42 span 15 (H06=ENTAILED, training corpus — reference only)
```

Developer-mode example (same turn):

```
> what's the deal with consultants under this NDA?

  [orchestrator] thinking…
  <think>The user is asking about third-party disclosure. I should retrieve
   relevant spans before answering…</think>

  [tool_call] retrieve(query="consultants third-party disclosure", mode="vector", top_k=5)
  [tool_result] 5 spans returned (top score 0.87)
  …
ANSWER
──────
…
```

Approval prompt (always visible, in both modes):

```
  ⚠  Orchestrator wants to run: run_full_analysis(contract=test.json#0)
     This will perform 17 LLM calls (~Xs total) and write a RunTrace.
     Proceed? [Y/n] _
```

### 2.10 CLI shape

```bash
python agent.py --contract data/test.json --idx 0           # → REPL, user mode
python agent.py --contract data/test.json --idx 0 --dev     # → REPL, developer mode
python agent.py --contract data/test.json --idx 0 --prompt "..."  # one-shot, no REPL
python agent.py --eval                                       # batch: all 123 contracts → runs_ms3/
```

Slash commands inside REPL:

| Command                    | Effect                                                                |
| -------------------------- | --------------------------------------------------------------------- |
| `/dev`, `/verbose`         | Switch to developer mode                                              |
| `/user`                    | Switch back to user mode                                              |
| `/vector-rag`              | Use the FAISS vector RAG branch (default)                             |
| `/graph-rag`               | Use the networkx GraphRAG branch                                      |
| `/analyze`                 | Force-trigger `run_full_analysis` (still requires approval)           |
| `/reset`                   | Clear conversation history for the current contract                   |
| `/exit`                    | Quit the REPL                                                         |

### 2.11 Evaluation

- Run the agentic pipeline (`run_full_analysis` triggered programmatically) over all 123 test NDAs.
- Same metrics as MS1: **Label Accuracy, Groundedness, Quote Integrity Pass Rate, Avg Latency**.
- Output: single `evaluation.csv` with **MS1 and MS3 columns side-by-side** on the same 2,091 hypothesis instances.
- Outputs go to `runs_ms3/` (new dir, doesn't overwrite MS1 `runs/`).
- Final delivery: ZIP of `runs_ms3/` + `evaluation.csv` + `docs/CONTRIBUTIONS.md`.

---

## 3. Deliverables Checklist (per MS3 PDF)


| #   | Deliverable                         | Location                   |
| --- | ----------------------------------- | -------------------------- |
| a   | Codebase on new branch              | `ms-3`                     |
| b   | Combined evaluation CSV (MS1 + MS3) | `evaluation.csv`           |
| c   | Zip of all RunTraces                | `runs_ms3/*.json` (zipped) |
| d   | Contributions markdown              | `docs/CONTRIBUTIONS.md`    |


---

## 4. What Already Exists (from MS2)

Carries over unchanged:

- Vector RAG (`src/rag_vector.py`) + FAISS indexes
- GraphRAG (`src/rag_graph.py`) + networkx indexes
- `src/types.py` shared TypedDicts (extend with new RunTrace types only)
- `src/loaders/` factory pattern — `OpenRouterLoader` used in MS3
- `playbook.yaml` (used unedited per requirement d)
- MS1 evaluation infrastructure (kept for the MS1 column in the combined CSV)

---

## 5. New Surface Area for MS3


| Component                                                                                     | Status           |
| --------------------------------------------------------------------------------------------- | ---------------- |
| Orchestrator agent (intent routing, tool calling, conversation)                               | new              |
| `run_full_analysis` flow: Dispatcher → 17 hypothesis workers → Aggregator                     | new              |
| Approval-gated tool invocation                                                                | new              |
| Tool-call recorder (cross-cutting)                                                            | new              |
| Session RunTrace + contract RunTrace v3                                                       | new              |
| TUI REPL with user / developer modes                                                          | new              |
| Slash-command handler                                                                         | new              |
| Combined MS1/MS3 evaluation runner                                                            | new              |
| `OPENROUTER_ORCHESTRATOR_ID` → `qwen/qwen3.5-27b`, hypothesis-agent model → `qwen/qwen3.5-9b` | constants update |


---

## 6. Resolved Decisions

| #   | Decision                      | Resolution                                                                                                                                                                                       |
| --- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Hypothesis-worker concurrency | **`N_PARALLEL_AGENTS = 5`** concurrent OpenRouter calls. Credits-funded, comfortably under provider caps.                                                                                        |
| 2   | Aggregator feedback loop      | **Flag-only** — validation failures recorded in the `HypothesisTrace` and RunTrace; no automatic re-run. Keeps cost predictable and the trace honest.                                            |
| 3   | Streaming UX                  | **Yes** — OpenRouter SSE streaming for the orchestrator response so the answer appears live in the REPL.                                                                                         |
| 4   | Session ID scheme             | **`<contract_id>_<ISO8601-timestamp>`** — human-readable, sorts chronologically, no uuid dependency.                                                                                             |
| 5   | RAG selection                 | **No hybrid.** Branch is chosen per session via REPL toggle: `/vector-rag` (default) and `/graph-rag`. Active branch is recorded in the session RunTrace and in every `retrieve` tool call. |

---

## 7. Hard Constraints (inherited + new)


| Constraint          | Detail                                                                   |
| ------------------- | ------------------------------------------------------------------------ |
| Same model family   | Qwen only                                                                |
| No fine-tuned model | Forbidden by MS3 req (f)                                                 |
| Inference backend   | OpenRouter only for MS3                                                  |
| Retrieval corpus    | `data/train.json` only — never index `data/test.json`                    |
| Evidence grounding  | RAG = reasoning aid; cited evidence must come from the analyzed contract |
| Playbook            | Used unedited                                                            |
| Tool-call audit     | Every agent's every tool invocation logged to RunTrace                   |
| Test split          | Same 123 NDAs as MS1                                                     |
| Heavy tool gate     | `run_full_analysis` requires explicit user approval                      |


