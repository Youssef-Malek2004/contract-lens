# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

ContractLens is a multi-agent NDA review system built for ContractNLI. It classifies each of 17 fixed hypotheses (H01–H17) against an input NDA contract, producing a schema-valid RunTrace with labels, evidence spans, confidence scores, and playbook-driven risk assessments. The conversation agent lets users ask free-form questions about a contract with RAG-augmented answers.

**Dataset:** ContractNLI — 423 train NDAs (32,359 spans), 123 test NDAs.

**Model family:** Qwen only.

| Milestone | Orchestrator | NLI core / workers |
| --- | --- | --- |
| MS1/MS2 | local `Qwen3-4B` (or OpenRouter `qwen/qwen3-4b`) | fine-tuned `Qwen3-1.7B + LoRA` |
| MS3 | OpenRouter `qwen/qwen3.5-27b` | OpenRouter `qwen/qwen3.5-9b` (17 hypothesis workers, no fine-tuned model per MS3 req f) |

**Current milestone:** MS3 — agentic orchestrator with function-calling tools, audit-logged RunTrace v3, per-session conversation + per-contract analysis flavours. Branch: `ms-3`. See `docs/MILESTONE3_PLAN.md` (requirements) and `docs/MILESTONE3_WORKSPLIT.md` (5-member work split + integration cheatsheet).

---

## Environment

```bash
# Create once (Python 3.9 cannot install transformers from source — needs 3.11)
conda create -n genai-ms2 python=3.11 -y
conda activate genai-ms2

# transformers must be from source — Qwen3 support is not in any PyPI release yet
pip install torch torchvision torchaudio
pip install "git+https://github.com/huggingface/transformers.git"
pip install accelerate peft sentence-transformers \
            faiss-cpu networkx scikit-learn numpy huggingface_hub \
            safetensors tokenizers tqdm pyyaml ipykernel python-dotenv
```

All commands below assume `conda activate genai-ms2` and `cd contract-lens/` (repo root).

---

## Repo Layout

```
contract-lens/
├── agent.py                    MS2 conversation-agent CLI (M4 will rewrite for MS3)
├── playbook.yaml               deterministic rule layer
├── requirements.txt
│
├── src/                        core library — always import from here
│   ├── constants.py            NDA_TO_H, LABEL_MAP, HYPOTHESES, SYSTEM_PROMPT
│   ├── preprocessor.py
│   ├── types.py                ★ MS3-extended — TypedDicts every module imports
│   ├── model_loader.py         legacy MS1/MS2 loader shim
│   ├── rag_vector.py           FAISS vector retrieval (MS2)
│   ├── rag_graph.py            networkx GraphRAG retrieval (MS2)
│   ├── conversation_agent.py   MS2 conversation agent (kept; may fold into orchestrator)
│   │
│   ├── orchestrator.py         ★ MS3 — Qwen3.5-27B + SSE streaming + tool-call loop (M1)
│   ├── tools.py                ★ MS3 — retrieve / lookup_hypothesis / run_full_analysis (M1)
│   ├── approval.py             ★ MS3 — ApprovalGate for run_full_analysis (M1)
│   ├── bootstrap.py            ★ MS3 — setup_runtime() one-call wiring (M1)
│   ├── runtrace.py             ★ MS3 — RunTrace v3 builders + atomic writers (M1)
│   ├── runtrace_recorder.py    ★ MS3 — ContextVar-bound recorder (M1)
│   │
│   ├── dispatcher.py           ▢ MS3 — 17 HypothesisTasks + per-task RAG prefetch (M2)
│   ├── hypothesis_agent.py     ▢ MS3 — async 9B worker (M2)
│   ├── run_full_analysis.py    ▢ MS3 — dispatcher → workers → aggregator (M2)
│   ├── aggregator.py           ▢ MS3 — validation + playbook + metrics (M3)
│   ├── playbook_loader.py      ▢ MS3 — load playbook.yaml unedited (M3)
│   ├── tui.py                  ▢ MS3 — slash commands + rendering (M4)
│   ├── session.py              ▢ MS3 — SessionState (M4)
│   └── loaders/                MS2 — _constants.py updated for MS3 model IDs
│
├── pipeline/                   numbered ML pipeline steps
│   ├── 01_preprocess.py        build SFT .jsonl from train.json
│   ├── 02_finetune.sh          QLoRA training command
│   ├── 03_build_index.py       build FAISS + graph indexes
│   ├── 05_eval_runtrace.py     MS1 batch evaluation
│   ├── 05b_debug_single.py     MS1 single-contract debug
│   └── 06_eval_ms3.py          ▢ MS3 batch — 123 NDAs → runs_ms3/ (M5)
│
├── scripts/                    operational utilities
│   ├── demo_m1.py              ★ MS3 — interactive REPL demo of M1's slice
│   ├── download_models.py      pre-download all models
│   ├── merge_adapter.py        merge LoRA + MLX conversion
│   ├── quick_infer.py          single-contract two-pass inference
│   ├── setup_models.sh
│   └── stop_servers.sh
│
├── tests/                      test suite (run from repo root)
│   ├── test_model_loader.py
│   ├── test_indexes.py
│   └── test_vllm_endpoints.py
│
├── data/
│   ├── train.json              RAG corpus (never index test.json)
│   ├── test.json               evaluation only
│   └── indexes/                gitignored — rebuild locally
│
├── runs/                       123 RunTrace JSONs (MS1 output)
├── runs_ms3/                   ★ MS3 — per-contract + per-session RunTraces v3
├── RunTrace.json               all 123 merged (MS1 output)
├── evaluation.csv              aggregate metrics (MS1; MS3 columns appended by M5)
│
├── schemas/
│   ├── playbook_schema.json
│   └── runtrace_schema.json    ★ bumped to v3.0-ms3 (oneOf on runtrace_type)
├── architecture/
├── notebooks/
│   └── training_notebook.ipynb
└── docs/
    ├── CONTRIBUTIONS.md
    ├── MILESTONE2_PLAN.md
    ├── MILESTONE3_PLAN.md         ★ MS3 requirements
    └── MILESTONE3_WORKSPLIT.md    ★ MS3 5-member work split + integration cheatsheet
```

Legend: ★ landed on `ms-3` · ▢ targeted for MS3, owner shown

---

## Common Commands

### Download models (run once, ~12 GB total)

```bash
python scripts/download_models.py
```

Downloads to `~/.cache/huggingface/`. Run directly in the terminal, **not** via `conda run` — subprocess stdout buffering hides progress bars.

### Smoke-test environment

```bash
python tests/test_model_loader.py
```

Runs three live-streaming tests: Orchestrator (Qwen3-4B, thinking ON), base model (adapter OFF, thinking ON), NLI Core (adapter ON, thinking OFF). All three must PASS.

### Build RAG indexes (required before using the conversation agent)

```bash
python pipeline/03_build_index.py --mode vector    # FAISS index, ~5 min
python pipeline/03_build_index.py --mode graph     # networkx graph, ~3 min
python pipeline/03_build_index.py --mode all       # both
```

Outputs go to `data/indexes/` (gitignored — rebuild locally from `data/train.json`).

### Run the MS3 demo (M1 slice — orchestrator + tools + audit chain)

`scripts/demo_m1.py` is an interactive REPL that exercises the orchestrator, the three function-calling tools, the approval gate, and the audit chain end-to-end. Requires `OPENROUTER_API_KEY` in `.env` (see `.env.example`).

```bash
# Real RAG + real orchestrator, stubbed run_full_analysis pipeline (M2 not landed yet)
python scripts/demo_m1.py --idx 0 --dev

# No FAISS indexes built? Use stub retrievers
python scripts/demo_m1.py --idx 0 --auto-approve --no-rag

# One-shot — no REPL, just run a single prompt and exit
python scripts/demo_m1.py --prompt "what does this NDA say about consultants?" --dev
```

CLI flags: `--contract data/test.json` · `--idx N` · `--dev` (show `<think>`) · `--auto-approve` (skip the run_full_analysis Y/n prompt — used by M5's eval) · `--no-rag` (stub retrievers) · `--prompt ...` (one-shot).

Slash commands inside the REPL: `/dev` `/user` `/vector-rag` `/graph-rag` `/analyze` `/reset` `/audit` `/exit`. `/audit` dumps every recorded tool call so far; `/exit` writes the session RunTrace to `runs_ms3/session_<session_id>.json`.

### Run the conversation agent (MS2)

```bash
python agent.py --contract data/test.json --idx 0 \
                --retrieval vector \
                --prompt "Does this NDA allow sharing with consultants?"

# Use graph retrieval branch:
python agent.py --contract data/test.json --idx 0 \
                --retrieval graph \
                --prompt "What are the termination obligations?"

# Use a vllm-mlx server instead of loading locally (Apple Silicon):
python agent.py --contract data/test.json --idx 0 \
                --retrieval vector \
                --prompt "..." --backend vllm

# Route orchestrator through OpenRouter (requires OPENROUTER_API_KEY in .env):
python agent.py --contract data/test.json --idx 0 \
                --retrieval vector \
                --prompt "..." --backend openrouter
```

Conversation history persists in `conversation_history.json`. History auto-rotates when `--idx` changes.

### Run single-contract NLI inference

```bash
python scripts/quick_infer.py              # first val doc
python scripts/quick_infer.py --idx 3      # 4th val doc
python scripts/quick_infer.py --data data/train.json --idx 0
```

Two-pass inference: pass 1 = NLI classification (adapter ON, thinking OFF), pass 2 = confidence + verbatim quotes (adapter OFF, thinking ON). Prints a results table vs gold labels.

### Run evaluation (MS1, full test split)

```bash
python pipeline/05_eval_runtrace.py         # 123 contracts → runs/ + evaluation.csv
python pipeline/05b_debug_single.py --idx 0 # full model output for one contract
```

### Test RAG indexes

```bash
python tests/test_indexes.py
python tests/test_vllm_endpoints.py        # requires all three vllm-mlx servers running
```

### Regenerate architecture diagram

```bash
conda install -c conda-forge graphviz python-graphviz -y
python architecture/generate_diagram.py
```

### Recompile LaTeX report

```bash
cd architecture && pdflatex report.tex
```

### Merge LoRA adapter for vllm-mlx serving

```bash
python scripts/merge_adapter.py               # merge only
python scripts/merge_adapter.py --convert     # merge + MLX 4-bit conversion
```

---

## Architecture

### Milestone scope

| Component                                          | Status            | Notes                                                          |
| -------------------------------------------------- | ----------------- | -------------------------------------------------------------- |
| NLI Core (fine-tuned inference, MS1/MS2)           | **implemented**   | `scripts/quick_infer.py`, `pipeline/05_eval_runtrace.py`       |
| Vector RAG                                         | **implemented**   | `src/rag_vector.py`                                            |
| GraphRAG                                           | **implemented**   | `src/rag_graph.py`                                             |
| MS2 conversation agent + CLI                       | **implemented**   | `src/conversation_agent.py`, `agent.py`                        |
| MS3 RunTrace v3 schema + writers                   | **implemented**   | `schemas/runtrace_schema.json`, `src/runtrace.py` (M1)         |
| MS3 tool-call recorder (audit chain)               | **implemented**   | `src/runtrace_recorder.py` (M1)                                |
| MS3 orchestrator (Qwen3.5-27B, SSE, tool dispatch) | **implemented**   | `src/orchestrator.py` (M1)                                     |
| MS3 tools (retrieve / lookup / run_full_analysis)  | **implemented**   | `src/tools.py` + `src/approval.py` (M1)                        |
| MS3 demo runner (interactive REPL)                 | **implemented**   | `scripts/demo_m1.py`                                           |
| MS3 dispatcher + hypothesis workers                | **architectural** | `src/dispatcher.py`, `src/hypothesis_agent.py` — owner M2      |
| MS3 aggregator + playbook validation               | **architectural** | `src/aggregator.py`, `src/playbook_loader.py` — owner M3       |
| MS3 TUI + REPL rewrite                             | **architectural** | `src/tui.py`, `src/session.py`, `agent.py` rewrite — owner M4  |
| MS3 batch evaluation                               | **architectural** | `pipeline/06_eval_ms3.py` — owner M5                           |

### Data flow (MS2 conversation agent)

```
User prompt
    │
    ├─ ConversationAgent.run_turn()
    │      ├─ Load contract from test.json (idx N)
    │      ├─ Truncate to MAX_CONTRACT_TOKENS=14,000
    │      ├─ retrieve(query, top_k=5) via --retrieval branch
    │      │      vector: embed query → FAISS search → re-rank → list[RetrievedSpan]
    │      │      graph:  synonym match → concept_index → score → list[RetrievedSpan]
    │      ├─ _build_messages() → system + history (max 5 pairs) + [RAG BLOCK] + CONTRACT + QUESTION
    │      └─ Generate (Qwen3-4B, thinking=True) → strip <think> → return
    │
    └─ conversation_history.json (appended, rotates on contract change)
```

### Data flow (MS1 NLI pipeline, still used for evaluation)

```
Contract → build_chunks() → numbered spans
                │
                ├─ Pass 1: NLI Core (Qwen3-1.7B, adapter ON, thinking=False)
                │      Input:  all spans + 17 hypothesis texts
                │      Output: [{hypothesis_id, label, evidence_spans}] × 17
                │
                └─ Pass 2: Base model (adapter OFF, thinking=True)
                       Input:  NLI results + contract
                       Output: {H01…H17: {confidence, quote}}
```

### Data flow (MS3 agentic system — M1 slice)

```
User prompt
    │
    ├─ Orchestrator.run_turn()  (Qwen3.5-27B on OpenRouter, SSE streaming)
    │      ├─ system prompt + conversation history + [CONTRACT] block
    │      ├─ delta.reasoning  → "think" event
    │      ├─ delta.content    → "content" event
    │      └─ delta.tool_calls → reassemble, dispatch
    │
    ├─ Tool dispatch (TOOL_HANDLERS in src/tools.py)
    │      ├─ retrieve(query, mode, top_k, hypothesis_id?, label_filter?)
    │      │      → src/rag_vector.py OR src/rag_graph.py
    │      ├─ lookup_hypothesis(h_id)
    │      │      → ctx.session.cached_traces[h_id]
    │      └─ run_full_analysis(contract_id, retrieval_mode)
    │             ├─ ApprovalGate.request()  → Y/n on stderr
    │             ├─ append ApprovalEvent to session
    │             └─ await ctx.pipeline()    → M2's src/run_full_analysis.py
    │
    ├─ Every tool call wrapped by RunTraceRecorder (ContextVar-bound)
    │      → ToolCallRecord {agent_id, name, arguments, output, count, started_at, duration_ms}
    │
    └─ On /exit: write_session_runtrace() → runs_ms3/session_<session_id>.json
       After run_full_analysis: write_contract_runtrace() → runs_ms3/runtrace_doc_<id>.json
```

---

## MS3 Audit Chain (Tool-Call Recorder)

MS3 req (h): *every agent's every tool invocation must appear in the RunTrace*. This is enforced by a single cross-cutting recorder in `src/runtrace_recorder.py`.

**Two usage patterns** — pick one per tool site, never log calls by hand:

```python
# 1. Context-manager form (inline tool call)
from src.runtrace_recorder import record_tool_call

with record_tool_call("retrieve", {"query": q, "mode": "vector"}, agent_id="H06") as call:
    result = rag_vector.retrieve(q, top_k=5)
    call.set_output(result)

# 2. Decorator form (you own the handler)
from src.runtrace_recorder import recorded_tool

@recorded_tool("get_span")
def get_span(idx, *, agent_id="orchestrator", contract):
    return contract["chunks"][idx]["text"]
```

`agent_id` is `"orchestrator"` for M1/M4 calls, `"H01"`…`"H17"` for M2's hypothesis workers (one per agent so the audit can attribute every call to a specific worker).

**Lifecycle (M4 owns this at REPL boot):**

```python
from src.runtrace_recorder import RunTraceRecorder, set_active_recorder
recorder = RunTraceRecorder()
set_active_recorder(recorder)   # ContextVar — all async tasks inherit it
```

`asyncio.gather`-fanned workers automatically see the same recorder because the binding lives in `contextvars.ContextVar`. No need to thread it through every layer.

`recorder.snapshot()` returns a list of `ToolCallRecord` dicts ready to serialize into a RunTrace. The schema requires `count` (per-`(agent_id, name)` ordinal) — the recorder assigns this; callers never set it.

**Convenience wrapper for the demo / agent.py:**

```python
from src.bootstrap import setup_runtime

recorder = setup_runtime(
    session=session,                 # any object satisfying SessionLike
    pipeline=run_full_analysis,      # M2's async entry function
    approval_gate=gate.request,      # ApprovalGate.request bound method
    use_real_rag=True,               # False to stub retrievers
)
```

This binds the recorder + `ToolContext` in one call. Both the demo runner and (eventually) M4's `agent.py` use this exact path.

---

## RunTrace v3 (`schemas/runtrace_schema.json`)

The MS3 schema uses `oneOf` on a `runtrace_type` discriminator. Both flavours share `schema_version: "3.0-ms3"` and `additionalProperties: false` so M5's validator catches field drift.

| Flavour | When | File | Required fields |
| --- | --- | --- | --- |
| `contract_analysis` | one per `run_full_analysis` invocation | `runs_ms3/runtrace_doc_<contract_id>.json` | `run`, `contract`, `retrieval_mode`, `playbook`, `hypothesis_traces[17]`, `metrics`, `tool_calls`, `approval_events` |
| `conversation_session` | one per REPL session | `runs_ms3/session_<session_id>.json` | `session_id`, `contract_id`, `started_at`, `retrieval_mode`, `conversation_history`, `tool_calls` |

**Writers** in `src/runtrace.py` — atomic (`tempfile` + `os.replace`). Nobody else opens these files directly.

```python
from src.runtrace import (
    build_contract_runtrace, write_contract_runtrace,    # M3 calls these
    build_session_runtrace, write_session_runtrace,      # M4 calls these
    make_session_id, utc_now_iso,                        # helpers
)

session_id = make_session_id(contract_id)   # "<contract_id>_<ISO8601>" with ':' stripped
write_session_runtrace(build_session_runtrace(
    session_id=session_id, contract_id=contract_id, retrieval_mode="vector",
    started_at=session.started_at,
    conversation_history=session.history,
    tool_calls=recorder.snapshot(),
    approval_events=session.approval_events,
    retrieval_mode_switches=session.mode_switches,
    referenced_contract_runtraces=session.referenced_runtraces,
))
```

`HypothesisTrace` v3 adds optional fields: `validation_failures`, per-trace `tool_calls`, `latency_ms`, `started_at`, `ended_at`. M2 returns a partial trace; M3's aggregator fills validation fields. Both shapes type-check (`TypedDict total=False`).

**Approval events** are recorded separately from tool calls. The `ApprovalEvent` shape:

```python
{
  "tool": "run_full_analysis",
  "arguments": {"contract_id": "doc_001", "retrieval_mode": "vector"},
  "approved": true, "approved_by": "user",
  "timestamp": "2026-05-20T12:34:56.789Z"
}
```

`src/approval.py` ships `ApprovalGate` (auto-approve mode for M5's `--eval`, `prompt_fn` for custom UIs), `console_prompt` (Claude-Code-style Y/n), and `auto_approve_prompt`.

---

## MS3 Hard Constraints (carried from `docs/MILESTONE3_PLAN.md` §7)

| Constraint           | Detail                                                                   |
| -------------------- | ------------------------------------------------------------------------ |
| Same model family    | Qwen only                                                                |
| No fine-tuned model  | Forbidden by MS3 req (f) — adapter kept for MS1 reproduction only        |
| Inference backend    | OpenRouter only                                                          |
| Retrieval corpus     | `data/train.json` only — never index `data/test.json`                    |
| Evidence grounding   | RAG = reasoning aid; cited evidence must come from the analyzed contract |
| Playbook             | Used unedited                                                            |
| Tool-call audit      | Every agent's every tool invocation logged via `src/runtrace_recorder`   |
| Test split           | Same 123 NDAs as MS1                                                     |
| Heavy tool gate      | `run_full_analysis` requires explicit user approval (skipped in `--eval`) |
| Worker concurrency   | `N_PARALLEL_AGENTS = 5` (env-tunable, in `src/loaders/_constants.py`)    |
| Aggregator policy    | Flag failures (`validation_failures: [...]`); no auto-retry              |
| RAG selection        | One branch active per session — toggle via `/vector-rag` ⇄ `/graph-rag`  |
| Session ID           | `<contract_id>_<ISO8601-timestamp>` (colons stripped for filesystem)     |
| Streaming            | OpenRouter SSE for orchestrator response                                 |

---

## Model Loading

**Never call `AutoModelForCausalLM.from_pretrained` directly.** Always go through `src/model_loader.py`.

```python
from src.model_loader import get_device, load_orchestrator, load_nli_model

device = get_device()                          # "mps" | "cuda" | "cpu"
model, tokenizer = load_orchestrator(device)   # Qwen3-4B, thinking=True at generate time
model, tokenizer = load_nli_model(device)      # Qwen3-1.7B + LoRA, adapter ON
```

**Quantization strategy by device:**

- CUDA → `BitsAndBytesConfig` NF4 4-bit. CUDA has native int4 GEMM kernels — genuinely faster and ~2.5× smaller in memory.
- MPS → `dtype=torch.float16`. MPS has no native int4 GEMM kernels; any int4 library dequantizes on every forward pass, adding startup cost with no speedup. float16 is optimal.
- CPU → `dtype=torch.float16` (fallback only).

**Adapter toggle (NLI model only):**

```python
# Adapter ON (default) — fine-tuned NLI Core, thinking=False
output = model.generate(...)

# Adapter OFF — base Qwen3-1.7B, thinking=True
with model.disable_adapter():
    output = model.generate(...)
# adapter back ON after context exit
```

This is an in-place toggle — no weight reloading. `scripts/quick_infer.py` switches between both modes in a single run.

**Tokenizer source:** Load the tokenizer from the adapter repo (`Youssef-Malek/contractnli-vast-ai-qwen3-1.7b`), not from `Qwen/Qwen3-1.7B`. The adapter repo has a patched chat template set during Unsloth training.

**Backend selection (loader factory):**

```python
from src.loaders import get_loader

get_loader("local").load_orchestrator()      # in-process weights (MPS/CUDA/CPU)
get_loader("vllm").load_orchestrator()       # vllm-mlx @ http://localhost:8001/v1
get_loader("openrouter").load_orchestrator() # OpenRouter API
```

`get_loader` dispatches by mode (`local` | `vllm` | `openrouter`). All three return a `ModelHandle` with the same `.stream()` / `.generate()` shape, so callers don't branch on backend.

- **vllm** — server must be running `mlx-community/Qwen3-4b-4bit` (lowercase `b` — case-sensitive). Start via `../serving-local-models/serve-qwen3.sh`. NLI/PEFT cannot be served this way.
- **openrouter** — reads `OPENROUTER_API_KEY` from `.env` (auto-loaded if `python-dotenv` is installed) or env. Defaults to `qwen/qwen3-4b` / `qwen/qwen3-1.7b` (override via `OpenRouterConfig` or kwargs to `get_loader`). `load_nli_model()` raises `NotImplementedError` — the contractnli adapter is local-only; use `LocalLoader` for NLI.

Legacy shim: `src/model_loader.py::load_orchestrator(remote=True)` still works and is equivalent to `get_loader("vllm")`.

**Token streaming pattern (used everywhere):**

```python
from transformers import TextIteratorStreamer
import threading

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0)
gen_thread = threading.Thread(target=lambda: model.generate(**gen_kwargs, streamer=streamer))
gen_thread.start()
for chunk in streamer:
    print(chunk, end="", flush=True)
gen_thread.join()
```

The spinner pattern in `ConversationAgent._generate()` and `scripts/quick_infer.py` works by calling `next(iter(streamer))` before displaying output — this blocks until prefill is complete (first token arrives).

---

## Thinking Policy

Qwen3 models emit optional `<think>…</think>` chain-of-thought blocks controlled by `enable_thinking` in `apply_chat_template()`.

| Model                                 | Adapter | thinking  | Why                                                                                                             |
| ------------------------------------- | ------- | --------- | --------------------------------------------------------------------------------------------------------------- |
| Qwen3-4B (orchestrator)               | N/A     | **True**  | Unmodified checkpoint. Thinking gives better reasoning.                                                         |
| Qwen3-1.7B (NLI Core)                 | **ON**  | **False** | SFT-trained without thinking. Turning it ON injects `<think>` tokens the model never saw, breaking JSON output. |
| Qwen3-1.7B (base / hypothesis agents) | OFF     | **True**  | Unmodified checkpoint. Thinking improves confidence calibration and quote extraction.                           |

Always strip `<think>` blocks before parsing structured output or displaying to users:

```python
import re
clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
```

---

## Shared Types (`src/types.py`)

Import from here — never redefine these shapes locally.

```python
from src.types import RetrievedSpan, HypothesisTask, HypothesisTrace
```

`RetrievedSpan` is returned by both `rag_vector.retrieve()` and `rag_graph.retrieve()` with identical shape. `hypothesis_annotations` is the highest-signal field: `{"H06": "ENTAILED"}` means a human annotator cited this span as evidence for H06=ENTAILED in a training NDA.

**Do not compare `score` across RAG branches.** Vector score is cosine similarity (0–1). Graph score is `matched_concepts / total + 0.5 gold bonus` (unbounded, higher=better).

---

## Hypothesis ID Mapping

ContractNLI uses `nda-N` keys in its annotation JSON. The codebase maps these to `H01`–`H17`, but **the mapping is non-sequential** (gaps exist for hypotheses excluded from this dataset):

```
nda-1 → H01, nda-2 → H02, nda-3 → H03, nda-4 → H04,
nda-5 → H05, nda-7 → H06,   ← gap: nda-6 excluded
nda-8 → H07, nda-10 → H08,  ← gap: nda-9 excluded
nda-11 → H09, nda-12 → H10, nda-13 → H11, nda-15 → H12,  ← gap: nda-14
nda-16 → H13, nda-17 → H14, nda-18 → H15, nda-19 → H16, nda-20 → H17
```

`src/constants.py` has `NDA_TO_H` and `H_TO_NDA`. Both RAG modules include a `normalize_hypothesis_id()` that accepts `H06`, `H6`, `nda-7`, `nda7`, `nda-07` and normalises to the canonical form.

---

## RAG Pipeline Details

### Vector RAG (`src/rag_vector.py`)

Index format:

- `data/indexes/vector/spans.jsonl` — one JSON line per span; row `i` matches FAISS vector `i`
- `data/indexes/vector/faiss.index` — `IndexFlatIP` binary
- `data/indexes/vector/metadata.json` — build metadata

`IndexFlatIP` was chosen over `IndexIVFFlat` because 32k vectors × 384 dims = ~47 MB in RAM with exact search completing in <10ms. IVF variants only pay off at 500k+ vectors.

Both the index vectors and query vectors are L2-normalised before insertion/search, so inner product equals cosine similarity.

Query re-ranking: `retrieve(hypothesis_id="H06", label_filter="ENTAILED")` over-fetches `top_k×3` candidates, then boosts scores for spans that have `hypothesis_annotations["nda-7"] == "ENTAILED"` (+0.3 for any annotation on this hypothesis, +0.5 additional if label matches).

### GraphRAG (`src/rag_graph.py`)

Three node types: `SpanNode` (~32k), `ConceptNode` (19 hardcoded legal terms), `HypothesisNode` (17).

Three edge types:

- `CONTAINS` (Span→Concept): substring match of concept synonyms in lowercased span text. No NLP/NER.
- `CITED_FOR` (Span→Hypothesis, attr: label): from ContractNLI gold annotations — highest-signal edge.
- `INVOLVES` (Hypothesis→Concept): synonym match on hypothesis text.

Index format:

- `data/indexes/graph/graph.pkl` — serialized `networkx.DiGraph`
- `data/indexes/graph/hypothesis_index.json` — `{nda_id: {label: [span_node_ids]}}` — O(1) hypothesis-targeted lookup
- `data/indexes/graph/concept_index.json` — `{canonical_name: [span_node_ids]}` — O(1) concept→spans
- `data/indexes/graph/metadata.json` — build metadata

Two query paths: if `hypothesis_id` is given, use `hypothesis_index` (cite frequency scoring); otherwise lowercase the query, match against synonym lists, union `concept_index` results, score by concept hit count + 0.5 gold bonus.

### RAG usage constraint

Retrieved spans are reasoning aids only. They must **never** appear as cited evidence in the RunTrace or in a user-facing answer — all evidence must come from the analyzed contract. The conversation agent enforces this via the `[RETRIEVAL CONTEXT]` block label and the system prompt.

---

## Data Constraints

| Constraint                   | Detail                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------- |
| Retrieval corpus             | `data/train.json` only — **never index `data/test.json`**                       |
| Evaluation contracts         | `data/test.json` — the same 123 NDAs used in MS1                                |
| One retrieval branch per run | `--retrieval vector` OR `--retrieval graph`, not both                           |
| History persistence          | `conversation_history.json` persists between runs; rotates when `--idx` changes |

---

## Key Constants (`src/constants.py`)

- `NDA_TO_H` / `H_TO_NDA` — bidirectional mapping between ContractNLI annotation keys and H-IDs
- `LABEL_MAP` — `{"Entailment": "ENTAILED", "Contradiction": "CONTRADICTED", "NotMentioned": "NOT_MENTIONED"}`
- `HYPOTHESES` — `{"H01": "...", ..., "H17": "..."}` — the 17 hypothesis texts
- `SYSTEM_PROMPT` — NLI classification system prompt for the fine-tuned model

---

## Conversation Agent Context Budget

`src/conversation_agent.py` enforces a 20,000-token context window:

- `MAX_CONTRACT_TOKENS = 14_000` — contract text ceiling (p90 of test.json is ~5,500 tokens)
- `MAX_HISTORY_TURNS = 5` — older history pairs are trimmed
- `MIN_NEW_TOKENS = 256` — generation is refused if prompt leaves less output budget
- `SAFETY_MARGIN = 256` — slack for chat-template tokens and tokenizer drift

The full contract is re-injected into every user turn so the model has grounding regardless of how much of earlier history fits in attention.

---

## Outstanding Work (MS3)

**Landed on `ms-3` (M1's slice):**

- `src/types.py` MS3 extensions — `ToolCallRecord`, `ApprovalEvent`, `RetrievalModeSwitchEvent`, `ValidationFailure`, `ConversationTurn`, `ContractAnalysisRunTrace`, `ConversationSessionRunTrace`; `HypothesisTrace` v3 optional fields.
- `src/runtrace_recorder.py` — ContextVar-bound recorder + `@recorded_tool` + `record_tool_call` ctx manager.
- `src/runtrace.py` — atomic builders + writers for both RunTrace flavours.
- `schemas/runtrace_schema.json` v3.0-ms3 — oneOf discriminator on `runtrace_type`.
- `src/orchestrator.py` — Qwen3.5-27B on OpenRouter, SSE streaming, OpenAI-style function-calling loop (with tool-call delta reassembly), 8-iteration cap.
- `src/tools.py` — three handlers + `TOOL_SCHEMAS` + `ToolContext` + `set_tool_context()`.
- `src/approval.py` — `ApprovalGate`, console + auto prompts.
- `src/bootstrap.py` — `setup_runtime()` one-call wiring.
- `scripts/demo_m1.py` — interactive REPL exercising the full audit chain.

**Remaining (other members):**

- **M2** — `src/dispatcher.py`, `src/hypothesis_agent.py`, `src/run_full_analysis.py`. Exposes `async def run_full_analysis(contract_id, retrieval_mode) -> dict` which M1's tool handler awaits.
- **M3** — `src/aggregator.py`, `src/playbook_loader.py`. Reads `playbook.yaml` unedited, validates groundedness + quote integrity, attaches `validation_failures`, writes contract RunTrace via M1's writer.
- **M4** — `agent.py` rewrite, `src/tui.py`, `src/session.py`. Sets the active recorder at boot, wires `ApprovalGate.prompt_fn` to the TUI's input queue, persists `conversation_history.json` for back-compat.
- **M5** — `pipeline/06_eval_ms3.py`, combined `evaluation.csv`, `runs_ms3.zip`, `docs/CONTRIBUTIONS.md`.

See `docs/MILESTONE3_WORKSPLIT.md` for the full split and the **Integration Cheatsheet** section (copy-pasteable recorder usage, writer call patterns, wire-format JSON, Don'ts list).

**Legacy MS3 architecture notes:** `architecture/architecture.yaml` predates the MS3 resolved decisions — tool schemas there (`run_nli_core`, `dispatch_hypothesis_tasks`, etc.) are superseded by the three-tool surface in `src/tools.py` per MS3 §2.4.

---

## Gitignored Artifacts (must be built locally)

- `data/indexes/` — all RAG indexes (`python pipeline/03_build_index.py --mode all`)
- `*.jsonl` — SFT training data (`python pipeline/01_preprocess.py`)
- `conversation_history.json` — runtime artifact
- `.server_pids` — server PID tracking
- `logs/` — vllm-mlx server logs
- `merged-*/` `mlx-*/` — merged and quantized model artifacts
