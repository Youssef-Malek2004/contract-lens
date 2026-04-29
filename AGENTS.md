# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

ContractLens is a multi-agent NDA review system built for ContractNLI. It classifies each of 17 fixed hypotheses (H01–H17) against an input NDA contract, producing a schema-valid RunTrace with labels, evidence spans, confidence scores, and playbook-driven risk assessments. The conversation agent lets users ask free-form questions about a contract with RAG-augmented answers.

**Dataset:** ContractNLI — 423 train NDAs (32,359 spans), 123 test NDAs.  
**Model family:** Qwen3 only. Orchestrator is Qwen3-4B, NLI core is fine-tuned Qwen3-1.7B + LoRA.

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
            safetensors tokenizers tqdm pyyaml ipykernel
```

All commands below assume `conda activate genai-ms2` and `cd contract-lens/` (repo root).

---

## Repo Layout

```
contract-lens/
├── agent.py                    main CLI entry point
├── playbook.yaml               deterministic rule layer
├── requirements.txt
│
├── src/                        core library — always import from here
│   ├── constants.py
│   ├── preprocessor.py
│   ├── types.py
│   ├── model_loader.py
│   ├── rag_vector.py
│   ├── rag_graph.py
│   ├── conversation_agent.py
│   └── loaders/
│
├── pipeline/                   numbered ML pipeline steps
│   ├── 01_preprocess.py        build SFT .jsonl from train.json
│   ├── 02_finetune.sh          QLoRA training command
│   ├── 03_build_index.py       build FAISS + graph indexes
│   ├── 05_eval_runtrace.py     MS1 batch evaluation
│   └── 05b_debug_single.py     MS1 single-contract debug
│
├── scripts/                    operational utilities
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
├── RunTrace.json               all 123 merged (MS1 output)
├── evaluation.csv              aggregate metrics (MS1 output)
│
├── schemas/
├── architecture/
├── notebooks/
│   └── training_notebook.ipynb
└── docs/
    ├── CONTRIBUTIONS.md
    └── MILESTONE2_PLAN.md
```

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

### Run the conversation agent

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
                --prompt "..." --remote
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

| Component                                 | Status            | Notes                                                    |
| ----------------------------------------- | ----------------- | -------------------------------------------------------- |
| NLI Core (fine-tuned inference)           | **implemented**   | `scripts/quick_infer.py`, `pipeline/05_eval_runtrace.py` |
| Vector RAG                                | **implemented**   | `src/rag_vector.py`                                      |
| GraphRAG                                  | **implemented**   | `src/rag_graph.py`                                       |
| Conversation Agent + CLI                  | **implemented**   | `src/conversation_agent.py`, `agent.py`                  |
| Orchestrator (JSON tool-calling)          | **architectural** | Target: MS3                                              |
| Dispatcher + Hypothesis Pool + Aggregator | **architectural** | Target: MS3                                              |

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

**Remote mode (vllm-mlx server on Apple Silicon):**

```python
model, tokenizer = load_orchestrator(remote=True)
```

Returns a `RemoteOrchestrator` that talks to `http://localhost:8001/v1`. Server must be running `mlx-community/Qwen3-4b-4bit` (lowercase `b` — case-sensitive). Start via `../serving-local-models/serve-qwen3.sh`. NLI/PEFT cannot be served this way.

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

## Outstanding Work (MS3 targets)

These components are **designed in `architecture/architecture.yaml` but not yet coded**:

- Orchestrator Agent — Qwen3-4B with JSON function-call tool dispatch
- `src/dispatcher.py` — pure-Python fan-out that builds 17 `HypothesisTask` dicts and pre-fetches per-hypothesis RAG context
- Hypothesis worker pool — `N_PHYSICAL_AGENTS` workers (env var, default 2) share the loaded Qwen3-1.7B object; workers queue against a single GPU/NPU stream
- Aggregator — pure-Python, validates schema, computes metrics, writes RunTrace JSON

The tool schemas for the orchestrator (JSON function-calling format: `run_nli_core`, `retrieve`, `dispatch_hypothesis_tasks`, `run_hypothesis_workers`, `aggregate_results`, `answer_conversationally`) are fully specified in `architecture/architecture.yaml` → `orchestrator_tools`.

---

## Gitignored Artifacts (must be built locally)

- `data/indexes/` — all RAG indexes (`python pipeline/03_build_index.py --mode all`)
- `*.jsonl` — SFT training data (`python pipeline/01_preprocess.py`)
- `conversation_history.json` — runtime artifact
- `.server_pids` — server PID tracking
- `logs/` — vllm-mlx server logs
- `merged-*/` `mlx-*/` — merged and quantized model artifacts
