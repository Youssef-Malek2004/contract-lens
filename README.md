# ContractLens

**Dataset:** ContractNLI | **Model family:** Qwen3

---

- [Milestone 2](#milestone-2)
- [Milestone 1](#milestone-1)

---

# Milestone 2

**Branch:** `ms-2` | **Due:** 30 April 2026

## Quick Links

| Resource | Path |
|---|---|
| Architecture spec (living doc) | `architecture/architecture.yaml` |
| Architecture diagram | `architecture/architecture.pdf` |
| Work split + deadlines | `MILESTONE2_PLAN.md` |
| Shared types | `src/types.py` |
| Model loader | `src/model_loader.py` |
| Data | `data/train.json` (RAG) · `data/test.json` (eval only) |

---

## 1. Environment Setup

> **Use the `genai-ms2` env.** The old `generative` env runs Python 3.9 which
> cannot install transformers from source.

```bash
# Create once
conda create -n genai-ms2 python=3.11 -y
conda activate genai-ms2

# Install dependencies
pip install torch torchvision torchaudio
pip install "git+https://github.com/huggingface/transformers.git"
pip install accelerate peft sentence-transformers \
            faiss-cpu networkx scikit-learn numpy huggingface_hub \
            safetensors tokenizers tqdm pyyaml ipykernel

# Register Jupyter kernel (optional)
python -m ipykernel install --user --name genai-ms2 --display-name "genai-ms2"
```

> `transformers` must be installed from source — Qwen3 support requires the
> latest version. Install it **before** other packages, or **last** so nothing
> overwrites it.

---

## 2. Download Models

Run once from your terminal (not `conda run` — it buffers progress bars):

```bash
conda activate genai-ms2
cd contract-lens
python download_models.py
```

| Model | Size | Purpose |
|---|---|---|
| `Qwen/Qwen3-4B` | ~8 GB | Orchestrator + Conversation Agent |
| `Qwen/Qwen3-1.7B` | ~3.5 GB | NLI Core base + Hypothesis Agents |
| `Youssef-Malek/contractnli-vast-ai-qwen3-1.7b` | ~100 MB | Fine-tuned LoRA adapter |
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | Vector RAG embeddings |

Everything goes to `~/.cache/huggingface/` — all scripts find it automatically.

---

## 3. Verify Setup

```bash
python test_model_loader.py
```

Expected: three PASS results with live token streaming. If all three pass, your environment is fully working.

---

## 4. Shared Infrastructure

### 4.1 Types — `src/types.py`

Three TypedDicts shared across all modules. **Import from here, never redefine locally.**

```python
from src.types import RetrievedSpan, HypothesisTask, HypothesisTrace
```

| Type | Produced by | Consumed by |
|---|---|---|
| `RetrievedSpan` | `rag_vector.py` / `rag_graph.py` | Dispatcher, Conversation Agent, Hypothesis Workers |
| `HypothesisTask` | `dispatcher.py` | Hypothesis Worker Pool |
| `HypothesisTrace` | Hypothesis Workers | Aggregator |

```python
class RetrievedSpan(TypedDict):
    text: str                              # verbatim span text from training contract
    doc_id: str                            # training contract id
    span_idx: int                          # original span index
    score: float                           # relevance score (do NOT compare across branches)
    hypothesis_annotations: Dict[str, str] # {"H06": "ENTAILED"} — gold labels for this span
```

### 4.2 Model Loader — `src/model_loader.py`

Handles device detection and quantization for every model. **Never call
`AutoModelForCausalLM.from_pretrained` directly — always go through model_loader.**

```python
from src.model_loader import get_device, load_orchestrator, load_nli_model

device = get_device()   # "mps" | "cuda" | "cpu" — auto-detected

# Orchestrator / Conversation Agent  (Qwen3-4B, thinking ON)
model, tokenizer = load_orchestrator(device)

# NLI Core  (Qwen3-1.7B + LoRA adapter, thinking OFF)
model, tokenizer = load_nli_model(device)

# Switch to base model behaviour (adapter OFF, thinking ON):
with model.disable_adapter():
    output = model.generate(...)
# adapter is back ON after the context exits
```

**Quantization (automatic):**
- MPS → `dtype=torch.float16` (~3.5 GB for 1.7B, ~8 GB for 4B)
- CUDA → `BitsAndBytesConfig NF4 4-bit` (~1 GB for 1.7B, ~2.5 GB for 4B)
- CPU → `dtype=torch.float16` (fallback only)

**Thinking policy:**
- `load_orchestrator()` → `enable_thinking=True`
- `load_nli_model()` with adapter ON → `enable_thinking=False`
- `load_nli_model()` with adapter OFF → `enable_thinking=True`

**Token streaming pattern:**

```python
import threading
from transformers import TextIteratorStreamer

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
thread = threading.Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, ...})
thread.start()
for chunk in streamer:
    print(chunk, end="", flush=True)
thread.join()
```

---

## 5. Per-Member Tasks

### Member 1 — Architecture + Diagram

**Status:** `architecture/architecture.yaml` ✓ · `src/types.py` ✓ · `src/model_loader.py` ✓

Regenerate the PDF if the spec changes:
```bash
conda install -c conda-forge graphviz python-graphviz -y
python architecture/generate_diagram.py
```

---

### Member 2 — Vector RAG (`src/rag_vector.py`)

**Interface to expose** (exact signature — Member 4 depends on this):

```python
from src.types import RetrievedSpan

def retrieve(
    query: str,
    top_k: int = 5,
    hypothesis_id: str = None,   # e.g. "H06" — re-ranks to this hypothesis
    label_filter: str = None,    # "ENTAILED" | "CONTRADICTED" | "NOT_MENTIONED"
) -> list:                       # list[RetrievedSpan]
```

**Index files to produce:**
```
data/indexes/vector/spans.jsonl    # metadata — one JSON line per span
data/indexes/vector/faiss.index    # FAISS IndexFlatIP binary
data/indexes/vector/metadata.json  # {embedding_model, build_timestamp, total_spans}
```

**Steps:**
1. Load `data/train.json` → `build_chunks()` from `src/preprocessor.py` for each doc
2. For each span scan gold annotations to populate `hypothesis_annotations`
3. Embed with `sentence-transformers/all-MiniLM-L6-v2` (384 dims, L2-normalised)
4. Build `faiss.IndexFlatIP` — exact search, ~47 MB for 32k vectors
5. At query time: embed → search → re-rank if `hypothesis_id` given → return `list[RetrievedSpan]`

**Build command:** `python 03_build_index.py --mode vector`

Key imports:
```python
from src.constants import NDA_TO_H, LABEL_MAP
from src.preprocessor import build_chunks
from src.types import RetrievedSpan
```

---

### Member 3 — GraphRAG (`src/rag_graph.py`)

**Same interface signature as Member 2** — callers cannot tell which branch ran.

**Graph structure:**
```
SpanNode ──CONTAINS──► ConceptNode ◄──INVOLVES── HypothesisNode
SpanNode ──CITED_FOR (label)──────────────────► HypothesisNode
```

- **SpanNode** (~32k): one per training span
- **ConceptNode** (~19): hardcoded legal terms with synonym lists — full vocabulary in `architecture/architecture.yaml` → `graph_rag.graph_schema.node_types.ConceptNode.vocabulary`
- **HypothesisNode** (17): from `src/constants.py` `HYPOTHESES`

**Edge creation:**
- `CONTAINS`: `any(syn in span_text.lower() for syn in concept.synonyms)`
- `CITED_FOR`: from gold annotations (same data as Member 2's `hypothesis_annotations`)
- `INVOLVES`: synonym scan on each hypothesis text

**Pre-build two O(1) lookup files:**
```python
# data/indexes/graph/hypothesis_index.json
{"H06": {"ENTAILED": ["doc_42__span_15", ...], "CONTRADICTED": [...], ...}, ...}

# data/indexes/graph/concept_index.json
{"third_party": ["doc_42__span_15", ...], ...}
```

**Query paths:**
1. `hypothesis_id` given → `hypothesis_index[id][label]` → rank by citation count → return
2. Free-text → synonym match → `concept_index` union → score → return

**Build command:** `python 03_build_index.py --mode graph`

Coordinate with Member 2 on the shared `03_build_index.py` entry point before writing it.

---

### Member 4 — Conversation Agent (`src/conversation_agent.py` + `agent.py`)

**CLI:**
```bash
python agent.py --contract data/test.json --idx 0 \
                --retrieval vector \
                --prompt "Does this NDA allow sharing with consultants?"
```

**Model loading:**
```python
from src.model_loader import get_device, load_orchestrator
model, tokenizer = load_orchestrator(get_device())   # Qwen3-4B, thinking ON
```

**Retrieval call:**
```python
from src.rag_vector import retrieve as vector_retrieve
from src.rag_graph  import retrieve as graph_retrieve

retrieve_fn = vector_retrieve if args.retrieval == "vector" else graph_retrieve
rag_context = retrieve_fn(query=args.prompt, top_k=5)  # list[RetrievedSpan]
```

**Stub while waiting for Members 2 + 3:**
```python
def retrieve(query, top_k=5, **_): return []   # swap out later
```

**Prompt structure:**
```
system:
  You are a contract analysis assistant. Retrieved examples are for reasoning
  only — never cite them as evidence. All evidence must come from the contract.

user:
  [RETRIEVAL CONTEXT — training corpus, not from analyzed contract]
  Example 1 | doc_42, span_15 | H06=ENTAILED
  "The Receiving Party may disclose to its agents and consultants..."
  [END RETRIEVAL CONTEXT]

  CONTRACT: <full text>
  QUESTION: <user prompt>

assistant: <response>
user: <follow-up>   ← loaded from conversation_history.json
```

**History persistence:**
```python
import json
from pathlib import Path

HISTORY_FILE = Path("conversation_history.json")
load_history = lambda: json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
save_history = lambda h: HISTORY_FILE.write_text(json.dumps(h, indent=2))
```

---

### Member 5 — Schemas + Report

**Playbook schema** (`schemas/playbook_schema.json`) — extend `playbook.yaml`:
- Add `"schema_version": "2.0-ms2"`, `"retrieval_mode": "vector" | "graph"`
- Per check: add `"agent_id"` (H01–H17) and `"conversation_turn"`

**RunTrace schema** (`schemas/runtrace_schema.json`) — extend `data/runtrace_ms1_schema.json`:
- Bump `"schema_version"` to `"2.0-ms2"`
- Add `"retrieval_mode"`, `"conversation_history"`, `"retrieval_context"`
- Per `hypothesis_trace`: add `"agent_metadata": {agent_id, rag_query, rag_hits, rag_mode}`

**Report** (`architecture/report.pdf`) — source from `architecture/architecture.yaml`:
1. System overview — broadcast → fan-out → aggregate
2. Orchestrator — tool-calling, JSON function call schema
3. NLI Core — one forward pass, thinking OFF
4. Vector RAG — FAISS, embedding, query modes
5. GraphRAG — nodes/edges, concept vocabulary, query paths
6. Conversation Agent — history, retrieval routing, evidence distinction
7. Playbook changes vs MS1
8. RunTrace schema changes vs MS1
9. Implemented (RAG + conversation) vs MS3 architectural (orchestrator, hypothesis pool, aggregator)

---

## 6. Repo Structure

```
contract-lens/
├── src/
│   ├── constants.py            MS1 — NDA_TO_H, LABEL_MAP, HYPOTHESES, SYSTEM_PROMPT
│   ├── preprocessor.py         MS1 — build_chunks(), build_prompt(), build_answer()
│   ├── types.py                MS2 — RetrievedSpan, HypothesisTask, HypothesisTrace
│   ├── model_loader.py         MS2 — get_device(), load_orchestrator(), load_nli_model()
│   ├── rag_vector.py           MS2 — Member 2
│   ├── rag_graph.py            MS2 — Member 3
│   └── conversation_agent.py   MS2 — Member 4
├── schemas/
│   ├── playbook_schema.json    MS2 — Member 5
│   └── runtrace_schema.json    MS2 — Member 5
├── architecture/
│   ├── architecture.yaml       MS2 — living spec
│   ├── architecture.pdf        MS2 — rendered diagram
│   ├── generate_diagram.py     MS2 — python architecture/generate_diagram.py
│   └── report.pdf              MS2 — Member 5
├── data/
│   ├── train.json              423 NDAs · 32,359 spans (RAG index source)
│   ├── test.json               123 NDAs (evaluation only — never index this)
│   ├── runtrace_ms1_schema.json reference for Member 5
│   └── indexes/                gitignored — built locally
├── 03_build_index.py           MS2 — Members 2 + 3
├── agent.py                    MS2 — Member 4
├── test_model_loader.py        MS2 — smoke-test all models
├── download_models.py          MS2 — pre-download all models
├── quick_infer.py              MS1 extended — single-contract NLI test
├── 01_preprocess.py            MS1
├── 02_finetune.sh              MS1
├── 05_eval_runtrace.py         MS1
└── MILESTONE2_PLAN.md          work split, interfaces, deadlines
```

---

## 7. Hard Constraints

| Constraint | Detail |
|---|---|
| Model family | Qwen3 only — orchestrator Qwen3-4B, NLI core Qwen3-1.7B fine-tuned |
| Thinking | Fine-tuned NLI core: `enable_thinking=False`. All other models: `True` |
| Evidence grounding | RAG = reasoning aid only. Evidence must come from the analyzed contract |
| Retrieval corpus | `data/train.json` only — never index `data/test.json` |
| One retrieval branch per run | `--retrieval vector` OR `--retrieval graph` |
| History | `conversation_history.json` persists between runs |

---

---

# Milestone 1

## Model

- **Base model**: `unsloth/Qwen3-1.7B-bnb-4bit`
- **Adapter method**: QLoRA (rank 4, all attention + MLP layers)
- **Fine-tuned weights**: https://huggingface.co/Youssef-Malek/contractnli-vast-ai-qwen3-1.7b
- **Training**: 1269 steps, 8192 context, 1.28 hrs on RTX 5090

## Evaluation Results

| Metric | Value |
|---|---|
| Label Accuracy | 0.8513 |
| Groundedness | 1.0000 |
| Quote Integrity Pass Rate | 0.5768 |
| Avg Latency (ms) | 14106 |

Evaluated on the full ContractNLI test split (123 contracts, 2091 hypothesis instances).

## Reproducing the Evaluation

### 1. Environment setup
```bash
pip install "unsloth[cu128-torch260]" --upgrade
pip install trl scikit-learn pyyaml
```

### 2. Required files (place in working directory)
```
test.jsonl        # generated by 01_preprocess.py from ContractNLI test.json
test.json         # original ContractNLI test split
playbook.yaml     # included in this submission
```

### 3. Run evaluation
```bash
python 05_eval_runtrace.py
```

This runs two inference passes per contract:
- **Pass 1**: NLI classification — predicts label + evidence span indices for all 17 hypotheses
- **Pass 2**: Verbal confidence + quote elicitation — model states confidence and copies a verbatim excerpt

Outputs:
- `runs/runtrace_doc_NNN.json` — one schema-valid RunTrace per contract
- `evaluation.csv` — aggregate metrics

### 4. Debug a single contract
```bash
python 05b_debug_single.py --idx 0   # prints full model output for both passes
```

## Submission Contents

| File | Description |
|---|---|
| `evaluation.csv` | Final aggregate metrics (required) |
| `runs/` | 123 RunTrace JSON files, one per test contract |
| `RunTrace.json` | All 123 RunTraces combined as a single JSON array |
| `playbook.yaml` | Deterministic rule layer used in evaluation |
| `training_notebook.ipynb` | Annotated training notebook |
| `01_preprocess.py` | Builds SFT dataset from ContractNLI JSON |
| `02_finetune.sh` | Training command wrapping mlx_lm.lora |
| `05_eval_runtrace.py` | Evaluation + RunTrace generation script |
| `05b_debug_single.py` | Single-contract debug tool |
| `src/constants.py` | NDA_TO_H mapping, LABEL_MAP, hypothesis definitions |
| `src/preprocessor.py` | Prompt and answer builders for SFT format |
| `CONTRIBUTIONS.md` | Group member contributions |

## Directory Structure

```
submission/
├── 01_preprocess.py          # Step 1 — build train.jsonl + valid.jsonl from ContractNLI
├── 02_finetune.sh            # Step 2 — QLoRA fine-tuning command
├── 05_eval_runtrace.py       # Step 3 — inference + RunTrace generation + metrics
├── 05b_debug_single.py       # Optional — print full model output for one contract
├── src/
│   ├── constants.py          # NDA_TO_H, LABEL_MAP, LABEL_TO_STATUS, HYPOTHESES
│   └── preprocessor.py       # build_prompt(), build_answer(), build_chunks()
├── training_notebook.ipynb   # Annotated notebook from vast.ai training run
├── playbook.yaml             # Deterministic rule layer (17 hypothesis checks)
├── evaluation.csv            # Final aggregate metrics
├── runs/                     # 123 individual RunTrace JSON files
│   ├── runtrace_doc_000.json
│   ├── runtrace_doc_001.json
│   └── ... (runtrace_doc_122.json)
├── RunTrace.json        # All 123 RunTraces as a single JSON array
├── CONTRIBUTIONS.md          # Group member contributions
└── README.md                 # This file
```

## Architecture Notes

**Quote integrity (0.577)**: The pipeline performs a two-pass inference. In pass 2, the model is asked to output a verbatim quote from the contract. Quote integrity checks whether that quote is a substring of the canonical contract text. The 1.7B model, trained with thinking disabled, tends to output hypothesis text rather than contract text — these correctly fail the integrity check.

**Groundedness (1.000)**: The model learned from SFT training data to always cite evidence spans for ENTAILED/CONTRADICTED predictions. All cited span indices are valid (within range of the document's span array).

**Confidence**: Set to a heuristic (0.9 for ENTAILED/CONTRADICTED, 0.7 for NOT_MENTIONED). The 1.7B model was trained with `enable_thinking=False` and cannot self-calibrate confidence reliably.

## Results Analysis and Limitations

### What we achieved
The pipeline is fully end-to-end: preprocessing → QLoRA fine-tuning → two-pass inference → playbook application → schema-valid RunTrace generation. All 123 test contracts were evaluated and every RunTrace passes schema validation. Label accuracy of **85.1%** on 2,091 hypothesis instances is a strong baseline for a 1.7B parameter model trained on only 381 examples.

### Why results are not higher

**1. Model size and training data volume**
Qwen3-1.7B is a small model trained on 381 NDA examples. Legal NLI is a demanding task — contracts are long (averaging ~4,000 tokens of spans), hypotheses are subtle, and the distinction between CONTRADICTED and NOT_MENTIONED requires careful cross-referencing of multiple clauses. A 1.7B model without reasoning capability is working at the edge of what is achievable at this scale.

**2. Training without thinking traces (`enable_thinking=False`)**
The model was trained using standard SFT with `enable_thinking=False` — the assistant turn is a direct JSON array with no intermediate reasoning. This means the model pattern-matches from contract spans to labels in a single forward pass with no chain-of-thought. For straightforward hypotheses this works well, but for ambiguous cases that require multi-step reasoning across several contract clauses, the model lacks the mechanism to work through the evidence systematically before committing to a label.

**3. What thinking traces would have done**
We identified this limitation and began generating reasoning traces using knowledge distillation: a larger reasoning model (Groq `qwen3-32b`) was used to generate step-by-step `<think>` blocks for each training document, anchored to the ContractNLI gold labels. The approach produces traces like:

> *"H01 says all Confidential Information must be expressly identified. Span 14 defines Confidential Information as anything designated as confidential at the time of disclosure — this is an explicit identification requirement. Therefore H01 is ENTAILED by span 14."*

Training the 1.7B model on these traces with `enable_thinking=True` would teach it to reason through the evidence before producing a label, directly addressing the failure mode on ambiguous contracts. We expect this would push label accuracy above 90% and significantly improve quote integrity, since a reasoning model is more likely to cite actual contract text rather than paraphrase the hypothesis.

**4. Quote integrity (0.577)**
The 0.577 quote integrity rate reflects the model's tendency in Pass 2 to reproduce hypothesis text rather than copy a verbatim excerpt from the contract. This is a direct consequence of training without thinking — the model has not learned to ground its outputs in specific contract language. A thinking-enabled model that cites span text during its reasoning phase would produce much more faithful verbatim quotes.

### Next iteration
The trace generation pipeline (`06_generate_traces.py`) is already implemented and running. The planned next step is:
1. Complete Groq-generated thinking traces for all 381 training documents
2. Retrain Qwen3-1.7B with `enable_thinking=True` and LoRA rank 8
3. Re-evaluate on the same test split for a direct comparison
