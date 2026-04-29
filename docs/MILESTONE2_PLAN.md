# Milestone 2 — Work Split (5 Members)

**Due:** 30 April 2026
**Branch:** `ms-2`
**Architecture spec:** `architecture/architecture.yaml` (living doc — read before coding)

---

## Deliverables Checklist

| #   | Deliverable                                            | File(s)                                                                          |
| --- | ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| a   | Agentic architecture diagram                           | `architecture/architecture.pdf` + `architecture/architecture.yaml`               |
| b   | Architecture report                                    | `architecture/report.pdf`                                                        |
| c   | Source code — Vector RAG, GraphRAG, Conversation Agent | `src/rag_vector.py`, `src/rag_graph.py`, `src/conversation_agent.py`, `agent.py` |
| d   | Updated playbook                                       | `schemas/playbook_schema.json`                                                   |
| e   | Updated RunTrace schema                                | `schemas/runtrace_schema.json`                                                   |
| f   | All work on `ms-2` branch                              | GitHub                                                                           |

---

## Shared Foundations (read before starting)

### Data

- `data/train.json` — 423 NDAs, **32,359 spans total** (avg 76.5 per doc). RAG index source.
- `data/test.json` — 123 NDAs. Evaluation only. Never index this.
- `data/runtrace_ms1_schema.json` — MS1 schema for reference.

### Models

| Object                            | Model                                          | Thinking | Quantization                                        |
| --------------------------------- | ---------------------------------------------- | -------- | --------------------------------------------------- |
| Orchestrator + Conversation Agent | `Qwen/Qwen3-4B`                                | ON       | BitsAndBytesConfig NF4 on CUDA · float16 on MPS/CPU |
| NLI Core (fine-tuned, adapter ON) | `Youssef-Malek/contractnli-vast-ai-qwen3-1.7b` | OFF      | same                                                |
| Hypothesis agents (adapter OFF)   | same object as NLI Core                        | ON       | same                                                |

### Shared Types (implement in `src/types.py` first)

```python
class RetrievedSpan(TypedDict):
    text: str                            # verbatim span text
    doc_id: str                          # training contract id
    span_idx: int                        # original span index
    score: float                         # cosine sim (vector) or concept score (graph)
    hypothesis_annotations: dict[str, str]  # {"H06": "ENTAILED", ...} — gold labels

class HypothesisTask(TypedDict):
    hypothesis_id: str
    hypothesis_text: str
    label: str                           # ENTAILED | CONTRADICTED | NOT_MENTIONED
    evidence_spans: list[int]
    contract_chunks: list[dict]
    rag_context: list[RetrievedSpan]

class HypothesisTrace(TypedDict):
    hypothesis_id: str
    label: str
    confidence: float
    evidence_spans: list[int]
    verbatim_quote: str | None
    groundedness_check: bool
    quote_integrity_check: bool
    playbook_result: dict                # {severity, action, rationale}
    agent_metadata: dict                 # {agent_id, rag_query, rag_hits, rag_mode}
```

### Device / Quantization (`src/model_loader.py` — already implemented ✓)

```python
from src.model_loader import get_device, load_orchestrator, load_nli_model

device = get_device()                    # "mps" | "cuda" | "cpu" — auto-detected
model, tokenizer = load_orchestrator()   # Qwen3-4B, thinking=ON
model, tokenizer = load_nli_model()      # Qwen3-1.7B + LoRA, thinking=OFF

# CUDA  → BitsAndBytesConfig NF4 4-bit (native int4 kernels)
# MPS   → dtype=torch.float16 (no quantization — MPS has no native int4 GEMM)
# CPU   → dtype=torch.float16 (fallback only)
```

---

## Member 1 — Architecture Design + Diagram

**Deliverables:** (a)
**Status:** ✓ Complete

**Files delivered:**

```
architecture/
├── architecture.yaml   ✓ living spec — update as decisions change
├── architecture.pdf    ✓ rendered diagram (regenerate: python architecture/generate_diagram.py)
└── generate_diagram.py ✓
src/
├── types.py            ✓ RetrievedSpan, HypothesisTask, HypothesisTrace
└── model_loader.py     ✓ get_device(), load_orchestrator(), load_nli_model()
```

**Scope:**

- `architecture.yaml` is the primary deliverable — already contains the full multi-agent design, RAG blueprints, tool schemas, type definitions. Keep it updated as decisions are made.
- Render `architecture.pdf` from the YAML (draw.io, Mermaid, or any diagram tool) showing:
  - Orchestrator (Qwen3-4B) → tool calls → NLI Core, Dispatcher, Hypothesis Pool, Aggregator
  - Conversation Agent alongside the NLI pipeline (not inside it)
  - Vector RAG branch and GraphRAG branch, both feeding Dispatcher + Conversation Agent
  - Training corpus vs analyzed contract clearly visually distinct
  - Intermediate inputs/outputs at each step
- Implement `src/types.py` — the three TypedDicts that all other members code against
- Implement `src/model_loader.py` — the single place for device/quantization logic so no member has to handle it themselves

**Dependency note:** `src/types.py` and `src/model_loader.py` must exist before Members 2, 3, 4 can finalize. Push stubs on day 1.

---

## Member 2 — Vector RAG Pipeline

**Deliverables:** part of (c)

**Files to create:**

```
src/rag_vector.py
03_build_index.py          ← shared with Member 3, agree on structure first
data/indexes/vector/       ← gitignored, built locally
```

**Exact retrieve() interface to implement:**

```python
def retrieve(
    query: str,
    top_k: int = 5,
    hypothesis_id: str | None = None,    # e.g. "H06" — filters/reranks results
    label_filter: str | None = None,     # "ENTAILED" | "CONTRADICTED" | "NOT_MENTIONED"
) -> list[RetrievedSpan]:               # from src/types.py
```

**Indexing — what to build:**

- Load `data/train.json` → `build_chunks()` (reuse `src/preprocessor.py`) for each doc
- For each span, also read gold annotations to populate `hypothesis_annotations`
- Embed each span with `sentence-transformers/all-MiniLM-L6-v2` (384 dims, CPU is fine)
- L2-normalise all vectors (so inner product = cosine similarity)
- FAISS `IndexFlatIP` — exact search, no training step, ~47MB for 32k vectors

**Index files to produce:**

```
data/indexes/vector/spans.jsonl    # one line per span: text, doc_id, span_idx,
                                   # char_start, char_end, hypothesis_annotations
                                   # row i matches FAISS vector i
data/indexes/vector/faiss.index    # FAISS binary
data/indexes/vector/metadata.json  # {embedding_model, index_size, build_timestamp}
```

**Query logic:**

1. Embed query → L2-normalise → `faiss_index.search(vec, top_k * 3)`
2. Load metadata rows from `spans.jsonl`
3. If `hypothesis_id` given: re-rank to prefer spans where `hypothesis_annotations[hypothesis_id] == label_filter`
4. Return top_k as `list[RetrievedSpan]`

**Build command:** `python 03_build_index.py --mode vector`

**Depends on:** `src/types.py` from Member 1.
**Independent of:** Member 3.

---

## Member 3 — GraphRAG Pipeline

**Deliverables:** part of (c)

**Files to create:**

```
src/rag_graph.py
03_build_index.py          ← shared with Member 2, coordinate first
data/indexes/graph/        ← gitignored, built locally
```

**Exact retrieve() interface — identical signature to Member 2:**

```python
def retrieve(
    query: str,
    top_k: int = 5,
    hypothesis_id: str | None = None,
    label_filter: str | None = None,
) -> list[RetrievedSpan]:               # from src/types.py
```

**Graph structure — 3 node types, 3 edge types:**

```
SpanNode ──CONTAINS──► ConceptNode ◄──INVOLVES── HypothesisNode
SpanNode ──CITED_FOR──────────────────────────► HypothesisNode
```

- **SpanNode** — one per training span (~32k). Attrs: text, doc_id, span_idx, char offsets.
- **ConceptNode** — ~19 hardcoded legal terms with synonym lists (full list in `architecture.yaml`). Examples:
  - `confidential_information`: ["confidential information", "proprietary information", "trade secret", ...]
  - `third_party`: ["third party", "consultant", "advisor", "agent", "contractor", ...]
  - `termination`: ["termination", "expiration", "expiry", "end of agreement", ...]
- **HypothesisNode** — 17 nodes, from `HYPOTHESES` constant in `src/constants.py`.

**Edge creation:**

- `CONTAINS` (SpanNode → ConceptNode): scan each span's lowercased text for synonym matches. Simple `substring in text` check. No NLP/NER.
- `CITED_FOR` (SpanNode → HypothesisNode, attr: label): read gold annotations from `train.json`. If `span_idx` appears in `annotations[nda_key]["spans"]`, add edge with `label=LABEL_MAP[choice]`.
- `INVOLVES` (HypothesisNode → ConceptNode): scan each hypothesis text for synonym matches.

**Index files to produce:**

```
data/indexes/graph/graph.pkl              # serialised networkx.DiGraph
data/indexes/graph/hypothesis_index.json # {hypothesis_id: {label: [span_node_ids]}}
                                          # pre-computed from CITED_FOR edges — O(1) lookup
data/indexes/graph/concept_index.json    # {canonical_name: [span_node_ids]}
                                          # pre-computed from CONTAINS edges — O(1) lookup
data/indexes/graph/metadata.json         # node/edge counts, build_timestamp
```

**Two query paths:**

_Hypothesis-targeted_ (hypothesis_id provided):

```
hypothesis_index[hypothesis_id][label_filter] → span_ids → load attrs → return top_k
Score by: citation frequency (how many training docs cite this span for this hypothesis+label)
```

_Free-text_ (no hypothesis_id):

```
1. Lowercase query → match against synonym lists → matched ConceptNodes
2. concept_index[canonical_name] → span_ids for each match
3. Union all candidates
4. Score: base = matched_concept_count / total_matched; gold_bonus = +0.5 if span has any CITED_FOR edge
5. Return top_k
```

**Build command:** `python 03_build_index.py --mode graph`

**Coordinate with Member 2** on `03_build_index.py` entry point before either writes it.

**Depends on:** `src/types.py` from Member 1.
**Independent of:** Member 2.

---

## Member 4 — Conversation Agent + CLI

**Deliverables:** part of (c)

**Files to create:**

```
src/conversation_agent.py
agent.py                         ← CLI entry point
```

**CLI interface:**

```bash
python agent.py --contract data/test.json --idx 0 \
                --retrieval vector \
                --prompt "Does this NDA allow third-party sharing?"

python agent.py --contract data/test.json --idx 0 \
                --retrieval graph \
                --prompt "What are the termination obligations?"
```

**Model:** `Qwen/Qwen3-4B` (orchestrator_model) with `enable_thinking=True`.
Use `src/model_loader.py` from Member 1 for loading. Strip `<think>` blocks before printing response.

**Message structure:**

```
system: [conversation agent system prompt — explains external memory vs contract evidence]

user turn 1:
  [RETRIEVAL CONTEXT — external memory from training corpus]
  [For reasoning reference only. Never cite as evidence.]
  Example 1 | Source: doc_42, span_15 | H06=ENTAILED
  "The Receiving Party may disclose to its agents and consultants..."
  [END RETRIEVAL CONTEXT]

  CONTRACT:
  [full analyzed contract text]

  QUESTION: [user prompt]

assistant turn 1: [response — strips <think> block before display]

user turn 2: [follow-up question, loaded from history file]
assistant turn 2: [response]
...
```

**History persistence:**

- Load history from `conversation_history.json` at start (empty list if file absent)
- Append `{role, content, timestamp}` after each turn
- Write back to `conversation_history.json`
- History is per-contract-session — clear or rotate when `--idx` changes

**Critical constraint:** Retrieved spans from RAG are labeled `[RETRIEVAL CONTEXT]` in the prompt. They inform reasoning only. The model must never present them as evidence for the analyzed contract. The system prompt must make this explicit.

**Retrieve call:**

```python
from src.rag_vector import retrieve as vector_retrieve
from src.rag_graph  import retrieve as graph_retrieve

retrieve_fn = vector_retrieve if args.retrieval == "vector" else graph_retrieve
rag_context: list[RetrievedSpan] = retrieve_fn(query=args.prompt, top_k=5)
```

**Depends on:** Members 2 and 3 (`retrieve()` interface). Can stub with dummy data while waiting.

---

## Member 5 — Updated Schemas + Architecture Report

**Deliverables:** (b), (d), (e)

**Files to create:**

```
schemas/
├── playbook_schema.json
└── runtrace_schema.json
architecture/
└── report.pdf
```

### Updated Playbook (`schemas/playbook_schema.json`)

Start from `playbook.yaml`. Convert to JSON and add:

- `schema_version: "2.0-ms2"`
- `retrieval_mode: "vector" | "graph"` — which branch was used
- Per-check additions:
  - `agent_id` — which hypothesis agent (H01–H17) produced this result
  - `conversation_turn` — which turn triggered this check (null if not from conversation)
- Keep all 17 hypothesis checks, criticality tiers, override rules unchanged

### Updated RunTrace Schema (`schemas/runtrace_schema.json`)

Start from `data/runtrace_ms1_schema.json`. Add:

- `schema_version: "2.0-ms2"`
- `retrieval_mode: "vector" | "graph"`
- `conversation_history: [{role, content, timestamp}]`
- `retrieval_context: {external_memory: [...], contract_evidence: [...]}`
- Per hypothesis_trace, add `agent_metadata: {agent_id, rag_query, rag_hits, rag_mode}`

### Architecture Report (`architecture/report.pdf`)

Cover these sections (source all details from `architecture/architecture.yaml`):

1. System overview and design rationale — why multi-agent, why broadcast→fan-out
2. Orchestrator (Qwen3-4B) — tool-calling design, JSON function call schema
3. NLI Core — why it runs once, why thinking is OFF, what it produces
4. Vector RAG — embedding, FAISS IndexFlatIP, both query modes
5. GraphRAG — 3-node/3-edge structure, ConceptNode vocabulary, two query paths, why no NLP dependency
6. Conversation Agent — history handling, external-memory vs contract-evidence distinction
7. Updated playbook — what changed from MS1
8. Updated RunTrace schema — what changed from MS1
9. What is **implemented** (RAG + conversation agent) vs **architectural for MS3** (orchestrator, dispatcher, hypothesis pool, aggregator)

**Depends on:** Member 1 (architecture must be stable). Can write sections 7, 8, 9 now.

---

## Dependency Graph

```
April 25–26  ✓ DONE
└── Member 1: architecture.yaml ✓, src/types.py ✓, src/model_loader.py ✓, architecture.pdf ✓
        │
        ├── April 26–28 (parallel) ← YOU ARE HERE
        │   ├── Member 2: Vector RAG (src/rag_vector.py, 03_build_index.py --mode vector)
        │   ├── Member 3: GraphRAG  (src/rag_graph.py,  03_build_index.py --mode graph)
        │   └── Member 5: playbook_schema.json + runtrace_schema.json
        │
        ├── April 27–29 (unblocked once retrieve() interface exists)
        │   └── Member 4: Conversation Agent + CLI
        │
        └── April 29–30
            └── Member 5: Architecture report (finalises last)
```

**Critical path:** Members 2 + 3 → Member 4. Conversation agent needs `retrieve()` stable.
**Deadline:** 30 April 2026.

---

## Shared Files — Coordination Required

| File                  | Owners                          | Notes                                                        |
| --------------------- | ------------------------------- | ------------------------------------------------------------ |
| `03_build_index.py`   | Members 2 + 3                   | Agree on `--mode vector\|graph` arg structure before writing |
| `src/types.py`        | Member 1 (writes), all (import) | Define first — everyone codes against these types            |
| `src/model_loader.py` | Member 1 (writes), all (import) | Single source of truth for device/quantization               |
| `src/__init__.py`     | Members 2, 3, 4                 | Add new module exports as created                            |
| `requirements.txt`    | All                             | Add deps as needed; no version pinning unless conflict       |

---

## Final File Structure After MS2

```
contract-lens/
├── src/
│   ├── constants.py           (MS1 — unchanged)
│   ├── preprocessor.py        (MS1 — unchanged)
│   ├── types.py               ← Member 1  (RetrievedSpan, HypothesisTask, HypothesisTrace)
│   ├── model_loader.py        ← Member 1  (device detection + quantization)
│   ├── rag_vector.py          ← Member 2
│   ├── rag_graph.py           ← Member 3
│   └── conversation_agent.py  ← Member 4
├── schemas/
│   ├── playbook_schema.json   ← Member 5
│   └── runtrace_schema.json   ← Member 5
├── architecture/
│   ├── architecture.yaml      ← Member 1 (living spec)
│   ├── architecture.pdf       ← Member 1 (submission diagram)
│   └── report.pdf             ← Member 5
├── data/
│   ├── train.json             ✓ (32,359 spans across 423 docs)
│   ├── test.json              ✓ (123 evaluation NDAs)
│   ├── runtrace_ms1_schema.json ✓
│   └── indexes/               (gitignored — built locally by 03_build_index.py)
│       ├── vector/
│       └── graph/
├── 01_preprocess.py           (MS1 — unchanged)
├── 02_finetune.sh             (MS1 — unchanged)
├── 03_build_index.py          ← Members 2 + 3
├── 05_eval_runtrace.py        (MS1 — unchanged)
├── agent.py                   ← Member 4
├── quick_infer.py             (MS1 extended — unchanged for MS2)
└── requirements.txt           (updated by all)
```

---

## Hard Constraints (from milestone spec)

| Constraint                   | Detail                                                                                 |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| Same model family            | Qwen3 only — orchestrator is Qwen3-4B, NLI core is Qwen3-1.7B fine-tuned               |
| Thinking policy              | Fine-tuned NLI core: `enable_thinking=False`. All others: `enable_thinking=True`       |
| Evidence grounding           | RAG context = reasoning aid only. Hypothesis evidence must come from analyzed contract |
| Retrieval corpus             | `data/train.json` only — never index or query `data/test.json`                         |
| Test contracts               | Same 123 NDAs from MS1 evaluation split                                                |
| One retrieval branch per run | `--retrieval vector` OR `--retrieval graph`, not both                                  |
| History must persist         | `conversation_history.json` survives between runs for follow-up turns                  |
| Quantization                 | BitsAndBytesConfig NF4 4-bit on CUDA · float16 on MPS/CPU                              |
