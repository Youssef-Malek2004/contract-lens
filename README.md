# ContractLens

Multi-agent NDA review system built on ContractNLI. Classifies all 17 ContractNLI hypotheses (H01–H17) against an input NDA, produces schema-valid RunTrace output, and provides a RAG-augmented conversation agent for free-form contract Q&A.

**Dataset:** ContractNLI — 423 train NDAs (32,359 spans), 123 test NDAs  
**Model family:** Qwen3 only — Orchestrator: Qwen3-4B, NLI Core: fine-tuned Qwen3-1.7B + LoRA  
**Fine-tuned adapter:** [Youssef-Malek/contractnli-vast-ai-qwen3-1.7b](https://huggingface.co/Youssef-Malek/contractnli-vast-ai-qwen3-1.7b)

---

- [Milestone 2 (current)](#milestone-2)
- [Milestone 1 results](#milestone-1)

---

## Milestone 2

### Environment Setup

> **Use the `genai-ms2` conda env.** Python 3.9 cannot install `transformers` from source — Qwen3 support requires the latest HEAD.

```bash
# Create once
conda create -n genai-ms2 python=3.11 -y
conda activate genai-ms2

# transformers must come from source — install it first
pip install torch torchvision torchaudio
pip install "git+https://github.com/huggingface/transformers.git"
pip install accelerate peft sentence-transformers \
            faiss-cpu networkx scikit-learn numpy huggingface_hub \
            safetensors tokenizers tqdm pyyaml ipykernel

# Optional: register the Jupyter kernel
python -m ipykernel install --user --name genai-ms2 --display-name "genai-ms2"
```

All commands below assume `conda activate genai-ms2` and working directory = repo root (`contract-lens/`).

---

### Download Models

Run once from the terminal — **not** via `conda run` (subprocess stdout buffering hides progress bars):

```bash
python scripts/download_models.py
```

| Model                                          | Size    | Purpose                           |
| ---------------------------------------------- | ------- | --------------------------------- |
| `Qwen/Qwen3-4B`                                | ~8 GB   | Orchestrator + Conversation Agent |
| `Qwen/Qwen3-1.7B`                              | ~3.5 GB | NLI Core base + Hypothesis Agents |
| `Youssef-Malek/contractnli-vast-ai-qwen3-1.7b` | ~100 MB | Fine-tuned LoRA adapter           |
| `sentence-transformers/all-MiniLM-L6-v2`       | ~90 MB  | Vector RAG embeddings             |

All downloads go to `~/.cache/huggingface/` — every script finds them automatically.

---

### Verify Setup

```bash
python tests/test_model_loader.py
```

Runs three live-streaming tests: Orchestrator (Qwen3-4B, thinking ON), base model (adapter OFF, thinking ON), NLI Core (adapter ON, thinking OFF). All three must print `PASS`.

---

### Build RAG Indexes

Required before using the conversation agent. Outputs go to `data/indexes/` (gitignored — rebuild locally from `data/train.json`).

```bash
python pipeline/03_build_index.py --mode vector    # FAISS index, ~5 min
python pipeline/03_build_index.py --mode graph     # networkx graph, ~3 min
python pipeline/03_build_index.py --mode all       # both at once
```

---

### Run the Conversation Agent

Ask free-form questions about any NDA in the test set:

```bash
# Vector RAG backend (default)
python agent.py --contract data/test.json --idx 0 \
                --retrieval vector \
                --prompt "Does this NDA allow sharing with consultants?"

# Graph RAG backend
python agent.py --contract data/test.json --idx 0 \
                --retrieval graph \
                --prompt "What are the termination obligations?"

# Remote mode — use the vllm-mlx server instead of loading Qwen3-4B locally (Apple Silicon)
python agent.py --contract data/test.json --idx 0 \
                --retrieval vector \
                --prompt "What restrictions apply to sublicensing?" \
                --remote
```

Conversation history persists in `conversation_history.json` between runs. History auto-resets when `--idx` changes.

**Arguments:**

| Flag          | Default          | Description                                            |
| ------------- | ---------------- | ------------------------------------------------------ |
| `--contract`  | `data/test.json` | Path to ContractNLI JSON file                          |
| `--idx`       | `0`              | Zero-based document index within the file              |
| `--retrieval` | `vector`         | RAG backend: `vector` or `graph`                       |
| `--prompt`    | _(required)_     | Natural language question                              |
| `--remote`    | off              | Route to vllm-mlx server at `http://localhost:8001/v1` |

---

### Run Single-Contract NLI Inference

Two-pass inference on one contract (no unsloth required — plain transformers):

```bash
python scripts/quick_infer.py                          # first val doc
python scripts/quick_infer.py --idx 3                  # 4th val doc
python scripts/quick_infer.py --data data/train.json --idx 0
```

Pass 1: NLI Core (adapter ON, thinking OFF) — predicts label + evidence spans for all 17 hypotheses.  
Pass 2: Base model (adapter OFF, thinking ON) — confidence score + verbatim quote.  
Prints a results table against gold labels.

---

### vllm-mlx Remote Mode (Apple Silicon only)

The `--remote` flag routes Qwen3-4B calls to a local vllm-mlx server. The NLI/PEFT path is always local (LoRA adapters cannot be toggled in a pre-quantized vllm-mlx model).

```bash
# Start the server (from ServeLM/ — serves mlx-community/Qwen3-4b-4bit, port 8001)
../serving-local-models/serve-qwen3.sh

# Test all three endpoints (requires servers on ports 8001, 8002, 8003)
python tests/test_vllm_endpoints.py
```

To merge the NLI adapter into a standalone model for vllm-mlx serving:

```bash
python scripts/merge_adapter.py               # merge only
python scripts/merge_adapter.py --convert     # merge + MLX 4-bit conversion
```

Outputs go to `merged-nli-1.7b/` and `mlx-nli-1.7b-4bit/` (gitignored).

---

### Architecture Diagram

```bash
conda install -c conda-forge graphviz python-graphviz -y
python architecture/generate_diagram.py       # → architecture/architecture.pdf
```

LaTeX report:

```bash
cd architecture && pdflatex report.tex        # → architecture/report.pdf
```

---

## Repo Structure

```
contract-lens/
│
├── agent.py                    ← CLI entry point for the conversation agent
├── playbook.yaml               ← deterministic rule layer (17 hypothesis checks)
├── requirements.txt
│
├── src/                        ← core library (import from here)
│   ├── constants.py            NDA_TO_H, LABEL_MAP, HYPOTHESES, SYSTEM_PROMPT
│   ├── preprocessor.py         build_chunks(), build_prompt(), build_answer()
│   ├── types.py                RetrievedSpan, HypothesisTask, HypothesisTrace
│   ├── model_loader.py         get_device(), load_orchestrator(), load_nli_model()
│   ├── rag_vector.py           FAISS vector retrieval
│   ├── rag_graph.py            networkx GraphRAG retrieval
│   ├── conversation_agent.py   ConversationAgent class
│   └── loaders/                LocalLoader, VllmLoader, RemoteOrchestrator
│
├── pipeline/                   ← numbered ML pipeline steps (run from repo root)
│   ├── 01_preprocess.py        build SFT dataset from train.json → .jsonl
│   ├── 02_finetune.sh          QLoRA fine-tuning command (wraps mlx_lm.lora)
│   ├── 03_build_index.py       build vector + graph RAG indexes
│   ├── 05_eval_runtrace.py     batch inference → runs/ + evaluation.csv (MS1)
│   └── 05b_debug_single.py     single-contract debug tool (MS1)
│
├── scripts/                    ← operational utilities
│   ├── download_models.py      pre-download all models to HF cache
│   ├── merge_adapter.py        merge LoRA adapter + MLX conversion
│   ├── quick_infer.py          single-contract two-pass NLI inference
│   ├── setup_models.sh         server-side model setup
│   └── stop_servers.sh         stop all vllm-mlx servers
│
├── tests/                      ← test suite (run from repo root)
│   ├── test_model_loader.py    smoke-test all three models with streaming
│   ├── test_indexes.py         RAG index correctness tests
│   └── test_vllm_endpoints.py  unit tests for the three vllm-mlx endpoints
│
├── data/                       ← source data
│   ├── train.json              423 NDAs · 32,359 spans (RAG index source only)
│   ├── test.json               123 NDAs (evaluation only — never index this)
│   ├── runtrace_ms1_schema.json MS1 RunTrace schema reference
│   └── indexes/                gitignored — rebuild with pipeline/03_build_index.py
│
├── runs/                       ← per-contract RunTrace JSONs (123 files, MS1 output)
├── RunTrace.json               all 123 RunTraces as a single JSON array (MS1)
├── evaluation.csv              aggregate evaluation metrics (MS1)
│
├── schemas/
│   ├── playbook_schema.json
│   └── runtrace_schema.json
│
├── architecture/
│   ├── architecture.yaml       living system spec
│   ├── architecture.pdf        rendered diagram
│   ├── generate_diagram.py
│   ├── report.tex              LaTeX source
│   └── report.pdf
│
├── notebooks/
│   └── training_notebook.ipynb annotated QLoRA training notebook
│
└── docs/
    ├── CONTRIBUTIONS.md        group member contributions
    └── MILESTONE2_PLAN.md      work split + interfaces + deadlines
```

---

## Hard Constraints

| Constraint           | Detail                                                                       |
| -------------------- | ---------------------------------------------------------------------------- |
| Model family         | Qwen3 only — orchestrator Qwen3-4B, NLI core fine-tuned Qwen3-1.7B           |
| Thinking policy      | Fine-tuned NLI Core: `enable_thinking=False`. All other calls: `True`        |
| Evidence grounding   | RAG = reasoning aid only — evidence must come from the analyzed contract     |
| Retrieval corpus     | `data/train.json` only — **never index `data/test.json`**                    |
| One retrieval branch | `--retrieval vector` OR `--retrieval graph`, not both per run                |
| History              | `conversation_history.json` persists between runs; resets on contract change |

---

---

## Milestone 1

### Model

|                    |                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| Base model         | `unsloth/Qwen3-1.7B-bnb-4bit`                                                                                       |
| Adapter method     | QLoRA (rank 4, all attention + MLP layers)                                                                          |
| Fine-tuned weights | [Youssef-Malek/contractnli-vast-ai-qwen3-1.7b](https://huggingface.co/Youssef-Malek/contractnli-vast-ai-qwen3-1.7b) |
| Training           | 1269 steps · 8192 context · 1.28 hrs on RTX 5090                                                                    |

### Evaluation Results

| Metric                    | Value  |
| ------------------------- | ------ |
| Label Accuracy            | 0.8513 |
| Groundedness              | 1.0000 |
| Quote Integrity Pass Rate | 0.5768 |
| Avg Latency (ms)          | 14106  |

Evaluated on the full ContractNLI test split (123 contracts, 2091 hypothesis instances).

### Reproducing the Evaluation

**1. Environment**

```bash
pip install "unsloth[cu128-torch260]" --upgrade
pip install trl scikit-learn pyyaml
```

**2. Required files** (place in working directory)

```
test.jsonl        # generated by pipeline/01_preprocess.py from data/test.json
```

**3. Run**

```bash
python pipeline/05_eval_runtrace.py
```

Two inference passes per contract:

- **Pass 1** — NLI classification (label + evidence span indices for all 17 hypotheses)
- **Pass 2** — Confidence + verbatim quote elicitation

Outputs: `runs/runtrace_doc_NNN.json` per contract + `evaluation.csv`

**4. Debug a single contract**

```bash
python pipeline/05b_debug_single.py --idx 0    # full model output, both passes
```

### Results Analysis

**Label accuracy 85.1%** on 2,091 hypothesis instances is a strong baseline for a 1.7B model trained on 381 examples.

**Groundedness 1.000**: The model always cites valid evidence spans for ENTAILED/CONTRADICTED predictions — all cited span indices are within range.

**Quote integrity 0.577**: In Pass 2 the model tends to reproduce hypothesis text rather than copy a verbatim contract excerpt. This is a direct consequence of `enable_thinking=False` training — the model lacks a reasoning phase to ground its outputs in specific contract language.

**Next iteration**: A knowledge-distillation pipeline (`06_generate_traces.py`) generates step-by-step `<think>` blocks from `qwen3-32b` for all training documents. Retraining with `enable_thinking=True` and LoRA rank 8 is expected to push label accuracy above 90% and significantly improve quote integrity.
