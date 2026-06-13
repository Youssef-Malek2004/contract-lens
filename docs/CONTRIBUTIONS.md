# Contributions - ContractNLI

## Milestone 3 Contributions

| Member | MS3 Role | Contribution |
|---|---|---|
| Youssef Malek | M1 - Orchestrator, tools, audit infrastructure | Implemented the OpenRouter orchestrator, function-calling tools, approval gate, RunTrace v3 writers/schema, and cross-agent tool-call recorder. |
| Ahmed Hatim Backar | M2 - Analysis pipeline, LLM side | Implemented dispatcher fan-out, 17 hypothesis workers, OpenRouter 9B calls, worker tool use, and the `run_full_analysis` orchestration entry point. |
| David George Atef | M3 - Analysis pipeline, validation side | Implemented playbook loading, hypothesis validation, playbook result attachment, contract metrics, and per-contract RunTrace writing. |
| Mostafa Ashraf Ezzat | M4 - CLI, TUI, session state | Implemented the MS3 REPL, slash commands, session state, approval prompt rendering, conversation RunTrace snapshots, and interactive/one-shot CLI modes. |
| Daniel Ashraf Ekdawi | M5 - Evaluation and packaging | Implemented the headless 123-contract MS3 evaluation runner, MS1/MS3 combined `evaluation.csv`, per-hypothesis detail CSV, and `runs_ms3.zip` packaging. |
| All | Integration and QA | Integrated the agentic pipeline end to end, verified RunTrace auditability, and prepared the final submission artifacts. |

---

# Contributions - ContractNLI Milestone 1

## Group Members

| Name | Student ID | Contribution |
|---|---|---|
| Youssef Malek | — | Data preprocessing, SFT dataset construction, model training pipeline, evaluation script, RunTrace generation |
| Ahmed Hatim Backar | — | Playbook integration, schema validation, evidence span mapping |
| David George | — | Hyperparameter tuning, training configuration, HuggingFace checkpointing |
| Daniel Ashraf | — | Inference pipeline, output formatting, debugging |
| Mostafa Ashraf | — | Dataset analysis, documentation, results reporting |

## Detailed Contributions

### Youssef Malek
- Designed and implemented the full SFT data preprocessing pipeline (`01_preprocess.py`) — converting ContractNLI JSON annotations into Qwen3 chat-template formatted training examples
- Built the 90/10 train/validation split (381 train / 42 val, seed=42)
- Set up and ran QLoRA fine-tuning on vast.ai (RTX 5090) using Unsloth + TRL SFTTrainer, managing VRAM constraints and fault-tolerant checkpointing to HuggingFace Hub
- Implemented the two-pass evaluation script (`05_eval_runtrace.py`) — Pass 1 for NLI classification, Pass 2 for verbal confidence and verbatim quote elicitation
- Built the RunTrace generation pipeline conforming to `runtrace_ms1_schema.json`, producing 123 schema-valid RunTrace files across the full test split
- Debugged and resolved model output parse failures including swapped schema formats and JSON array vs dict mismatches

### Ahmed Hatim Backar
- Integrated the deterministic playbook layer (`playbook_mapper.py`) — loading `playbook.yaml`, applying per-hypothesis overrides and global defaults, and generating rationale/severity/action fields
- Implemented schema validation against `runtrace_ms1_schema.json` using `jsonschema`
- Mapped model-predicted span indices to character-offset evidence spans for RunTrace evidence fields

### David George
- Tuned training hyperparameters (LoRA rank, learning rate, sequence length, batch size) based on VRAM constraints and ContractNLI document length distribution
- Managed HuggingFace Hub checkpointing strategy (`hub_strategy=all_checkpoints`, SAVE_STEPS=120) to ensure fault tolerance across the training run
- Verified training convergence and documented final configuration (1269 steps, 8192 context, 1.28 hrs)

### Daniel Ashraf
- Implemented the inference pipeline including model loading, adapter merging, and greedy decoding (`temp=0.0`)
- Handled `<think>` tag stripping and JSON extraction from raw model output
- Debugged second-pass seeded thinking behaviour and confidence elicitation fallback logic

### Mostafa Ashraf
- Analysed ContractNLI dataset distribution (label frequencies, document lengths, span counts)
- Compiled and verified final evaluation metrics (label accuracy, groundedness, quote integrity, avg latency)
- Prepared submission documentation including README, architecture notes, and this contributions file

## Infrastructure

- **Training**: RTX 5090 (31 GB VRAM) on vast.ai
- **Base model**: `unsloth/Qwen3-1.7B-bnb-4bit`
- **Fine-tuned model**: `Youssef-Malek/contractnli-vast-ai-qwen3-1.7b` (HuggingFace Hub)
- **Evaluation**: 123 contracts from the ContractNLI test split
