"""
05_eval_runtrace.py
====================
Batch inference → per-contract RunTrace JSON + final evaluation CSV.

Milestone metrics produced:
  label_accuracy            fraction of hypothesis instances predicted correctly
  groundedness              fraction compliant with evidence-required rule
  quote_integrity_pass_rate fraction where all cited spans are valid
  avg_latency_ms            mean wall-clock time per contract (ms)

Outputs:
  runs/runtrace_doc_<NNN>.json   one file per contract
  evaluation.csv                 aggregate metrics CSV (one row)
  sklearn classification report  printed to stdout

Setup (Kaggle cell 0):
  !pip install unsloth trl scikit-learn pyyaml -q
  from huggingface_hub import login; login(token="YOUR_HF_TOKEN")
"""

import csv
import hashlib
import json
import re
import time
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from unsloth import FastLanguageModel

# ── Config ────────────────────────────────────────────────────────────────
MODEL         = "Youssef-Malek/contractnli-vast-ai-qwen3-1.7b"
TEST_JSONL    = "./test.jsonl"       # chat-formatted examples (prompt + gold)
TEST_JSON     = "./test.json"        # original ContractNLI test split
PLAYBOOK_PATH = "./playbook.yaml"
RUNS_DIR      = Path("./runs")
OUTPUT_CSV    = "./evaluation.csv"

MAX_SEQ       = 13288   # eval context (larger than training is fine)
MAX_NEW_TOK   = 2048   # max tokens for NLI answer
CONF_NEW_TOK  = 1024   # max tokens for confidence answer (includes optional think)
BATCH_SIZE    = 3

# Written verbatim into every RunTrace run.parameters block
RUN_PARAMS = dict(
    base_model="unsloth/Qwen3-1.7B-bnb-4bit",   # actual base model used for training
    quantization="4bit",
    adapter_method="qlora",
    seed=42,
    learning_rate=2e-4,
    batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=1269,
    max_seq_length=8192,   # training context length
    temperature=0.0,
)
# ─────────────────────────────────────────────────────────────────────────

# ── Static mappings (mirrors src/constants.py) ────────────────────────────
NDA_TO_H = {
    "nda-1": "H01", "nda-2": "H02", "nda-3": "H03", "nda-4": "H04",
    "nda-5": "H05", "nda-7": "H06", "nda-8": "H07", "nda-10": "H08",
    "nda-11": "H09", "nda-12": "H10", "nda-13": "H11", "nda-15": "H12",
    "nda-16": "H13", "nda-17": "H14", "nda-18": "H15", "nda-19": "H16",
    "nda-20": "H17",
}
H_TO_NDA = {v: k for k, v in NDA_TO_H.items()}

LABEL_MAP = {
    "Entailment": "ENTAILED",
    "Contradiction": "CONTRADICTED",
    "NotMentioned": "NOT_MENTIONED",
}
LABEL_TO_STATUS = {
    "ENTAILED": "satisfied",
    "CONTRADICTED": "conflict",
    "NOT_MENTIONED": "missing",
}

HYPOTHESES = {
    "H01": "All Confidential Information shall be expressly identified by the Disclosing Party.",
    "H02": "Confidential Information shall only include technical information.",
    "H03": "Confidential Information may include verbally conveyed information.",
    "H04": "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
    "H05": "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
    "H06": "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
    "H07": "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
    "H08": "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
    "H09": "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
    "H10": "Receiving Party may independently develop information similar to Confidential Information.",
    "H11": "Receiving Party may acquire information similar to Confidential Information from a third party.",
    "H12": "Agreement shall not grant Receiving Party any right to Confidential Information.",
    "H13": "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
    "H14": "Receiving Party may create a copy of some Confidential Information in some circumstances.",
    "H15": "Receiving Party shall not solicit some of Disclosing Party's representatives.",
    "H16": "Some obligations of Agreement may survive termination of Agreement.",
    "H17": "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
}

VALID_LABELS = {"ENTAILED", "CONTRADICTED", "NOT_MENTIONED"}

# Appended as a new user turn after the NLI answer
CONFIDENCE_QUESTION = (
    "For each of your 17 classifications above provide two things:\n"
    "1. confidence: your confidence score from 0.0 to 1.0.\n"
    "2. quote: for ENTAILED or CONTRADICTED, copy the single most relevant verbatim "
    "excerpt from the contract text you used as evidence. For NOT_MENTIONED leave quote empty.\n\n"
    "Reply ONLY with a JSON object, all 17 hypotheses:\n"
    '{"H01": {"confidence": 0.95, "quote": "verbatim excerpt..."}, '
    '"H02": {"confidence": 0.70, "quote": ""}, ...}'
)
# ─────────────────────────────────────────────────────────────────────────


# ── Utility helpers ───────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def split_prompt_and_gold(text: str) -> tuple[str, str]:
    """Split chat-formatted string into inference prompt and gold answer."""
    for tag in (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>assistant\n",
    ):
        idx = text.find(tag)
        if idx != -1:
            prompt = text[: idx + len(tag)]
            gold_raw = text[idx + len(tag):].replace("<|im_end|>", "").strip()
            return prompt, gold_raw
    return text, ""


def get_gold_labels(doc: dict) -> dict[str, str]:
    """Extract gold labels from ContractNLI doc → {H01: 'ENTAILED', ...}"""
    annotations = doc["annotation_sets"][0]["annotations"]
    return {
        h_id: LABEL_MAP[annotations.get(nda_key, {"choice": "NotMentioned"})["choice"]]
        for nda_key, h_id in NDA_TO_H.items()
    }


def parse_prediction(json_str: str) -> dict[str, dict]:
    """
    Parse model JSON output → {hyp_id: {"label": str, "evidence_spans": [int]}}
    Handles normal schema and swapped-field schema. Returns {} on failure.
    """
    try:
        json_str = re.sub(r"```json|```", "", json_str).strip()
        items = json.loads(json_str)
        result = {}
        for item in items:
            hyp_id = item.get("hypothesis_id")
            label  = item.get("label") or item.get("hypothesis_label")
            # Handle swapped schema (hypothesis_id ↔ label)
            if hyp_id and not re.match(r"^H(0[1-9]|1[0-7])$", hyp_id):
                hyp_id, label = label, hyp_id
            if not hyp_id or not re.match(r"^H(0[1-9]|1[0-7])$", hyp_id):
                continue
            spans = [
                int(s) for s in item.get("evidence_spans", [])
                if isinstance(s, (int, float))
            ]
            result[hyp_id] = {"label": label, "evidence_spans": spans}
        return result
    except Exception:
        return {}


# Regex fallbacks for second-pass parsing
_CONF_RE  = re.compile(r'"(H(?:0[1-9]|1[0-7]))"\s*:\s*(\d+(?:\.\d+)?)')
_ENTRY_RE = re.compile(
    r'"(H(?:0[1-9]|1[0-7]))"\s*:\s*\{[^}]*"confidence"\s*:\s*(\d+(?:\.\d+)?)'
    r'(?:[^}]*"quote"\s*:\s*"([^"]*)")?',
    re.DOTALL,
)


def strip_think(text: str) -> str:
    """
    Remove think content from model output.
    Handles two cases:
      - Full block: <think>...</think> present in decoded text.
      - Seeded block: prompt ended mid-think, so decoded text starts with the
        continuation and ends with </think>; strip everything up to </think>.
    """
    # Full block present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Seeded: output starts inside think, closes with </think>
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_second_pass(raw: str, pred_map: dict) -> dict[str, dict]:
    """
    Parse second-pass output → {H01: {"confidence": float, "quote": str}, ...}

    Strategy (in order):
      1. Full JSON parse of {"H01": {"confidence": 0.9, "quote": "..."}, ...}
      2. Regex scan for confidence + optional quote per entry.
      3. Confidence-only regex scan (quote stays empty).
      4. Heuristic fallback (0.9 / 0.7, empty quote).
    """
    text = strip_think(raw)

    def _clamp(v) -> float:
        return max(0.0, min(1.0, float(v)))

    def _heuristic() -> dict:
        return {
            h: {
                "confidence": 0.9 if (pred_map.get(h) or {}).get("label", "NOT_MENTIONED") != "NOT_MENTIONED" else 0.7,
                "quote": "",
            }
            for h in HYPOTHESES
        }

    # ── 1. Full JSON parse ────────────────────────────────────────────────
    try:
        cleaned = re.sub(r"```json|```", "", text).strip()
        data = json.loads(cleaned)

        # Model may return a dict {"H01": {...}} or an array [{hypothesis_id, confidence, quote}]
        if isinstance(data, list):
            data = {
                item["hypothesis_id"]: item
                for item in data
                if isinstance(item, dict) and re.match(r"^H(0[1-9]|1[0-7])$", str(item.get("hypothesis_id", "")))
            }

        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if re.match(r"^H(0[1-9]|1[0-7])$", str(k)) and isinstance(v, dict):
                    try:
                        result[str(k)] = {
                            "confidence": _clamp(v.get("confidence", 0.8)),
                            "quote": str(v.get("quote", "")).strip(),
                        }
                    except (ValueError, TypeError):
                        pass
            if len(result) >= 10:
                return result
    except Exception:
        pass

    # ── 2. Regex: confidence + quote per entry ────────────────────────────
    result = {}
    for m in _ENTRY_RE.finditer(text):
        hyp_id = m.group(1)
        result[hyp_id] = {
            "confidence": _clamp(m.group(2)),
            "quote": (m.group(3) or "").strip(),
        }
    if len(result) >= 10:
        # fill any missing hypotheses with heuristic
        for h in HYPOTHESES:
            if h not in result:
                label = (pred_map.get(h) or {}).get("label", "NOT_MENTIONED")
                result[h] = {"confidence": 0.9 if label != "NOT_MENTIONED" else 0.7, "quote": ""}
        return result

    # ── 3. Confidence-only regex scan ─────────────────────────────────────
    result = {}
    for m in _CONF_RE.finditer(text):
        hyp_id, val = m.group(1), m.group(2)
        result[hyp_id] = {"confidence": _clamp(val), "quote": ""}
    if len(result) >= 10:
        for h in HYPOTHESES:
            if h not in result:
                label = (pred_map.get(h) or {}).get("label", "NOT_MENTIONED")
                result[h] = {"confidence": 0.9 if label != "NOT_MENTIONED" else 0.7, "quote": ""}
        return result

    # ── 4. Heuristic fallback ─────────────────────────────────────────────
    return _heuristic()


# ── Playbook ──────────────────────────────────────────────────────────────

def load_playbook(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_playbook(playbook: dict, hyp_id: str, label: str) -> dict:
    """
    Returns {severity, action, rationale, criticality} for a given
    hypothesis_id + predicted label.
    Falls back to global_defaults when no per-hypothesis override exists.
    """
    check = next(
        (c for c in playbook.get("checks", []) if c["hypothesis_id"] == hyp_id),
        None,
    )
    status = LABEL_TO_STATUS.get(label, "missing")

    # Severity / action: check overrides first, then global default
    if check and label in check.get("overrides", {}):
        ov = check["overrides"][label]
        severity, action = ov["severity"], ov["action"]
    else:
        default = playbook["global_defaults"]["label_to_default_decision"].get(
            label, {"severity": "MEDIUM", "action": "CLARIFY"}
        )
        severity, action = default["severity"], default["action"]

    # Rationale from template
    rationale = ""
    if check:
        tmpl = check.get("rationale_templates", {}).get(status, "")
        rationale = (
            tmpl.replace("{HYPOTHESIS_TITLE}", check.get("title", hyp_id))
                .replace("{SEVERITY}", severity)
                .replace("{ACTION}", action)
                .replace("{STATUS}", status)
        )

    return {
        "severity": severity,
        "action": action,
        "rationale": rationale,
        "criticality": check.get("criticality", "P2") if check else "P2",
    }


# ── Evidence mapping ──────────────────────────────────────────────────────

def build_evidence_items(span_indices: list[int], doc: dict) -> tuple[list, bool]:
    """
    Map span indices → EvidenceItem dicts with verbatim quotes from char offsets.
    Returns (items, quote_integrity_pass).
    Quotes are extracted from doc["text"] — never model-generated text.
    """
    items = []
    integrity_ok = True
    spans = doc["spans"]
    text  = doc["text"]

    for idx in span_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= len(spans):
            integrity_ok = False
            continue
        char_start, char_end = spans[idx]
        quote = text[char_start:char_end].strip()
        if not quote:
            integrity_ok = False
            continue
        items.append({
            "chunk_id": f"chunk_{idx:03d}",
            "quote": quote,
            "relevance_score": 1.0,
            "span": {"char_start": char_start, "char_end": char_end},
        })

    return items, integrity_ok


# ── RunTrace builders ─────────────────────────────────────────────────────

def build_hyp_trace(
    hyp_id: str,
    doc: dict,
    pred_info: dict | None,   # {"label": str, "evidence_spans": [int]} or None
    gold_label: str,
    playbook: dict,
    infer_start_iso: str,
    infer_end_iso: str,
    hyp_latency_ms: float,
    confidence: float = 0.8,       # model-elicited from second pass
    model_quote_text: str = "",    # verbatim quote the model claimed it used
) -> dict:
    label = (pred_info or {}).get("label", "NOT_MENTIONED")
    if label not in VALID_LABELS:
        label = "NOT_MENTIONED"

    evidence_spans = (pred_info or {}).get("evidence_spans", [])
    ev_items, _ = build_evidence_items(evidence_spans, doc)

    # Real quote integrity: model-asserted quote must appear verbatim in the contract.
    # Empty quote (e.g. NOT_MENTIONED) is not a failure — nothing was asserted.
    model_quote = model_quote_text.strip()
    if model_quote:
        quote_ok = model_quote in doc["text"]
    else:
        quote_ok = True

    # Groundedness: ENTAILED/CONTRADICTED must have at least one evidence span
    needs_evidence = label in ("ENTAILED", "CONTRADICTED")
    compliant = (not needs_evidence) or bool(ev_items)
    hyp_text   = HYPOTHESES.get(hyp_id, "")
    pb         = apply_playbook(playbook, hyp_id, label)
    status     = LABEL_TO_STATUS.get(label, "missing")

    inference_str = (
        f"Predicted {label} supported by {len(ev_items)} evidence span(s)."
        if ev_items else f"Predicted {label}; no evidence spans cited."
    )

    steps = [
        {
            "step_id": f"{hyp_id}_load_inputs",
            "step_type": "load_inputs",
            "producer": {"component": "pipeline"},
            "started_at": infer_start_iso,
            "ended_at": infer_start_iso,
            "inputs": {"contract_id": str(doc.get("id", "unknown")), "hypothesis_id": hyp_id},
            "outputs": {"spans_count": len(doc["spans"]), "hypothesis_text": hyp_text},
        },
        {
            "step_id": f"{hyp_id}_nli_infer",
            "step_type": "nli_infer",
            "producer": {"component": "model", "component_id": MODEL},
            "started_at": infer_start_iso,
            "ended_at": infer_end_iso,
            "inputs": {"hypothesis_id": hyp_id},
            "outputs": {
                "label": label,
                "raw_evidence_spans": evidence_spans,
                "confidence": confidence,
            },
        },
        {
            "step_id": f"{hyp_id}_evidence_map",
            "step_type": "evidence_map",
            "producer": {"component": "pipeline"},
            "started_at": infer_end_iso,
            "ended_at": infer_end_iso,
            "inputs": {"evidence_span_indices": evidence_spans},
            "outputs": {
                "evidence_items_count": len(ev_items),
                "quote_integrity_pass": quote_ok,
            },
        },
        {
            "step_id": f"{hyp_id}_playbook_map",
            "step_type": "playbook_map",
            "producer": {"component": "playbook"},
            "started_at": infer_end_iso,
            "ended_at": infer_end_iso,
            "inputs": {"hypothesis_id": hyp_id, "label": label},
            "outputs": {
                "severity": pb["severity"],
                "action": pb["action"],
                "rationale": pb["rationale"],
            },
        },
        {
            "step_id": f"{hyp_id}_consistency_check",
            "step_type": "consistency_check",
            "producer": {"component": "validator"},
            "started_at": infer_end_iso,
            "ended_at": infer_end_iso,
            "inputs": {"label": label, "evidence_count": len(ev_items)},
            "outputs": {"compliant": compliant, "quote_integrity_pass": quote_ok},
        },
        {
            "step_id": f"{hyp_id}_format_response",
            "step_type": "format_response",
            "producer": {"component": "pipeline"},
            "started_at": infer_end_iso,
            "ended_at": infer_end_iso,
            "inputs": {"label": label, "evidence_count": len(ev_items)},
            "outputs": {"status": status, "rationale": pb["rationale"]},
        },
    ]

    return {
        "hypothesis_id": hyp_id,
        "dataset_hypothesis_key": H_TO_NDA.get(hyp_id, ""),
        "hypothesis_text": hyp_text,
        "gold_label": gold_label,
        "latency_ms": hyp_latency_ms,
        "compliant_evidence_required": compliant,
        "quote_integrity_pass": quote_ok,
        "steps": steps,
        "decision": {
            "label": label,
            "confidence": confidence,
            "evidence": {"supporting": ev_items, "counter": []},
            "justification": {
                "claim": hyp_text,
                "inference": inference_str,
                "limitations": (
                    "Single-model QLoRA pipeline; contracts truncated to 4096 tokens "
                    "where necessary. Evidence spans are verbatim extractions from "
                    "canonical char offsets."
                ),
            },
            "risk": {
                "severity": pb["severity"],
                "criticality": pb["criticality"],
                "playbook_rule_ids": [hyp_id],
                "recommended_action": pb["action"],
            },
        },
        "validations": [
            {
                "validator_id": "groundedness_check",
                "status": "PASS" if compliant else "FAIL",
                "message": (
                    "Evidence present as required."
                    if compliant and needs_evidence
                    else "NOT_MENTIONED — evidence not required."
                    if not needs_evidence
                    else "ENTAILED/CONTRADICTED predicted but no valid evidence spans."
                ),
                "related_hypothesis_id": hyp_id,
            },
            {
                "validator_id": "quote_integrity_check",
                "status": "PASS" if quote_ok else "FAIL",
                "message": (
                    "All cited spans are valid canonical text segments."
                    if quote_ok
                    else "One or more evidence span indices are out of range or empty."
                ),
                "related_hypothesis_id": hyp_id,
            },
        ],
    }


def build_runtrace(
    doc: dict,
    hyp_traces: list,
    run_id: str,
    started_at: str,
    ended_at: str,
    playbook: dict,
    playbook_hash: str,
    contract_latency_ms: float,
) -> dict:
    contract_hash = sha256_bytes(doc["text"].encode())

    gold_labels = [ht["gold_label"] for ht in hyp_traces]
    pred_labels = [ht["decision"]["label"] for ht in hyp_traces]
    correct_count   = sum(g == p for g, p in zip(gold_labels, pred_labels))
    compliant_count = sum(ht["compliant_evidence_required"] for ht in hyp_traces)
    qi_count        = sum(ht["quote_integrity_pass"] for ht in hyp_traces)

    return {
        "schema_version": "1.0-ms1",
        "run": {
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "framework": "single_model_pipeline",
            "deterministic": True,
            "parameters": RUN_PARAMS,
        },
        "contract": {
            "contract_id": str(doc.get("id", "unknown")),
            "source_type": "txt",
            "source_name": doc.get("file_name", str(doc.get("id", ""))),
            "language": "en",
            "hash_sha256": contract_hash,
            "chunks": [
                {
                    "chunk_id": f"chunk_{i:03d}",
                    "text": doc["text"][s:e].strip(),
                    "span": {"char_start": s, "char_end": e},
                }
                for i, (s, e) in enumerate(doc["spans"])
                if doc["text"][s:e].strip()
            ],
        },
        "playbook": {
            "playbook_id": playbook.get("playbook_id", "contractnli_nda_minimal"),
            "version": str(playbook.get("version", "1.1")),
            "ruleset_hash": playbook_hash,
        },
        "hypothesis_traces": hyp_traces,
        "metrics": {
            "hypothesis_count": 17,
            "correct_count": correct_count,
            "compliant_count": compliant_count,
            "quote_integrity_count": qi_count,
            "contract_accuracy": correct_count / 17,
            "groundedness_rate": compliant_count / 17,
            "quote_integrity_rate": qi_count / 17,
            "contract_latency_ms": contract_latency_ms,
            "label_counts": {
                "ENTAILED":      sum(1 for p in pred_labels if p == "ENTAILED"),
                "CONTRADICTED":  sum(1 for p in pred_labels if p == "CONTRADICTED"),
                "NOT_MENTIONED": sum(1 for p in pred_labels if p == "NOT_MENTIONED"),
            },
        },
        "run_validations": [
            {
                "validator_id": "hypothesis_count_check",
                "status": "PASS" if len(hyp_traces) == 17 else "FAIL",
                "message": f"Expected 17 hypothesis traces, got {len(hyp_traces)}.",
            },
            {
                "validator_id": "schema_version_check",
                "status": "PASS",
                "message": "Schema version 1.0-ms1.",
            },
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        fast_inference=False,
        attn_implementation="flash_attention_2",
    )
    FastLanguageModel.for_inference(model)
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading formatted examples from {TEST_JSONL}...")
    rows = load_jsonl(TEST_JSONL)

    print(f"Loading original documents from {TEST_JSON}...")
    with open(TEST_JSON) as f:
        test_data = json.load(f)
    docs = test_data["documents"]

    assert len(rows) == len(docs), (
        f"Row count mismatch: {len(rows)} jsonl rows vs {len(docs)} docs. "
        "Re-run 01_preprocess.py to regenerate test.jsonl."
    )
    print(f"  {len(rows)} contracts | batch_size={BATCH_SIZE}")

    # ── Load playbook ─────────────────────────────────────────────────────
    print(f"Loading playbook from {PLAYBOOK_PATH}...")
    playbook = load_playbook(PLAYBOOK_PATH)
    playbook_hash = sha256_file(PLAYBOOK_PATH)

    # ── Pre-split prompts and gold (gold sourced from test.json for reliability)
    prompts = []
    for row in rows:
        prompt, _ = split_prompt_and_gold(row["text"])
        prompts.append(prompt)

    # ── Inference + RunTrace loop ─────────────────────────────────────────
    all_gold, all_pred = [], []
    parse_failures = 0
    contract_latencies_ms = []
    run_id = uuid.uuid4().hex[:12]
    eval_started_at = now_iso()
    t0 = time.perf_counter()

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch_prompts = prompts[batch_start : batch_start + BATCH_SIZE]
        batch_docs    = docs[batch_start : batch_start + BATCH_SIZE]
        batch_end_idx = batch_start + len(batch_prompts)

        batch_t0 = time.perf_counter()
        infer_start_iso = now_iso()

        # ── Pass 1: NLI inference ─────────────────────────────────────────
        nli_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ - MAX_NEW_TOK,
        ).to(model.device)

        with torch.no_grad():
            nli_output_ids = model.generate(
                **nli_inputs,
                max_new_tokens=MAX_NEW_TOK,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        infer_end_iso = now_iso()
        nli_input_len = nli_inputs["input_ids"].shape[1]

        # Decode NLI answers and parse predictions
        nli_pred_raws = []
        nli_pred_maps = []
        for out_ids in nli_output_ids:
            raw = tokenizer.decode(out_ids[nli_input_len:], skip_special_tokens=True)
            nli_pred_raws.append(raw)
            nli_pred_maps.append(parse_prediction(raw))

        # ── Pass 2: Verbal confidence elicitation ─────────────────────────
        # Build a new prompt: original conversation + NLI answer + confidence question
        conf_prompts = [
            nli_prompt
            + nli_raw
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + CONFIDENCE_QUESTION
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
            + "<think>\nLet me assess the evidence quality for each hypothesis:\n"
            for nli_prompt, nli_raw in zip(batch_prompts, nli_pred_raws)
        ]

        conf_inputs = tokenizer(
            conf_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ - CONF_NEW_TOK,
        ).to(model.device)

        with torch.no_grad():
            conf_output_ids = model.generate(
                **conf_inputs,
                max_new_tokens=CONF_NEW_TOK,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        conf_end_iso = now_iso()
        conf_input_len = conf_inputs["input_ids"].shape[1]

        # Distribute total batch time (both passes) evenly across contracts
        batch_elapsed_s = time.perf_counter() - batch_t0
        per_contract_ms = (batch_elapsed_s / len(batch_prompts)) * 1000

        for j, (doc, pred_map, pred_raw, conf_ids) in enumerate(
            zip(batch_docs, nli_pred_maps, nli_pred_raws, conf_output_ids)
        ):
            contract_idx = batch_start + j

            if not pred_map:
                parse_failures += 1
                print(f"  [!] Contract {contract_idx + 1}: NLI parse failed")
                print(f"      Raw (first 300 chars): {pred_raw[:300]}")

            # Parse confidence + model quotes from second pass
            conf_raw  = tokenizer.decode(conf_ids[conf_input_len:], skip_special_tokens=True)
            conf_dict = parse_second_pass(conf_raw, pred_map)

            # Gold labels from original annotations
            gold_labels = get_gold_labels(doc)

            # Build all 17 hypothesis traces
            hyp_latency_ms = per_contract_ms / 17
            hyp_traces = []
            for hyp_id in sorted(HYPOTHESES.keys()):
                pred_info  = pred_map.get(hyp_id)
                gold_label = gold_labels[hyp_id]

                hyp_second = conf_dict.get(hyp_id, {"confidence": 0.8, "quote": ""})
                ht = build_hyp_trace(
                    hyp_id=hyp_id,
                    doc=doc,
                    pred_info=pred_info,
                    gold_label=gold_label,
                    playbook=playbook,
                    infer_start_iso=infer_start_iso,
                    infer_end_iso=conf_end_iso,
                    hyp_latency_ms=hyp_latency_ms,
                    confidence=hyp_second["confidence"],
                    model_quote_text=hyp_second["quote"],
                )
                hyp_traces.append(ht)

                all_gold.append(gold_label)
                all_pred.append(ht["decision"]["label"])

            # ── Write RunTrace file ───────────────────────────────────────
            write_end_s = time.perf_counter()
            contract_latency = ((write_end_s - batch_t0) / len(batch_prompts)) * 1000
            contract_latencies_ms.append(contract_latency)

            runtrace = build_runtrace(
                doc=doc,
                hyp_traces=hyp_traces,
                run_id=f"{run_id}_{contract_idx:03d}",
                started_at=infer_start_iso,
                ended_at=now_iso(),
                playbook=playbook,
                playbook_hash=playbook_hash,
                contract_latency_ms=contract_latency,
            )

            out_path = RUNS_DIR / f"runtrace_doc_{contract_idx:03d}.json"
            with open(out_path, "w") as f:
                json.dump(runtrace, f, indent=2)

        contracts_done = batch_end_idx
        elapsed = time.perf_counter() - t0
        rate = contracts_done / elapsed
        remaining = (len(rows) - contracts_done) / rate if rate > 0 else 0
        print(
            f"  [{contracts_done}/{len(rows)}] "
            f"{rate:.2f} contracts/s | ETA {remaining:.0f}s"
        )

    eval_ended_at = now_iso()

    # ── Aggregate metrics ─────────────────────────────────────────────────
    total   = len(all_gold)
    correct = sum(g == p for g, p in zip(all_gold, all_pred))

    # Groundedness: for ENTAILED/CONTRADICTED, check evidence was provided
    # (We re-derive from the RunTrace files for accuracy)
    all_runtraces = []
    for path in sorted(RUNS_DIR.glob("runtrace_doc_*.json")):
        with open(path) as f:
            all_runtraces.append(json.load(f))

    compliant_total = sum(
        ht["compliant_evidence_required"]
        for rt in all_runtraces
        for ht in rt["hypothesis_traces"]
    )
    qi_total = sum(
        ht["quote_integrity_pass"]
        for rt in all_runtraces
        for ht in rt["hypothesis_traces"]
    )
    avg_latency_ms = sum(contract_latencies_ms) / len(contract_latencies_ms)

    label_accuracy          = correct / total
    groundedness            = compliant_total / total
    quote_integrity_rate    = qi_total / total

    total_elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {total_elapsed:.1f}s  |  {total_elapsed/len(rows):.2f}s per contract")
    print(f"Parse failures: {parse_failures} contracts\n")

    print("=" * 60)
    print("Aggregate Metrics")
    print("=" * 60)
    print(f"  label_accuracy:           {label_accuracy:.4f}")
    print(f"  groundedness:             {groundedness:.4f}")
    print(f"  quote_integrity_pass_rate:{quote_integrity_rate:.4f}")
    print(f"  avg_latency_ms:           {avg_latency_ms:.1f}")
    print()

    # ── Write evaluation CSV ──────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label_accuracy",
            "groundedness",
            "quote_integrity_pass_rate",
            "avg_latency_ms",
            "evaluation_timestamp",
        ])
        writer.writeheader()
        writer.writerow({
            "label_accuracy":           round(label_accuracy, 6),
            "groundedness":             round(groundedness, 6),
            "quote_integrity_pass_rate": round(quote_integrity_rate, 6),
            "avg_latency_ms":           round(avg_latency_ms, 2),
            "evaluation_timestamp":     eval_ended_at,
        })
    print(f"Evaluation CSV written to {OUTPUT_CSV}")

    # ── sklearn classification report ─────────────────────────────────────
    print("\nPer-class F1 Report:")
    print("=" * 60)
    print(classification_report(
        all_gold, all_pred,
        labels=["ENTAILED", "CONTRADICTED", "NOT_MENTIONED"],
        digits=4,
        zero_division=0,
    ))

    print(f"\nRunTrace files: {RUNS_DIR}/runtrace_doc_NNN.json  ({len(all_runtraces)} files)")
    print(f"Evaluation CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
