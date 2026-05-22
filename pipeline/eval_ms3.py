"""
Headless MS3 evaluation runner.

Member 5 owns this file. It runs the agentic `run_full_analysis` pipeline
over the ContractNLI test split, reads the per-contract RunTrace files written
by M3, joins predictions with gold labels, writes the combined MS1/MS3
`evaluation.csv`, and zips `runs_ms3/` for submission.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    def _load_env_file(path: Path = Path(".env")) -> None:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    _load_env_file()

from src.bootstrap import setup_runtime
from src.constants import H_TO_NDA, LABEL_MAP
from src.runtrace import utc_now_iso
from src.types import ApprovalEvent, HypothesisTrace


DEFAULT_TEST_PATH = Path("data/test.json")
DEFAULT_RUNTRACE_DIR = Path("runs_ms3")
DEFAULT_EVALUATION_CSV = Path("evaluation.csv")
DEFAULT_ZIP_PATH = Path("runs_ms3.zip")
DETAILS_CSV_NAME = "evaluation_ms3_details.csv"
DEFAULT_MS1_AGGREGATE = {
    "ms1_label_accuracy": "0.851267",
    "ms1_groundedness": "1.0",
    "ms1_quote_integrity_pass_rate": "0.576758",
    "ms1_avg_latency_ms": "14106.34",
    "ms1_evaluation_timestamp": "2026-04-08T22:45:18.336+00:00",
}

MS3_FIELDNAMES = [
    "ms1_label_accuracy",
    "ms1_groundedness",
    "ms1_quote_integrity_pass_rate",
    "ms1_avg_latency_ms",
    "ms1_evaluation_timestamp",
    "ms3_label_accuracy",
    "ms3_groundedness",
    "ms3_quote_integrity_pass_rate",
    "ms3_avg_latency_ms",
    "ms3_contracts_evaluated",
    "ms3_hypotheses_evaluated",
    "ms3_runtrace_count",
    "ms3_retrieval_mode",
    "ms3_evaluation_timestamp",
]


@dataclass
class EvalSession:
    """Small SessionLike object for tools used inside the headless pipeline."""

    contract_id: str
    retrieval_mode: str
    cached_traces: Dict[str, HypothesisTrace] = field(default_factory=dict)
    approval_events: List[ApprovalEvent] = field(default_factory=list)


def load_documents(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    docs = payload.get("documents") if isinstance(payload, dict) else payload
    if not isinstance(docs, list):
        raise ValueError(f"{path} does not contain a ContractNLI document list")
    return docs


def gold_labels_for(doc: dict) -> Dict[str, str]:
    """Return {H01: ENTAILED|CONTRADICTED|NOT_MENTIONED} for one test doc."""
    sets = doc.get("annotation_sets") or []
    if not sets:
        return {}
    annotations = (sets[0] or {}).get("annotations") or {}
    gold: Dict[str, str] = {}
    for h_id, nda_key in H_TO_NDA.items():
        choice = (annotations.get(nda_key) or {}).get("choice")
        if choice in LABEL_MAP:
            gold[h_id] = LABEL_MAP[choice]
    return gold


def read_ms1_aggregate(path: Path) -> dict:
    """Read the existing aggregate-only MS1 evaluation.csv, if present."""
    if not path.exists():
        return dict(DEFAULT_MS1_AGGREGATE)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    row = rows[-1]
    ms1 = {
        "ms1_label_accuracy": row.get("label_accuracy", ""),
        "ms1_groundedness": row.get("groundedness", ""),
        "ms1_quote_integrity_pass_rate": row.get("quote_integrity_pass_rate", ""),
        "ms1_avg_latency_ms": row.get("avg_latency_ms", ""),
        "ms1_evaluation_timestamp": row.get("evaluation_timestamp", ""),
    }
    if not any(ms1.values()):
        ms1 = {k: row.get(k, "") for k in DEFAULT_MS1_AGGREGATE}
    if not any(ms1.values()):
        ms1 = dict(DEFAULT_MS1_AGGREGATE)
    return ms1


def atomic_write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent or Path(".")))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _contract_id_candidates(doc: dict, idx: int, result: Optional[dict] = None) -> List[str]:
    ids: List[str] = []
    if isinstance(result, dict):
        path = result.get("runtrace_path")
        if path:
            ids.append(str(path))
    raw_id = doc.get("id", idx)
    for value in (raw_id, f"doc_{raw_id}"):
        value = str(value)
        if value not in ids:
            ids.append(value)
    return ids


def find_runtrace(
    runtrace_dir: Path,
    doc: dict,
    idx: int,
    result: Optional[dict] = None,
) -> Path:
    for candidate in _contract_id_candidates(doc, idx, result):
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
        for name in (
            f"runtrace_doc_{candidate}.json",
            f"runtrace_{candidate}.json",
            f"{candidate}.json",
        ):
            path = runtrace_dir / name
            if path.exists():
                return path
    raise FileNotFoundError(
        f"Could not find MS3 RunTrace for test index {idx}; checked {runtrace_dir}"
    )


def load_runtrace(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def is_complete_contract_runtrace(rt: dict) -> bool:
    """Cheap validity gate for resume mode."""
    if rt.get("schema_version") != "3.0-ms3" or rt.get("runtrace_type") != "contract_analysis":
        return False
    traces = rt.get("hypothesis_traces")
    if not isinstance(traces, list) or len(traces) != 17:
        return False
    for trace in traces:
        if not {"hypothesis_id", "label", "evidence_spans", "playbook_result", "agent_metadata"} <= set(trace):
            return False
        if not {"severity", "action", "rationale"} <= set(trace.get("playbook_result") or {}):
            return False
        if not {"agent_id", "rag_query", "rag_hits", "rag_mode"} <= set(trace.get("agent_metadata") or {}):
            return False
    return True


def compute_metrics(runtraces: List[dict], docs_by_contract: Dict[str, dict]) -> tuple[dict, List[dict]]:
    total = 0
    correct = 0
    grounded = 0
    quote_ok = 0
    total_latency = 0.0
    detail_rows: List[dict] = []

    for rt in runtraces:
        contract = rt.get("contract") or {}
        contract_id = str(contract.get("contract_id", ""))
        doc = docs_by_contract.get(contract_id) or docs_by_contract.get(contract_id.replace("doc_", ""))
        gold = gold_labels_for(doc or {})

        metrics = rt.get("metrics") or {}
        total_latency += float(
            metrics.get("contract_latency_ms")
            or metrics.get("elapsed_ms")
            or metrics.get("avg_latency_ms")
            or 0.0
        )

        for trace in rt.get("hypothesis_traces") or []:
            h_id = trace.get("hypothesis_id")
            pred = trace.get("label")
            gold_label = trace.get("gold_label") or gold.get(h_id)
            if not h_id or not pred:
                continue
            total += 1
            is_correct = bool(gold_label and pred == gold_label)
            correct += int(is_correct)
            grounded += int(bool(trace.get("groundedness_check")))
            quote_ok += int(bool(trace.get("quote_integrity_check")))
            detail_rows.append(
                {
                    "contract_id": contract_id,
                    "hypothesis_id": h_id,
                    "gold_label": gold_label or "",
                    "ms3_label": pred,
                    "ms3_correct": int(is_correct),
                    "ms3_groundedness_check": int(bool(trace.get("groundedness_check"))),
                    "ms3_quote_integrity_check": int(bool(trace.get("quote_integrity_check"))),
                    "ms3_latency_ms": trace.get("latency_ms", ""),
                }
            )

    contract_count = len(runtraces)
    aggregate = {
        "ms3_label_accuracy": _ratio(correct, total),
        "ms3_groundedness": _ratio(grounded, total),
        "ms3_quote_integrity_pass_rate": _ratio(quote_ok, total),
        "ms3_avg_latency_ms": _ratio(total_latency, contract_count),
        "ms3_contracts_evaluated": contract_count,
        "ms3_hypotheses_evaluated": total,
        "ms3_runtrace_count": contract_count,
        "ms3_evaluation_timestamp": utc_now_iso(),
    }
    return aggregate, detail_rows


def _ratio(num: float, denom: float) -> float:
    return round(num / denom, 6) if denom else 0.0


def docs_index(docs: List[dict]) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for idx, doc in enumerate(docs):
        raw = str(doc.get("id", idx))
        mapping[raw] = doc
        mapping[f"doc_{raw}"] = doc
        mapping[str(idx)] = doc
        mapping[f"{idx:03d}"] = doc
        mapping[f"doc_{idx:03d}"] = doc
    return mapping


def import_pipeline():
    try:
        module = importlib.import_module("src.run_full_analysis")
    except ModuleNotFoundError as exc:
        try:
            dispatcher = importlib.import_module("src.dispatcher")
        except ModuleNotFoundError:
            raise RuntimeError(
                "M2/M3 pipeline is not available yet: expected either "
                "src/run_full_analysis.py with run_full_analysis(...) or "
                "src/dispatcher.py with make_pipeline(...)."
            ) from exc
        make_pipeline = getattr(dispatcher, "make_pipeline", None)
        if make_pipeline is None:
            raise RuntimeError("src.dispatcher exists but has no make_pipeline function") from exc
        return ("factory", make_pipeline)
    fn = getattr(module, "run_full_analysis", None)
    if fn is None:
        raise RuntimeError("src.run_full_analysis exists but has no run_full_analysis function")
    return ("function", fn)


def selected_retrievers(retrieval_mode: str, use_real_rag: bool) -> tuple[object | None, object | None]:
    """Bind only the requested RAG branch so missing optional deps do not break the other."""
    if not use_real_rag:
        return None, None
    if retrieval_mode == "graph":
        from src import rag_graph

        return None, rag_graph
    from src import rag_vector

    return rag_vector, None


async def run_batch(
    *,
    contract_path: Path = DEFAULT_TEST_PATH,
    retrieval_mode: str = "vector",
    limit: Optional[int] = None,
    start_idx: int = 0,
    runtrace_dir: Path = DEFAULT_RUNTRACE_DIR,
    evaluation_csv: Path = DEFAULT_EVALUATION_CSV,
    zip_path: Path = DEFAULT_ZIP_PATH,
    use_real_rag: bool = True,
    skip_existing: bool = False,
    resume: bool = False,
    write_zip: bool = True,
) -> dict:
    """
    Run MS3 over the test split and write evaluation artifacts.

    `skip_existing=True` is useful when M2/M3 already generated RunTraces and
    you only need to recompute the CSV and zip.
    """
    docs = load_documents(contract_path)
    selected = docs[start_idx : start_idx + limit if limit is not None else None]
    if not selected:
        raise ValueError("No documents selected for evaluation")

    pipeline_kind, pipeline_entry = (None, None) if skip_existing else import_pipeline()
    runtrace_dir.mkdir(parents=True, exist_ok=True)

    runtraces: List[dict] = []
    runtrace_paths: List[Path] = []
    started = time.perf_counter()
    for offset, doc in enumerate(selected, start=start_idx):
        contract_id = str(doc.get("id", offset))
        print(f"[MS3 eval] {offset + 1}/{len(docs)} contract_id={contract_id}", flush=True)

        result: Optional[dict] = None
        existing_path: Optional[Path] = None
        if resume or skip_existing:
            try:
                candidate = find_runtrace(runtrace_dir, doc, offset, None)
                if is_complete_contract_runtrace(load_runtrace(candidate)):
                    existing_path = candidate
            except FileNotFoundError:
                existing_path = None

        if existing_path is not None:
            print(f"[MS3 eval]   reuse {existing_path.name}", flush=True)
        elif not skip_existing:
            session = EvalSession(contract_id=contract_id, retrieval_mode=retrieval_mode)
            if pipeline_kind == "factory":
                analysis_fn = pipeline_entry(contract=doc)
            else:
                analysis_fn = pipeline_entry
            rag_vector, rag_graph = selected_retrievers(retrieval_mode, use_real_rag)
            setup_runtime(
                session=session,
                pipeline=analysis_fn,
                approval_gate=None,
                rag_vector=rag_vector,
                rag_graph=rag_graph,
                use_real_rag=False,
            )
            maybe_result = analysis_fn(contract_id, retrieval_mode)
            result = await maybe_result if asyncio.iscoroutine(maybe_result) else maybe_result

        path = existing_path or find_runtrace(runtrace_dir, doc, offset, result)
        runtrace_paths.append(path)
        runtraces.append(load_runtrace(path))

    aggregate, detail_rows = compute_metrics(runtraces, docs_index(docs))
    aggregate["ms3_retrieval_mode"] = retrieval_mode

    ms1 = read_ms1_aggregate(evaluation_csv)
    combined = {**{k: "" for k in MS3_FIELDNAMES}, **ms1, **aggregate}
    atomic_write_csv(evaluation_csv, MS3_FIELDNAMES, [combined])

    detail_path = runtrace_dir / DETAILS_CSV_NAME
    atomic_write_csv(
        detail_path,
        [
            "contract_id",
            "hypothesis_id",
            "gold_label",
            "ms3_label",
            "ms3_correct",
            "ms3_groundedness_check",
            "ms3_quote_integrity_check",
            "ms3_latency_ms",
        ],
        detail_rows,
    )

    if write_zip:
        zip_runtraces(runtrace_dir, zip_path, runtrace_paths)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        **aggregate,
        "evaluation_csv": str(evaluation_csv),
        "details_csv": str(detail_path),
        "zip_path": str(zip_path) if write_zip else "",
        "wall_clock_ms": round(elapsed_ms, 3),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def zip_runtraces(runtrace_dir: Path, zip_path: Path, runtrace_paths: Optional[List[Path]] = None) -> None:
    """Create the submission zip with all MS3 JSON RunTraces."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        paths = runtrace_paths if runtrace_paths is not None else sorted(runtrace_dir.glob("*.json"))
        for path in sorted(paths):
            zf.write(path, arcname=f"{runtrace_dir.name}/{path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ContractLens MS3 batch evaluation")
    parser.add_argument("--contract", default=str(DEFAULT_TEST_PATH), help="ContractNLI test JSON")
    parser.add_argument("--retrieval", choices=["vector", "graph"], default="vector")
    parser.add_argument("--limit", type=int, help="Evaluate only N docs")
    parser.add_argument("--start-idx", type=int, default=0, help="First document index")
    parser.add_argument("--runtrace-dir", default=str(DEFAULT_RUNTRACE_DIR))
    parser.add_argument("--evaluation-csv", default=str(DEFAULT_EVALUATION_CSV))
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP_PATH))
    parser.add_argument("--no-rag", action="store_true", help="Bind no real RAG retrievers")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not call the pipeline; recompute metrics from existing runs_ms3/*.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete existing RunTraces and call the pipeline only for missing/invalid contracts",
    )
    parser.add_argument("--no-zip", action="store_true", help="Do not create runs_ms3.zip")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(
            run_batch(
                contract_path=Path(args.contract),
                retrieval_mode=args.retrieval,
                limit=args.limit,
                start_idx=args.start_idx,
                runtrace_dir=Path(args.runtrace_dir),
                evaluation_csv=Path(args.evaluation_csv),
                zip_path=Path(args.zip_path),
                use_real_rag=not args.no_rag,
                skip_existing=args.skip_existing,
                resume=args.resume,
                write_zip=not args.no_zip,
            )
        )
    except Exception as exc:
        print(f"[MS3 eval error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
