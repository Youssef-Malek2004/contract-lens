"""
src/aggregator.py

Post-dispatch enrichment layer. Sits between M2's hypothesis workers and the
contract RunTrace writer:

    dispatcher.run_full_analysis()
        ├─ asyncio.gather(17 workers)            → list[HypothesisTrace] (raw)
        ├─ aggregate_traces(traces, playbook)    ← THIS MODULE
        │      ├─ enrich each trace.playbook_result with severity/action/rationale
        │      ├─ add evidence-required validation_failures missed by the worker
        │      └─ compute risk metrics across all 17
        └─ write_contract_runtrace(...)

The aggregator does NOT mutate trace shape beyond what TypedDicts allow —
worker fields (label, evidence_spans, verbatim_quote, etc.) are kept as-is.
Only `playbook_result` is replaced and `validation_failures` may be appended.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.playbook_loader import Playbook
from src.types import HypothesisTrace


@dataclass
class AggregationResult:
    traces: List[HypothesisTrace]
    playbook_envelope: dict
    risk_metrics: dict


def aggregate_traces(
    traces: List[HypothesisTrace],
    playbook: Playbook,
) -> AggregationResult:
    """
    Enrich every trace's `playbook_result` from the playbook, append any
    missing groundedness/evidence-required validation failures, and compute
    aggregate risk metrics.

    Pure function — does not write to disk or mutate the playbook.
    """
    enriched: List[HypothesisTrace] = []
    severity_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    criticality_counts: Counter[str] = Counter()

    for trace in traces:
        h_id = str(trace.get("hypothesis_id") or "")
        label = str(trace.get("label") or "NOT_MENTIONED")
        evidence_ids = list(trace.get("evidence_spans") or [])

        status = playbook.status_for_label(label)
        decision = playbook.resolve_decision(h_id, label)
        severity = decision["severity"]
        action = decision["action"]
        title = playbook.title(h_id) or h_id
        template = playbook.rationale_template(h_id, status)

        rationale = _fill_template(
            template=template,
            title=title,
            severity=severity,
            action=action,
            fallback_reason=str(
                ((trace.get("playbook_result") or {}).get("reason") or "")
            ),
        )

        playbook_result = {
            "severity": severity,
            "action": action,
            "rationale": rationale,
        }

        new_trace: HypothesisTrace = dict(trace)  # type: ignore[assignment]
        new_trace["playbook_result"] = playbook_result  # type: ignore[typeddict-item]

        failures = list(new_trace.get("validation_failures") or [])
        if (
            playbook.evidence_required(label)
            and not evidence_ids
            and not _has_failure(failures, "groundedness_check")
        ):
            failures.append({
                "validator": "groundedness_check",
                "message": (
                    f"Label {label} requires evidence per playbook, "
                    "but no evidence_spans were cited."
                ),
                "related_hypothesis_id": h_id,
            })
            new_trace["groundedness_check"] = False
        if failures:
            new_trace["validation_failures"] = failures

        enriched.append(new_trace)

        severity_counts[severity] += 1
        action_counts[action] += 1
        status_counts[status] += 1
        criticality_counts[playbook.criticality(h_id) or "P2"] += 1

    risk_metrics = {
        "by_severity": dict(severity_counts),
        "by_action": dict(action_counts),
        "by_status": dict(status_counts),
        "by_criticality": dict(criticality_counts),
        "high_risk_count": severity_counts.get("HIGH", 0),
        "escalate_count": action_counts.get("ESCALATE", 0),
        "validation_failure_count": sum(
            len(t.get("validation_failures") or []) for t in enriched
        ),
    }

    playbook_envelope = {
        "playbook_id": playbook.playbook_id,
        "version": playbook.version,
        "ruleset_hash": playbook.ruleset_hash,
        "rule_params": {
            "applied": True,
            "raw": playbook.raw,
        },
    }

    return AggregationResult(
        traces=enriched,
        playbook_envelope=playbook_envelope,
        risk_metrics=risk_metrics,
    )


def _fill_template(
    *,
    template: Optional[str],
    title: str,
    severity: str,
    action: str,
    fallback_reason: str,
) -> str:
    """Substitute the documented placeholders in a rationale template."""
    if not template:
        if fallback_reason:
            return fallback_reason
        return f"{title}: severity {severity}; action {action}."
    return (
        template
        .replace("{HYPOTHESIS_TITLE}", title)
        .replace("{SEVERITY}", severity)
        .replace("{ACTION}", action)
    )


def _has_failure(failures: List[dict], validator: str) -> bool:
    return any(str(f.get("validator") or "") == validator for f in failures)
