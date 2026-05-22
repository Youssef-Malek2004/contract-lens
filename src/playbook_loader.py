"""Load and apply the MS1 playbook without editing playbook.yaml."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

from src.constants import LABEL_TO_STATUS


DEFAULT_PLAYBOOK_PATH = Path("playbook.yaml")


class Playbook:
    def __init__(self, raw: dict, ruleset_hash: str) -> None:
        self.raw = raw
        self.ruleset_hash = ruleset_hash
        self.checks = {c["hypothesis_id"]: c for c in raw.get("checks", [])}
        self.defaults = raw.get("global_defaults", {})

    def metadata(self) -> dict:
        return {
            "playbook_id": self.raw.get("playbook_id", "unknown"),
            "version": str(self.raw.get("version", "")),
            "ruleset_hash": self.ruleset_hash,
            "rule_params": {
                "source": self.raw.get("source", ""),
                "checks": len(self.raw.get("checks", [])),
            },
        }

    def apply(self, hypothesis_id: str, label: str, confidence: float | None = None) -> dict:
        check = self.checks.get(hypothesis_id, {})
        status = LABEL_TO_STATUS.get(label, "missing")
        default_decision = (
            self.defaults.get("label_to_default_decision", {}).get(label, {})
        )
        override = (check.get("overrides") or {}).get(label, {})
        severity = override.get("severity") or default_decision.get("severity") or "MEDIUM"
        action = override.get("action") or default_decision.get("action") or "CLARIFY"
        template = (check.get("rationale_templates") or {}).get(status)
        title = check.get("title") or hypothesis_id
        if not template:
            template = (
                "{HYPOTHESIS_TITLE}: classified as {STATUS}. "
                "Severity {SEVERITY}; action {ACTION}."
            )
        rationale = template.format(
            HYPOTHESIS_TITLE=title,
            STATUS=status,
            SEVERITY=severity,
            ACTION=action,
            EVIDENCE_SUMMARY="",
            TOP_CITATION="",
            CONFIDENCE="" if confidence is None else f"{confidence:.2f}",
        )
        return {
            "severity": severity,
            "action": action,
            "rationale": rationale,
        }


def load_playbook(path: str | Path = DEFAULT_PLAYBOOK_PATH) -> Playbook:
    fpath = Path(path)
    raw_bytes = fpath.read_bytes()
    raw = _parse_playbook_yaml(raw_bytes.decode("utf-8"))
    return Playbook(raw=raw, ruleset_hash=hashlib.sha256(raw_bytes).hexdigest())


def _parse_playbook_yaml(text: str) -> dict:
    """
    Tiny parser for this repository's fixed playbook.yaml shape.

    We avoid requiring PyYAML in eval environments while still reading the
    unedited playbook file. It extracts exactly the fields the aggregator uses.
    """
    raw: dict[str, Any] = {
        "global_defaults": {
            "label_to_default_decision": {
                "ENTAILED": {"severity": "LOW", "action": "ACCEPT"},
                "CONTRADICTED": {"severity": "HIGH", "action": "ESCALATE"},
                "NOT_MENTIONED": {"severity": "MEDIUM", "action": "CLARIFY"},
            }
        },
        "checks": [],
    }
    current: dict[str, Any] | None = None
    section: str | None = None
    subsection: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0 and not line.startswith("- "):
            key, value = _split_key_value(line)
            if key in {"playbook_id", "version", "source"}:
                raw[key] = _strip_scalar(value)
            elif key in {"global_defaults", "checks"}:
                section = key
                subsection = None
            continue

        if section == "global_defaults":
            key, value = _split_key_value(line)
            if key == "label_to_default_decision":
                subsection = key
                continue
            if subsection == "label_to_default_decision" and key in LABEL_TO_STATUS:
                raw["global_defaults"]["label_to_default_decision"][key] = _parse_inline_map(value)
            continue

        if section == "checks":
            if line.startswith("- hypothesis_id:"):
                current = {"hypothesis_id": _strip_scalar(line.split(":", 1)[1]), "overrides": {}, "rationale_templates": {}}
                raw["checks"].append(current)
                subsection = None
                continue
            if current is None:
                continue
            key, value = _split_key_value(line)
            if key in {"title", "hypothesis_text", "criticality"}:
                current[key] = _strip_scalar(value)
            elif key in {"overrides", "rationale_templates"}:
                subsection = key
            elif subsection == "overrides" and key in LABEL_TO_STATUS:
                current["overrides"][key] = _parse_inline_map(value)
            elif subsection == "rationale_templates" and key in {"satisfied", "conflict", "missing"}:
                current["rationale_templates"][key] = _strip_scalar(value)

    return raw


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        return line, ""
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _strip_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_inline_map(value: str) -> dict:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    out = {}
    for part in value.split(","):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        out[key.strip()] = _strip_scalar(val.strip())
    return out
