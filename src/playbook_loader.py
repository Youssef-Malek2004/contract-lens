"""
src/playbook_loader.py

Loads `playbook.yaml` (used unedited — MS3 hard constraint) and exposes a
typed view over it for the aggregator.

The playbook is a dict with:
  - global_defaults.evidence_required_for         list[str]
  - global_defaults.label_to_status               dict[label -> status]
  - global_defaults.label_to_default_decision     dict[label -> {severity, action}]
  - criticality_tiers                             dict[tier -> description]
  - template_slots                                dict (informational)
  - checks                                        list[check] (one per H01..H17)

Per-check decision resolution:
  1. If the check defines `overrides[label]`, use that {severity, action}.
  2. Else, fall back to global_defaults.label_to_default_decision[label].
  3. Status comes from global_defaults.label_to_status[label].

Rationale templates live under each check's `rationale_templates[status]`
keyed by the lowercased status string ("satisfied" / "conflict" / "missing").
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_PLAYBOOK_PATH = Path("playbook.yaml")


class PlaybookError(RuntimeError):
    """Raised when the playbook YAML is missing required structure."""


class Playbook:
    """
    Thin typed view over the raw playbook YAML. Holds the dict as-is (so it
    can be re-embedded into the contract RunTrace unmodified) and adds the
    lookups the aggregator needs.
    """

    def __init__(self, raw: Dict[str, Any], ruleset_hash: str) -> None:
        if not isinstance(raw, dict):
            raise PlaybookError("playbook.yaml must parse to a mapping")

        global_defaults = raw.get("global_defaults") or {}
        if not isinstance(global_defaults, dict):
            raise PlaybookError("playbook.global_defaults must be a mapping")

        self.raw: Dict[str, Any] = raw
        self.playbook_id: str = str(raw.get("playbook_id") or "unknown")
        self.version: str = str(raw.get("version") or "0")
        self.ruleset_hash: str = ruleset_hash

        self._label_to_status: Dict[str, str] = dict(
            global_defaults.get("label_to_status") or {}
        )
        self._label_to_default_decision: Dict[str, Dict[str, str]] = {
            label: dict(decision or {})
            for label, decision in (global_defaults.get("label_to_default_decision") or {}).items()
        }
        self._evidence_required_for: set[str] = set(
            global_defaults.get("evidence_required_for") or []
        )

        self._checks_by_id: Dict[str, Dict[str, Any]] = {}
        for check in raw.get("checks") or []:
            if not isinstance(check, dict):
                continue
            h_id = check.get("hypothesis_id")
            if isinstance(h_id, str):
                self._checks_by_id[h_id] = check

    def get_check(self, h_id: str) -> Optional[Dict[str, Any]]:
        return self._checks_by_id.get(h_id)

    def status_for_label(self, label: str) -> str:
        return self._label_to_status.get(label, "missing")

    def evidence_required(self, label: str) -> bool:
        return label in self._evidence_required_for

    def resolve_decision(self, h_id: str, label: str) -> Dict[str, str]:
        """
        Returns {"severity": ..., "action": ...} for this hypothesis+label,
        applying per-check overrides on top of global_defaults.
        """
        default = dict(self._label_to_default_decision.get(label) or {})
        check = self._checks_by_id.get(h_id) or {}
        override = (check.get("overrides") or {}).get(label) or {}
        if isinstance(override, dict):
            default.update({k: v for k, v in override.items() if v is not None})
        return {
            "severity": str(default.get("severity") or "MEDIUM"),
            "action": str(default.get("action") or "CLARIFY"),
        }

    def rationale_template(self, h_id: str, status: str) -> Optional[str]:
        check = self._checks_by_id.get(h_id) or {}
        templates = check.get("rationale_templates") or {}
        template = templates.get(status)
        return str(template) if template else None

    def title(self, h_id: str) -> Optional[str]:
        check = self._checks_by_id.get(h_id) or {}
        title = check.get("title")
        return str(title) if title else None

    def criticality(self, h_id: str) -> Optional[str]:
        check = self._checks_by_id.get(h_id) or {}
        crit = check.get("criticality")
        return str(crit) if crit else None


@lru_cache(maxsize=4)
def load_playbook(path: str | Path = DEFAULT_PLAYBOOK_PATH) -> Playbook:
    """
    Load and cache the playbook from disk. The cache is keyed by the
    string form of the path; cheap to call repeatedly from the dispatcher.
    """
    p = Path(path)
    if not p.exists():
        raise PlaybookError(f"playbook not found at {p}")
    raw_bytes = p.read_bytes()
    ruleset_hash = hashlib.sha256(raw_bytes).hexdigest()
    data = yaml.safe_load(raw_bytes.decode("utf-8"))
    return Playbook(data, ruleset_hash=ruleset_hash)
