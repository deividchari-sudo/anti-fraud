"""Persistence layer for rule definitions.

Follows the Repository pattern used elsewhere in the project (see
``src/repositories.py``). The default implementation persists rules to a
JSON file but the abstract base allows other backends (SQLite, S3, etc.).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from .rule import ActionType, Condition, ConditionType, Operator, Rule


class RuleRepository(ABC):
    """Abstract repository for rule persistence."""

    @abstractmethod
    def load_all(self) -> List[Rule]:
        """Return all persisted rules."""

    @abstractmethod
    def save_all(self, rules: List[Rule]) -> None:
        """Persist the given list of rules atomically."""


def _condition_to_dict(c: Condition) -> dict:
    return {
        "field": c.field.value,
        "operator": c.operator.value,
        "value": c.value,
        "negated": c.negated,
    }


def _condition_from_dict(data: dict) -> Condition:
    return Condition(
        field=ConditionType(data["field"]),
        operator=Operator(data["operator"]),
        value=data["value"],
        negated=bool(data.get("negated", False)),
    )


def rule_to_dict(rule: Rule) -> dict:
    """Serialize a Rule to a JSON-compatible dictionary."""
    return {
        "id": rule.id,
        "name": rule.name,
        "description": rule.description,
        "original_text": rule.original_text,
        "action": rule.action.value,
        "enabled": rule.enabled,
        "priority": rule.priority,
        "conditions": [_condition_to_dict(c) for c in rule.conditions],
        "condition_groups": [
            [_condition_to_dict(c) for c in group]
            for group in rule.condition_groups
        ],
    }


def rule_from_dict(data: dict) -> Rule:
    """Deserialize a Rule from a dictionary previously produced by ``rule_to_dict``."""
    return Rule(
        id=data["id"],
        name=data["name"],
        description=data.get("description", ""),
        original_text=data.get("original_text", ""),
        conditions=[_condition_from_dict(c) for c in data.get("conditions", [])],
        action=ActionType(data["action"]),
        enabled=bool(data.get("enabled", True)),
        priority=int(data.get("priority", 0)),
        condition_groups=[
            [_condition_from_dict(c) for c in group]
            for group in data.get("condition_groups", [])
        ],
    )


class JSONRuleRepository(RuleRepository):
    """File-backed repository persisting rules as JSON.

    The file is created on first ``save_all`` if it does not exist. Reads
    return an empty list when the file is missing, which is the expected
    cold-start behavior.
    """

    def __init__(self, file_path: str = "data/rules.json"):
        self.file_path = Path(file_path)

    def load_all(self) -> List[Rule]:
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError:
            return []
        return [rule_from_dict(d) for d in payload.get("rules", [])]

    def save_all(self, rules: List[Rule]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 2, "rules": [rule_to_dict(r) for r in rules]}
        # Atomic write: write to temp then rename.
        tmp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.file_path)


class InMemoryRuleRepository(RuleRepository):
    """In-memory repository (useful for tests and dependency injection)."""

    def __init__(self) -> None:
        self._rules: List[dict] = []

    def load_all(self) -> List[Rule]:
        return [rule_from_dict(d) for d in self._rules]

    def save_all(self, rules: List[Rule]) -> None:
        self._rules = [rule_to_dict(r) for r in rules]
