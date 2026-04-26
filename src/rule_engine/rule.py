"""
Rule data structures for fraud detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class ActionType(Enum):
    """Action to take when rule matches."""

    MARK_AS_FRAUD = "mark_as_fraud"
    MARK_AS_LEGITIMATE = "mark_as_legitimate"


class ConditionType(Enum):
    """Type of condition."""

    CANAL = "canal"
    VALOR = "valor"
    HORARIO = "horario"
    CPF_SENDER = "cpf_sender"
    CPF_RECEIVER = "cpf_receiver"
    BANCO_SENDER = "banco_sender"
    BANCO_RECEIVER = "banco_receiver"


class Operator(Enum):
    """Comparison operator."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    CONTAINS = "contains"
    AFTER = "after"
    BEFORE = "before"


@dataclass
class Condition:
    """Single condition in a rule.

    Attributes:
        field: Which transaction field this condition applies to.
        operator: Comparison operator.
        value: Right-hand value to compare against.
        negated: When True, the boolean result of the comparison is inverted (NOT).
    """

    field: "ConditionType"
    operator: Operator
    value: Any
    negated: bool = False

    def evaluate(self, transaction: dict) -> bool:
        """Evaluate condition against transaction."""
        result = self._evaluate_raw(transaction)
        return (not result) if self.negated else result

    def _evaluate_raw(self, transaction: dict) -> bool:
        field_value = self._get_field_value(transaction)

        if self.operator == Operator.EQUALS:
            return field_value == self.value
        elif self.operator == Operator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == Operator.GREATER_THAN:
            return field_value > self.value
        elif self.operator == Operator.LESS_THAN:
            return field_value < self.value
        elif self.operator == Operator.GREATER_THAN_OR_EQUAL:
            return field_value >= self.value
        elif self.operator == Operator.LESS_THAN_OR_EQUAL:
            return field_value <= self.value
        elif self.operator == Operator.CONTAINS:
            return self.value in str(field_value)
        elif self.operator == Operator.AFTER:
            return field_value > self.value
        elif self.operator == Operator.BEFORE:
            return field_value < self.value

        return False

    def _get_field_value(self, transaction: dict) -> Any:
        """Extract field value from transaction."""
        if self.field == ConditionType.CANAL:
            return transaction.get("canal", "").lower()
        elif self.field == ConditionType.VALOR:
            return float(transaction.get("valor", 0))
        elif self.field == ConditionType.HORARIO:
            # Extract hour from timestamp
            timestamp = transaction.get("timestamp", "")
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return dt.hour
            except Exception:
                return 0
        elif self.field == ConditionType.CPF_SENDER:
            return transaction.get("sender", {}).get("cpfSender", "")
        elif self.field == ConditionType.CPF_RECEIVER:
            return transaction.get("receiver", {}).get("cpfReceiver", "")
        elif self.field == ConditionType.BANCO_SENDER:
            return transaction.get("sender", {}).get("banco", 0)
        elif self.field == ConditionType.BANCO_RECEIVER:
            return transaction.get("receiver", {}).get("banco", 0)

        return None


@dataclass
class Rule:
    """Fraud detection rule.

    Logical model (V2):
        - ``conditions``: legacy AND-only list (kept for backward compatibility).
        - ``condition_groups``: ``OR`` of groups, with ``AND`` inside each group.
          When non-empty, takes precedence over ``conditions``.
    """

    id: str
    name: str
    description: str
    original_text: str
    conditions: List[Condition]
    action: ActionType
    enabled: bool = True
    priority: int = 0
    condition_groups: List[List[Condition]] = field(default_factory=list)

    def evaluate(self, transaction: dict) -> bool:
        """Evaluate rule against transaction.

        Uses ``condition_groups`` (OR of AND-groups) when present, otherwise
        falls back to ``conditions`` (single AND group).
        """
        if not self.enabled:
            return False

        if self.condition_groups:
            # OR across groups, AND within each group
            return any(
                all(c.evaluate(transaction) for c in group)
                for group in self.condition_groups
                if group
            )

        # Legacy AND-only path
        return all(condition.evaluate(transaction) for condition in self.conditions)

    def matches(self, transaction: dict) -> Optional[bool]:
        """Check if rule matches and return action result."""
        if self.evaluate(transaction):
            return self.action == ActionType.MARK_AS_FRAUD
        return None

    def signature(self) -> str:
        """Stable canonical signature of conditions for conflict detection.

        Two rules with identical signatures and opposite actions are conflicting.
        """
        groups = self.condition_groups or [self.conditions]
        norm_groups = []
        for group in groups:
            norm = sorted(
                (
                    c.field.value,
                    c.operator.value,
                    str(c.value),
                    bool(c.negated),
                )
                for c in group
            )
            norm_groups.append(norm)
        norm_groups.sort()
        return repr(norm_groups)
