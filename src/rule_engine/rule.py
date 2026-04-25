"""
Rule data structures for fraud detection.
"""
from enum import Enum
from typing import List, Optional, Any
from dataclasses import dataclass


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
    """Single condition in a rule."""
    field: ConditionType
    operator: Operator
    value: Any
    
    def evaluate(self, transaction: dict) -> bool:
        """Evaluate condition against transaction."""
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
            return transaction.get('canal', '').lower()
        elif self.field == ConditionType.VALOR:
            return float(transaction.get('valor', 0))
        elif self.field == ConditionType.HORARIO:
            # Extract hour from timestamp
            timestamp = transaction.get('timestamp', '')
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.hour
            except:
                return 0
        elif self.field == ConditionType.CPF_SENDER:
            return transaction.get('sender', {}).get('cpfSender', '')
        elif self.field == ConditionType.CPF_RECEIVER:
            return transaction.get('receiver', {}).get('cpfReceiver', '')
        elif self.field == ConditionType.BANCO_SENDER:
            return transaction.get('sender', {}).get('banco', 0)
        elif self.field == ConditionType.BANCO_RECEIVER:
            return transaction.get('receiver', {}).get('banco', 0)
        
        return None


@dataclass
class Rule:
    """Fraud detection rule."""
    id: str
    name: str
    description: str
    original_text: str
    conditions: List[Condition]
    action: ActionType
    enabled: bool = True
    priority: int = 0
    
    def evaluate(self, transaction: dict) -> bool:
        """Evaluate all conditions against transaction."""
        if not self.enabled:
            return False
        
        # All conditions must be true (AND logic)
        return all(condition.evaluate(transaction) for condition in self.conditions)
    
    def matches(self, transaction: dict) -> Optional[bool]:
        """Check if rule matches and return action result."""
        if self.evaluate(transaction):
            return self.action == ActionType.MARK_AS_FRAUD
        return None
