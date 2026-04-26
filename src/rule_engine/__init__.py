"""Rule Engine for natural language fraud detection rules.

V2 exports add: persistence (RuleRepository, JSONRuleRepository,
InMemoryRuleRepository), audit logging (RuleAuditLogger), per-rule metrics
(RuleMetricsRegistry, RuleMetric) and conflict detection (detect_conflicts).
"""

from .audit import RuleAuditLogger
from .conflicts import detect_conflicts
from .evaluator import RuleEvaluator
from .metrics import RuleMetric, RuleMetricsRegistry
from .parser import RuleParser
from .repository import (
    InMemoryRuleRepository,
    JSONRuleRepository,
    RuleRepository,
    rule_from_dict,
    rule_to_dict,
)
from .rule import ActionType, Condition, ConditionType, Operator, Rule

__all__ = [
    "RuleParser",
    "Rule",
    "Condition",
    "ActionType",
    "ConditionType",
    "Operator",
    "RuleEvaluator",
    "RuleRepository",
    "JSONRuleRepository",
    "InMemoryRuleRepository",
    "rule_to_dict",
    "rule_from_dict",
    "RuleAuditLogger",
    "RuleMetricsRegistry",
    "RuleMetric",
    "detect_conflicts",
]
