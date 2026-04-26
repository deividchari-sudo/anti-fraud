"""Rule evaluator for fraud detection.

V2 features:
    - Optional persistence via :class:`RuleRepository` (DI).
    - Per-rule metrics via :class:`RuleMetricsRegistry`.
    - BACEN audit log via :class:`RuleAuditLogger`.
    - Batch simulation (dry-run) for offline rule testing.
"""

from typing import Any, Dict, List, Optional

from .audit import RuleAuditLogger
from .metrics import RuleMetricsRegistry
from .repository import RuleRepository
from .rule import Rule


class RuleEvaluator:
    """Evaluate rules against transactions.

    Args:
        repository: Optional repository to persist rules across restarts.
            When provided, rules are loaded on init and saved on every
            mutating operation.
        metrics: Optional metrics registry. A new in-memory registry is
            created if not provided.
        audit_logger: Optional audit logger; when provided every match is
            persisted as a JSONL record for BACEN traceability.
    """

    def __init__(
        self,
        repository: Optional[RuleRepository] = None,
        metrics: Optional[RuleMetricsRegistry] = None,
        audit_logger: Optional[RuleAuditLogger] = None,
    ):
        self.rules: List[Rule] = []
        self._repository = repository
        self.metrics = metrics or RuleMetricsRegistry()
        self._audit_logger = audit_logger

        if repository is not None:
            self.rules = repository.load_all()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        if self._repository is not None:
            self._repository.save_all(self.rules)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the evaluator."""
        self.rules.append(rule)
        self._persist()

    def add_rules(self, rules: List[Rule]) -> None:
        """Add multiple rules to the evaluator."""
        self.rules.extend(rules)
        self._persist()

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                del self.rules[i]
                self._persist()
                return True
        return False

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def get_all_rules(self) -> List[Rule]:
        """Get all rules."""
        return self.rules

    def update_rule(self, rule_id: str, **changes: Any) -> Optional[Rule]:
        """Patch fields of an existing rule.

        Supported keys: ``enabled``, ``priority``, ``name``, ``description``.
        """
        rule = self.get_rule(rule_id)
        if rule is None:
            return None
        for key, value in changes.items():
            if hasattr(rule, key) and key in {
                "enabled",
                "priority",
                "name",
                "description",
            }:
                setattr(rule, key, value)
        self._persist()
        return rule

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, transaction: dict) -> Dict[str, Any]:
        """Evaluate transaction against all rules and record metrics/audit."""
        results: Dict[str, Any] = {
            "transaction_id": transaction.get("id", "unknown"),
            "matched_rules": [],
            "is_fraud_by_rules": False,
            "is_legitimate_by_rules": False,
            "rule_count": len(self.rules),
        }

        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if not rule.enabled:
                continue

            match_result = rule.matches(transaction)
            if match_result is not None:
                conditions_count = sum(
                    len(g) for g in (rule.condition_groups or [rule.conditions])
                )
                results["matched_rules"].append(
                    {
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "action": rule.action.value,
                        "is_fraud": match_result,
                    }
                )

                if match_result:
                    results["is_fraud_by_rules"] = True
                else:
                    results["is_legitimate_by_rules"] = True

                self.metrics.record_hit(rule.id)
                if self._audit_logger is not None:
                    self._audit_logger.log_match(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        transaction_id=str(transaction.get("id", "unknown")),
                        action=rule.action.value,
                        is_fraud=bool(match_result),
                        conditions_matched=conditions_count,
                        priority=rule.priority,
                    )

        return results

    # ------------------------------------------------------------------
    # Batch simulation (dry-run)
    # ------------------------------------------------------------------
    def simulate(
        self,
        transactions: List[dict],
        rules: Optional[List[Rule]] = None,
    ) -> Dict[str, Any]:
        """Run a stateless dry-run of ``rules`` against ``transactions``.

        Does not mutate metrics nor write to the audit log. Returns aggregate
        statistics and per-rule hit counts useful for offline validation.
        """
        rule_set = rules if rules is not None else self.rules
        per_rule: Dict[str, int] = {}
        flagged_fraud = 0
        flagged_legit = 0

        sorted_rules = sorted(rule_set, key=lambda r: r.priority, reverse=True)
        for tx in transactions:
            tx_flagged = False
            for rule in sorted_rules:
                if not rule.enabled:
                    continue
                match_result = rule.matches(tx)
                if match_result is None:
                    continue
                per_rule[rule.id] = per_rule.get(rule.id, 0) + 1
                if not tx_flagged:
                    if match_result:
                        flagged_fraud += 1
                    else:
                        flagged_legit += 1
                    tx_flagged = True
        return {
            "total_transactions": len(transactions),
            "flagged_fraud": flagged_fraud,
            "flagged_legitimate": flagged_legit,
            "untouched": len(transactions) - flagged_fraud - flagged_legit,
            "hits_by_rule": per_rule,
            "evaluated_rules": len(rule_set),
        }

    def evaluate_single(self, transaction: dict) -> Optional[bool]:
        """Evaluate transaction and return final decision (fraud or not)."""
        results = self.evaluate(transaction)

        # If any rule marks as fraud, it's fraud (unless a higher priority rule marks as legitimate)
        # Sort matched rules by priority
        matched_rules_sorted = sorted(
            results["matched_rules"],
            key=lambda r: (
                self.get_rule(r["rule_id"]).priority
                if self.get_rule(r["rule_id"])
                else 0
            ),
            reverse=True,
        )

        if matched_rules_sorted:
            # Return the action of the highest priority matched rule
            return matched_rules_sorted[0]["is_fraud"]

        return None  # No rules matched
