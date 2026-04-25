"""
Rule evaluator for fraud detection.
"""

from typing import Any, Dict, List, Optional

from .rule import Rule


class RuleEvaluator:
    """Evaluate rules against transactions."""

    def __init__(self):
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the evaluator."""
        self.rules.append(rule)

    def add_rules(self, rules: List[Rule]) -> None:
        """Add multiple rules to the evaluator."""
        self.rules.extend(rules)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                del self.rules[i]
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

    def evaluate(self, transaction: dict) -> Dict[str, Any]:
        """Evaluate transaction against all rules."""
        results = {
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

        return results

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
