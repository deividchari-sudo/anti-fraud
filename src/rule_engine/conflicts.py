"""Static analysis to detect conflicting rule pairs.

A pair ``(A, B)`` is considered conflicting when both rules have the same
condition signature but opposite actions (one ``MARK_AS_FRAUD`` and the other
``MARK_AS_LEGITIMATE``). Operators that override transactions in the same
input would fire both at evaluation time, so the operator must be alerted to
adjust priorities or remove duplicates.
"""

from __future__ import annotations

from typing import List

from .rule import ActionType, Rule


def detect_conflicts(rules: List[Rule]) -> List[dict]:
    """Return a list of conflict descriptors among the given rules."""
    conflicts: List[dict] = []
    n = len(rules)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rules[i], rules[j]
            if a.action == b.action:
                continue
            if a.signature() != b.signature():
                continue
            conflicts.append(
                {
                    "rule_a": {
                        "id": a.id,
                        "name": a.name,
                        "action": a.action.value,
                        "priority": a.priority,
                    },
                    "rule_b": {
                        "id": b.id,
                        "name": b.name,
                        "action": b.action.value,
                        "priority": b.priority,
                    },
                    "reason": "Mesma assinatura de condições com ações opostas",
                    "resolution": (
                        "Ajuste a prioridade da regra de whitelist para um "
                        "valor maior, ou remova uma das regras."
                    ),
                }
            )
    return conflicts
