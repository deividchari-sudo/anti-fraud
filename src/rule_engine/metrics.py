"""Per-rule operational metrics registry.

Tracks hit_count, last_match_at, and false_positive_count for each rule.
Stored in-memory but exposed via the API for observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class RuleMetric:
    hit_count: int = 0
    last_match_at: Optional[str] = None
    false_positive_count: int = 0
    last_false_positive_at: Optional[str] = None


@dataclass
class RuleMetricsRegistry:
    """Lightweight registry indexed by rule_id."""

    metrics: Dict[str, RuleMetric] = field(default_factory=dict)

    def record_hit(self, rule_id: str) -> None:
        m = self.metrics.setdefault(rule_id, RuleMetric())
        m.hit_count += 1
        m.last_match_at = datetime.now(timezone.utc).isoformat()

    def record_false_positive(self, rule_id: str) -> None:
        m = self.metrics.setdefault(rule_id, RuleMetric())
        m.false_positive_count += 1
        m.last_false_positive_at = datetime.now(timezone.utc).isoformat()

    def get(self, rule_id: str) -> RuleMetric:
        return self.metrics.get(rule_id, RuleMetric())

    def reset(self, rule_id: Optional[str] = None) -> None:
        if rule_id is None:
            self.metrics.clear()
        elif rule_id in self.metrics:
            del self.metrics[rule_id]

    def to_dict(self) -> Dict[str, dict]:
        return {
            rid: {
                "hit_count": m.hit_count,
                "last_match_at": m.last_match_at,
                "false_positive_count": m.false_positive_count,
                "last_false_positive_at": m.last_false_positive_at,
            }
            for rid, m in self.metrics.items()
        }
