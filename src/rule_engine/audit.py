"""BACEN-compliant audit logger for rule matches.

Each match is appended as a single JSON object on its own line (JSON Lines)
to allow streaming downstream tools (ELK, Splunk, BigQuery) to ingest the
log file without locking. The audit log is append-only by design.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class RuleAuditLogger:
    """Append-only JSONL logger for rule matches."""

    def __init__(self, log_path: str = "logs/rule_audit.jsonl") -> None:
        self.log_path = Path(log_path)

    def log_match(
        self,
        rule_id: str,
        rule_name: str,
        transaction_id: str,
        action: str,
        is_fraud: bool,
        conditions_matched: int,
        priority: int = 0,
        extra: Optional[dict] = None,
    ) -> None:
        """Persist a single match record."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rule_id": rule_id,
            "rule_name": rule_name,
            "transaction_id": transaction_id,
            "action": action,
            "is_fraud": is_fraud,
            "conditions_matched": conditions_matched,
            "priority": priority,
        }
        if extra:
            record["extra"] = extra

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
