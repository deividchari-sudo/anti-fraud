"""V2 tests for the rule engine: persistence, OR/NOT, audit, metrics,
simulate, conflicts, import/export, validate and PATCH endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.rule_engine import (
    ActionType,
    Condition,
    ConditionType,
    InMemoryRuleRepository,
    JSONRuleRepository,
    Operator,
    Rule,
    RuleAuditLogger,
    RuleEvaluator,
    RuleMetricsRegistry,
    RuleParser,
    detect_conflicts,
    rule_from_dict,
    rule_to_dict,
)


# ---------------------------------------------------------------------------
# Parser: OR and NOT
# ---------------------------------------------------------------------------


class TestParserV2:
    def test_or_creates_condition_groups(self):
        parser = RuleParser()
        rule = parser.parse(
            "Todo pix do app ou todo pix do web é fraude"
        )
        assert len(rule.condition_groups) == 2
        # Each group should have at least one condition
        assert all(len(g) >= 1 for g in rule.condition_groups)

    def test_no_or_keeps_legacy_path(self):
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        assert rule.condition_groups == []
        assert len(rule.conditions) >= 1

    def test_exceto_marks_condition_negated(self):
        parser = RuleParser()
        rule = parser.parse(
            "Todo pix exceto do CPF 12345678901 é fraude"
        )
        # The CPF condition should be negated
        cpf_conditions = [
            c for c in rule.conditions if c.field == ConditionType.CPF_SENDER
        ]
        assert cpf_conditions, "CPF condition should be parsed"
        assert cpf_conditions[0].negated is True

    def test_validate_returns_structured_report(self):
        parser = RuleParser()
        report = parser.validate(
            "Todo pix do CPF 12345678901 com valor superior a 1000 reais é fraude"
        )
        assert report["valid"] is True
        assert report["action"] == "mark_as_fraud"
        assert isinstance(report["groups"], list)
        # Validate must NOT consume an id slot
        assert parser.rules_count == 0


# ---------------------------------------------------------------------------
# OR / NOT evaluation semantics
# ---------------------------------------------------------------------------


class TestRuleEvaluationOrNot:
    def _make_or_rule(self) -> Rule:
        return Rule(
            id="r1",
            name="OR",
            description="",
            original_text="",
            conditions=[],
            action=ActionType.MARK_AS_FRAUD,
            condition_groups=[
                [
                    Condition(
                        field=ConditionType.CANAL,
                        operator=Operator.EQUALS,
                        value="app",
                    )
                ],
                [
                    Condition(
                        field=ConditionType.CANAL,
                        operator=Operator.EQUALS,
                        value="web",
                    )
                ],
            ],
        )

    def test_or_first_group_matches(self):
        rule = self._make_or_rule()
        assert rule.evaluate({"canal": "app"}) is True

    def test_or_second_group_matches(self):
        rule = self._make_or_rule()
        assert rule.evaluate({"canal": "web"}) is True

    def test_or_no_group_matches(self):
        rule = self._make_or_rule()
        assert rule.evaluate({"canal": "api"}) is False

    def test_negated_condition_inverts_result(self):
        rule = Rule(
            id="r2",
            name="NOT",
            description="",
            original_text="",
            conditions=[
                Condition(
                    field=ConditionType.CANAL,
                    operator=Operator.EQUALS,
                    value="app",
                    negated=True,
                )
            ],
            action=ActionType.MARK_AS_FRAUD,
        )
        assert rule.evaluate({"canal": "web"}) is True
        assert rule.evaluate({"canal": "app"}) is False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestRulePersistence:
    def test_inmemory_round_trip(self):
        repo = InMemoryRuleRepository()
        evaluator = RuleEvaluator(repository=repo)
        parser = RuleParser()
        evaluator.add_rule(parser.parse("Todo pix do CPF 12345678901 é fraude"))

        # Fresh evaluator instance loads from same repo
        ev2 = RuleEvaluator(repository=repo)
        assert len(ev2.get_all_rules()) == 1
        assert ev2.get_all_rules()[0].original_text.startswith("Todo pix")

    def test_json_repository_round_trip(self, tmp_path: Path):
        repo = JSONRuleRepository(str(tmp_path / "rules.json"))
        evaluator = RuleEvaluator(repository=repo)
        parser = RuleParser()
        evaluator.add_rule(parser.parse("Todo pix do app é fraude"))
        evaluator.add_rule(parser.parse("Todo pix do CPF 99999999999 não é fraude"))

        ev2 = RuleEvaluator(repository=JSONRuleRepository(str(tmp_path / "rules.json")))
        assert len(ev2.get_all_rules()) == 2

    def test_rule_dict_round_trip_preserves_groups_and_negation(self):
        rule = Rule(
            id="x",
            name="x",
            description="",
            original_text="",
            conditions=[],
            action=ActionType.MARK_AS_LEGITIMATE,
            condition_groups=[
                [
                    Condition(
                        field=ConditionType.CANAL,
                        operator=Operator.EQUALS,
                        value="app",
                        negated=True,
                    )
                ]
            ],
        )
        roundtripped = rule_from_dict(rule_to_dict(rule))
        assert roundtripped.action == ActionType.MARK_AS_LEGITIMATE
        assert roundtripped.condition_groups[0][0].negated is True


# ---------------------------------------------------------------------------
# Metrics & audit
# ---------------------------------------------------------------------------


class TestMetricsAndAudit:
    def test_metrics_registered_on_match(self):
        evaluator = RuleEvaluator()
        evaluator.add_rule(
            RuleParser().parse("Todo pix do CPF 12345678901 é fraude")
        )
        rule_id = evaluator.get_all_rules()[0].id
        evaluator.evaluate(
            {"id": "tx1", "sender": {"cpfSender": "12345678901"}}
        )
        m = evaluator.metrics.get(rule_id)
        assert m.hit_count == 1
        assert m.last_match_at is not None

    def test_metrics_not_changed_on_simulate(self):
        evaluator = RuleEvaluator()
        evaluator.add_rule(
            RuleParser().parse("Todo pix do CPF 12345678901 é fraude")
        )
        rule_id = evaluator.get_all_rules()[0].id
        evaluator.simulate(
            [{"id": "tx1", "sender": {"cpfSender": "12345678901"}}]
        )
        assert evaluator.metrics.get(rule_id).hit_count == 0

    def test_audit_log_appends_jsonl(self, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        audit = RuleAuditLogger(str(log_path))
        evaluator = RuleEvaluator(audit_logger=audit)
        evaluator.add_rule(
            RuleParser().parse("Todo pix do CPF 12345678901 é fraude")
        )
        evaluator.evaluate(
            {"id": "tx-99", "sender": {"cpfSender": "12345678901"}}
        )
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["transaction_id"] == "tx-99"
        assert record["action"] == "mark_as_fraud"
        assert record["is_fraud"] is True


# ---------------------------------------------------------------------------
# Simulate, conflicts
# ---------------------------------------------------------------------------


class TestSimulateAndConflicts:
    def test_simulate_aggregates_correctly(self):
        evaluator = RuleEvaluator()
        evaluator.add_rule(
            RuleParser().parse("Todo pix do CPF 12345678901 é fraude")
        )
        result = evaluator.simulate(
            [
                {"sender": {"cpfSender": "12345678901"}},
                {"sender": {"cpfSender": "00000000000"}},
                {"sender": {"cpfSender": "12345678901"}},
            ]
        )
        assert result["total_transactions"] == 3
        assert result["flagged_fraud"] == 2
        assert result["flagged_legitimate"] == 0
        assert result["untouched"] == 1

    def test_detect_conflicts_finds_blacklist_vs_whitelist(self):
        a = RuleParser().parse("Todo pix do CPF 12345678901 é fraude")
        b = RuleParser().parse("Todo pix do CPF 12345678901 não é fraude")
        conflicts = detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert {conflicts[0]["rule_a"]["action"], conflicts[0]["rule_b"]["action"]} == {
            "mark_as_fraud",
            "mark_as_legitimate",
        }

    def test_detect_conflicts_ignores_disjoint_rules(self):
        a = RuleParser().parse("Todo pix do CPF 12345678901 é fraude")
        b = RuleParser().parse("Todo pix do CPF 99999999999 não é fraude")
        assert detect_conflicts([a, b]) == []


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect persistence and audit to a temp dir to keep tests isolated.
    rules_path = tmp_path / "rules.json"
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.chdir(tmp_path)

    # Re-import the module fresh so its module-level globals point at tmp.
    import importlib
    import src.main as main_module

    importlib.reload(main_module)
    main_module.rule_repository = JSONRuleRepository(str(rules_path))
    main_module.rule_audit_logger = RuleAuditLogger(str(audit_path))
    main_module.rule_evaluator = RuleEvaluator(
        repository=main_module.rule_repository,
        audit_logger=main_module.rule_audit_logger,
    )
    main_module.rule_parser = RuleParser()
    return TestClient(main_module.app)


class TestRuleAPIV2:
    def test_validate_endpoint(self, client):
        r = client.post(
            "/rules/validate",
            json={"rule_text": "Todo pix do CPF 12345678901 é fraude"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is True
        assert body["action"] == "mark_as_fraud"

    def test_simulate_endpoint(self, client):
        r = client.post(
            "/rules/simulate",
            json={
                "rule_texts": ["Todo pix do CPF 12345678901 é fraude"],
                "transactions": [
                    {"sender": {"cpfSender": "12345678901"}},
                    {"sender": {"cpfSender": "11111111111"}},
                ],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["flagged_fraud"] == 1
        assert body["total_transactions"] == 2

    def test_export_then_import(self, client):
        client.post(
            "/rules",
            json={"rule_text": "Todo pix do CPF 12345678901 é fraude"},
        )
        exp = client.get("/rules/export").json()
        assert len(exp["rules"]) == 1

        # Import into a fresh state via replace=True
        r = client.post(
            "/rules/import",
            json={"rules": exp["rules"], "replace": True},
        )
        assert r.status_code == 200
        assert r.json()["imported"] == 1

    def test_conflicts_endpoint(self, client):
        client.post(
            "/rules",
            json={"rule_text": "Todo pix do CPF 12345678901 é fraude"},
        )
        client.post(
            "/rules",
            json={"rule_text": "Todo pix do CPF 12345678901 não é fraude"},
        )
        body = client.get("/rules/conflicts").json()
        assert body["total_conflicts"] == 1

    def test_metrics_endpoint(self, client):
        client.post(
            "/rules",
            json={"rule_text": "Todo pix do CPF 12345678901 é fraude"},
        )
        client.post(
            "/rules/evaluate",
            json={"id": "tx1", "sender": {"cpfSender": "12345678901"}},
        )
        body = client.get("/rules/metrics").json()
        # At least one rule has hit_count >= 1
        assert any(v["hit_count"] >= 1 for v in body.values())

    def test_patch_endpoint_updates_priority_and_enabled(self, client):
        created = client.post(
            "/rules",
            json={"rule_text": "Todo pix do CPF 12345678901 é fraude"},
        ).json()
        rule_id = created["rule_id"]
        r = client.patch(
            f"/rules/{rule_id}",
            json={"priority": 99, "enabled": False},
        )
        assert r.status_code == 200
        assert r.json()["priority"] == 99
        assert r.json()["enabled"] is False

    def test_get_rule_still_works_after_static_routes(self, client):
        created = client.post(
            "/rules",
            json={"rule_text": "Todo pix do CPF 12345678901 é fraude"},
        ).json()
        r = client.get(f"/rules/{created['rule_id']}")
        assert r.status_code == 200
        assert r.json()["rule_id"] == created["rule_id"]


# ---------------------------------------------------------------------------
# Metrics registry sanity
# ---------------------------------------------------------------------------


def test_metrics_registry_reset():
    reg = RuleMetricsRegistry()
    reg.record_hit("r1")
    reg.record_hit("r1")
    assert reg.get("r1").hit_count == 2
    reg.reset("r1")
    assert reg.get("r1").hit_count == 0
