"""
Tests for the rule engine - natural language rule parser and evaluator.
"""
import pytest
from src.rule_engine import RuleParser, RuleEvaluator, Rule, Condition, ActionType, ConditionType, Operator


class TestRuleParser:
    """Test natural language rule parser."""
    
    def test_parse_simple_cpf_rule(self):
        """Test parsing simple CPF rule."""
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field == ConditionType.CPF_SENDER
        assert rule.conditions[0].operator == Operator.EQUALS
        assert rule.conditions[0].value == "12345678901"
        assert rule.action == ActionType.MARK_AS_FRAUD
    
    def test_parse_cpf_not_fraud_rule(self):
        """Test parsing CPF rule with negative action."""
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 nao é fraude")
        
        assert rule.action == ActionType.MARK_AS_LEGITIMATE
    
    def test_parse_valor_rule(self):
        """Test parsing valor rule."""
        parser = RuleParser()
        rule = parser.parse("Todos os pix com valor superior a 1000 reais é fraude")
        
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field == ConditionType.VALOR
        assert rule.conditions[0].operator == Operator.GREATER_THAN
        assert rule.conditions[0].value == 1000.0
    
    def test_parse_horario_rule(self):
        """Test parsing horario rule."""
        parser = RuleParser()
        rule = parser.parse("Todos os pix depois das 22:00 é fraude")
        
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field == ConditionType.HORARIO
        assert rule.conditions[0].operator == Operator.AFTER
        assert rule.conditions[0].value == 22
    
    def test_parse_canal_rule(self):
        """Test parsing canal rule."""
        parser = RuleParser()
        rule = parser.parse("Todos os pix do app é fraude")
        
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field == ConditionType.CANAL
        assert rule.conditions[0].operator == Operator.EQUALS
        assert rule.conditions[0].value == "app"
    
    def test_parse_combined_rule(self):
        """Test parsing combined rule with multiple conditions."""
        parser = RuleParser()
        rule = parser.parse("Todos os pix do app com valor superior a 1000 reais depois das 22:00 é fraude")
        
        assert len(rule.conditions) == 3
        # Canal
        assert any(c.field == ConditionType.CANAL for c in rule.conditions)
        # Valor
        assert any(c.field == ConditionType.VALOR and c.operator == Operator.GREATER_THAN for c in rule.conditions)
        # Horário
        assert any(c.field == ConditionType.HORARIO and c.operator == Operator.AFTER for c in rule.conditions)
    
    def test_parse_multiple_rules(self):
        """Test parsing multiple rules."""
        parser = RuleParser()
        rules = parser.parse_multiple([
            "Todo pix do CPF 12345678901 é fraude",
            "Todos os pix com valor superior a 5000 reais é fraude"
        ])
        
        assert len(rules) == 2
        assert rules[0].conditions[0].value == "12345678901"
        assert rules[1].conditions[0].value == 5000.0
    
    def test_parse_menor_que_valor(self):
        """Test parsing menor que valor."""
        parser = RuleParser()
        rule = parser.parse("Todos os pix com valor menor que 100 reais é fraude")
        
        assert rule.conditions[0].operator == Operator.LESS_THAN
        assert rule.conditions[0].value == 100.0
    
    def test_parse_antes_das_horario(self):
        """Test parsing antes das horario."""
        parser = RuleParser()
        rule = parser.parse("Todos os pix antes das 06:00 é fraude")
        
        assert rule.conditions[0].operator == Operator.BEFORE
        assert rule.conditions[0].value == 6
    
    def test_parse_pix_produto(self):
        """Test parsing pix produto."""
        parser = RuleParser()
        rule = parser.parse("Todo pix é fraude")
        
        assert rule.conditions[0].field == ConditionType.CANAL
        assert rule.conditions[0].operator == Operator.CONTAINS
        assert rule.conditions[0].value == "pix"


class TestConditionEvaluation:
    """Test condition evaluation."""
    
    def test_cpf_condition_equals(self):
        """Test CPF equals condition."""
        condition = Condition(
            field=ConditionType.CPF_SENDER,
            operator=Operator.EQUALS,
            value="12345678901"
        )
        
        transaction = {
            'sender': {'cpfSender': '12345678901'}
        }
        
        assert condition.evaluate(transaction) == True
    
    def test_cpf_condition_not_equals(self):
        """Test CPF not equals condition."""
        condition = Condition(
            field=ConditionType.CPF_SENDER,
            operator=Operator.NOT_EQUALS,
            value="12345678901"
        )
        
        transaction = {
            'sender': {'cpfSender': '98765432100'}
        }
        
        assert condition.evaluate(transaction) == True
    
    def test_valor_greater_than(self):
        """Test valor greater than condition."""
        condition = Condition(
            field=ConditionType.VALOR,
            operator=Operator.GREATER_THAN,
            value=1000.0
        )
        
        transaction = {'valor': 1500.0}
        assert condition.evaluate(transaction) == True
        
        transaction = {'valor': 500.0}
        assert condition.evaluate(transaction) == False
    
    def test_valor_less_than(self):
        """Test valor less than condition."""
        condition = Condition(
            field=ConditionType.VALOR,
            operator=Operator.LESS_THAN,
            value=1000.0
        )
        
        transaction = {'valor': 500.0}
        assert condition.evaluate(transaction) == True
        
        transaction = {'valor': 1500.0}
        assert condition.evaluate(transaction) == False
    
    def test_horario_after(self):
        """Test horario after condition."""
        condition = Condition(
            field=ConditionType.HORARIO,
            operator=Operator.AFTER,
            value=22
        )
        
        transaction = {'timestamp': '2026-04-25T23:00:00'}
        assert condition.evaluate(transaction) == True
        
        transaction = {'timestamp': '2026-04-25T21:00:00'}
        assert condition.evaluate(transaction) == False
    
    def test_horario_before(self):
        """Test horario before condition."""
        condition = Condition(
            field=ConditionType.HORARIO,
            operator=Operator.BEFORE,
            value=6
        )
        
        transaction = {'timestamp': '2026-04-25T05:00:00'}
        assert condition.evaluate(transaction) == True
        
        transaction = {'timestamp': '2026-04-25T07:00:00'}
        assert condition.evaluate(transaction) == False
    
    def test_canal_equals(self):
        """Test canal equals condition."""
        condition = Condition(
            field=ConditionType.CANAL,
            operator=Operator.EQUALS,
            value="app"
        )
        
        transaction = {'canal': 'app'}
        assert condition.evaluate(transaction) == True
        
        transaction = {'canal': 'web'}
        assert condition.evaluate(transaction) == False


class TestRuleEvaluation:
    """Test rule evaluation."""
    
    def test_single_condition_rule(self):
        """Test rule with single condition."""
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        
        assert rule.matches(transaction) == True
        
        transaction['sender']['cpfSender'] = '98765432100'
        assert rule.matches(transaction) == None
    
    def test_multiple_conditions_rule(self):
        """Test rule with multiple conditions (AND logic)."""
        parser = RuleParser()
        rule = parser.parse("Todos os pix do app com valor superior a 1000 reais depois das 22:00 é fraude")
        
        # All conditions met
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 1500.0,
            'timestamp': '2026-04-25T23:00:00'
        }
        assert rule.matches(transaction) == True
        
        # Canal not met
        transaction['canal'] = 'web'
        assert rule.matches(transaction) == None
        
        # Valor not met
        transaction['canal'] = 'app'
        transaction['valor'] = 500.0
        assert rule.matches(transaction) == None
        
        # Horario not met
        transaction['valor'] = 1500.0
        transaction['timestamp'] = '2026-04-25T21:00:00'
        assert rule.matches(transaction) == None
    
    def test_legitimate_rule(self):
        """Test rule that marks as legitimate."""
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 nao é fraude")
        
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        
        assert rule.matches(transaction) == False


class TestRuleEvaluator:
    """Test rule evaluator."""
    
    def test_add_and_evaluate_single_rule(self):
        """Test adding and evaluating single rule."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        
        evaluator.add_rule(rule)
        
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        
        result = evaluator.evaluate(transaction)
        assert result['is_fraud_by_rules'] == True
        assert len(result['matched_rules']) == 1
    
    def test_add_multiple_rules(self):
        """Test adding multiple rules."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        
        rule1 = parser.parse("Todo pix do CPF 12345678901 é fraude")
        rule2 = parser.parse("Todos os pix com valor superior a 5000 reais é fraude")
        
        evaluator.add_rules([rule1, rule2])
        
        assert len(evaluator.get_all_rules()) == 2
    
    def test_evaluate_no_match(self):
        """Test evaluation when no rules match."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        
        evaluator.add_rule(rule)
        
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '98765432100'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        
        result = evaluator.evaluate(transaction)
        assert result['is_fraud_by_rules'] == False
        assert len(result['matched_rules']) == 0
    
    def test_evaluate_single(self):
        """Test evaluate_single method."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        
        evaluator.add_rule(rule)
        
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        
        assert evaluator.evaluate_single(transaction) == True
        
        transaction['sender']['cpfSender'] = '98765432100'
        assert evaluator.evaluate_single(transaction) is None
    
    def test_rule_priority(self):
        """Test rule priority evaluation."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        
        rule1 = parser.parse("Todo pix do CPF 12345678901 é fraude")
        rule1.priority = 1
        
        rule2 = parser.parse("Todo pix do CPF 12345678901 nao é fraude")
        rule2.priority = 10
        
        evaluator.add_rules([rule1, rule2])
        
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        
        # Higher priority rule (rule2) should win
        assert evaluator.evaluate_single(transaction) == False
    
    def test_disabled_rule(self):
        """Test that disabled rules are not evaluated."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        rule.enabled = False
        
        evaluator.add_rule(rule)
        
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        
        result = evaluator.evaluate(transaction)
        assert result['is_fraud_by_rules'] == False
        assert len(result['matched_rules']) == 0
    
    def test_remove_rule(self):
        """Test removing a rule."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        
        evaluator.add_rule(rule)
        assert len(evaluator.get_all_rules()) == 1
        
        assert evaluator.remove_rule(rule.id) == True
        assert len(evaluator.get_all_rules()) == 0
        
        assert evaluator.remove_rule(rule.id) == False
    
    def test_get_rule(self):
        """Test getting a specific rule."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        rule = parser.parse("Todo pix do CPF 12345678901 é fraude")
        
        evaluator.add_rule(rule)
        
        retrieved_rule = evaluator.get_rule(rule.id)
        assert retrieved_rule is not None
        assert retrieved_rule.id == rule.id
        
        assert evaluator.get_rule('non_existent') is None


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_whitelist_blacklist_scenario(self):
        """Test whitelist vs blacklist scenario."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        
        # Blacklist: CPF 12345678901 is always fraud
        rule1 = parser.parse("Todo pix do CPF 12345678901 é fraude")
        rule1.priority = 1
        
        # Whitelist: CPF 98765432100 is never fraud (higher priority)
        rule2 = parser.parse("Todo pix do CPF 98765432100 nao é fraude")
        rule2.priority = 10
        
        evaluator.add_rules([rule1, rule2])
        
        # Blacklisted CPF
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        assert evaluator.evaluate_single(transaction) == True
        
        # Whitelisted CPF (should override blacklist)
        transaction['sender']['cpfSender'] = '98765432100'
        assert evaluator.evaluate_single(transaction) == False
    
    def test_high_value_nighttime_scenario(self):
        """Test high value nighttime scenario."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        
        rule = parser.parse("Todos os pix do app com valor superior a 1000 reais depois das 22:00 é fraude")
        evaluator.add_rule(rule)
        
        # Should match: app, >1000, after 22:00
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 1500.0,
            'timestamp': '2026-04-25T23:00:00'
        }
        assert evaluator.evaluate_single(transaction) == True
        
        # Should not match: wrong canal
        transaction['canal'] = 'web'
        assert evaluator.evaluate_single(transaction) is None
        
        # Should not match: value too low
        transaction['canal'] = 'app'
        transaction['valor'] = 500.0
        assert evaluator.evaluate_single(transaction) is None
        
        # Should not match: wrong time
        transaction['valor'] = 1500.0
        transaction['timestamp'] = '2026-04-25T21:00:00'
        assert evaluator.evaluate_single(transaction) is None
    
    def test_multiple_rules_same_transaction(self):
        """Test multiple rules matching same transaction."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        
        rule1 = parser.parse("Todo pix do CPF 12345678901 é fraude")
        rule2 = parser.parse("Todos os pix com valor superior a 5000 reais é fraude")
        
        evaluator.add_rules([rule1, rule2])
        
        # Matches both rules
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 6000.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        
        result = evaluator.evaluate(transaction)
        assert result['is_fraud_by_rules'] == True
        assert len(result['matched_rules']) == 2
    
    def test_early_morning_scenario(self):
        """Test early morning scenario."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        
        rule = parser.parse("Todos os pix antes das 06:00 é fraude")
        evaluator.add_rule(rule)
        
        # Should match: before 06:00
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T05:00:00'
        }
        assert evaluator.evaluate_single(transaction) == True
        
        # Should not match: after 06:00
        transaction['timestamp'] = '2026-04-25T07:00:00'
        assert evaluator.evaluate_single(transaction) is None
    
    def test_low_value_scenario(self):
        """Test low value scenario."""
        evaluator = RuleEvaluator()
        parser = RuleParser()
        
        rule = parser.parse("Todos os pix com valor menor que 100 reais é fraude")
        evaluator.add_rule(rule)
        
        # Should match: < 100
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 50.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        assert evaluator.evaluate_single(transaction) == True
        
        # Should not match: >= 100
        transaction['valor'] = 150.0
        assert evaluator.evaluate_single(transaction) is None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_rule_text(self):
        """Test parsing empty rule text."""
        parser = RuleParser()
        rule = parser.parse("")
        
        assert len(rule.conditions) == 0
        assert rule.action == ActionType.MARK_AS_FRAUD  # Default
    
    def test_invalid_timestamp(self):
        """Test condition evaluation with invalid timestamp."""
        condition = Condition(
            field=ConditionType.HORARIO,
            operator=Operator.AFTER,
            value=22
        )
        
        transaction = {'timestamp': 'invalid-timestamp'}
        # Should not crash, return False
        assert condition.evaluate(transaction) == False
    
    def test_missing_field(self):
        """Test condition evaluation with missing field."""
        condition = Condition(
            field=ConditionType.CPF_SENDER,
            operator=Operator.EQUALS,
            value="12345678901"
        )
        
        transaction = {}  # No sender field
        # Should not crash, return False
        assert condition.evaluate(transaction) == False
    
    def test_case_insensitive_canal(self):
        """Test that canal matching is case-insensitive."""
        condition = Condition(
            field=ConditionType.CANAL,
            operator=Operator.EQUALS,
            value="app"
        )
        
        transaction = {'canal': 'APP'}
        assert condition.evaluate(transaction) == True
    
    def test_decimal_valor(self):
        """Test parsing decimal valor."""
        parser = RuleParser()
        rule = parser.parse("Todos os pix com valor superior a 1000.50 reais é fraude")
        
        assert rule.conditions[0].value == 1000.5
