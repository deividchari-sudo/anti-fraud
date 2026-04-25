"""
Comprehensive test suite for rule engine with 100 test cases.
Covers all possible scenarios with rules approving and rejecting.
"""
import pytest
from src.rule_engine import RuleParser, RuleEvaluator, ConditionType, Operator, ActionType


class TestComprehensiveRuleEngine:
    """100 comprehensive test cases for rule engine."""
    
    @pytest.fixture
    def parser(self):
        return RuleParser()
    
    @pytest.fixture
    def evaluator(self):
        return RuleEvaluator()
    
    # CPF Tests (10 tests)
    @pytest.mark.parametrize("rule_text,cpf,expected_match,is_fraud", [
        ("Todo pix do CPF 12345678901 é fraude", "12345678901", True, True),
        ("Todo pix do CPF 12345678901 nao é fraude", "12345678901", True, False),
        ("Todo pix do CPF 1234567890 é fraude", "1234567890", True, True),
        ("Todo pix do CPF 98765432101 é fraude", "98765432101", True, True),
        ("Todo pix do CPF 98765432100 nao é fraude", "98765432100", True, False),
        ("Todo pix do CPF 11111111111 é fraude", "11111111111", True, True),
        ("Todo pix do CPF 22222222222 nao é fraude", "22222222222", True, False),
        ("Todo pix do CPF 33333333333 é fraude", "33333333333", True, True),
        ("Todo pix do CPF 44444444444 nao é fraude", "44444444444", True, False),
        ("Todo pix do CPF 55555555555 é fraude", "55555555555", True, True),
    ])
    def test_cpf_rules(self, parser, rule_text, cpf, expected_match, is_fraud):
        """Test CPF-based rules with various CPFs."""
        rule = parser.parse(rule_text)
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': cpf},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        result = rule.matches(transaction)
        if expected_match:
            assert result == is_fraud
        else:
            assert result is None
    
    # Valor Tests (10 tests)
    @pytest.mark.parametrize("rule_text,valor,expected_match,is_fraud", [
        ("Todos os pix com valor superior a 1000 reais é fraude", 1500.0, True, True),
        ("Todos os pix com valor superior a 1000 reais é fraude", 500.0, False, None),
        ("Todos os pix com valor menor que 100 reais é fraude", 50.0, True, True),
        ("Todos os pix com valor menor que 100 reais é fraude", 150.0, False, None),
        ("Todos os pix com valor superior a 5000 reais é fraude", 6000.0, True, True),
        ("Todos os pix com valor superior a 5000 reais é fraude", 4000.0, False, None),
        ("Todos os pix com valor menor que 50 reais é fraude", 30.0, True, True),
        ("Todos os pix com valor menor que 50 reais é fraude", 60.0, False, None),
        ("Todos os pix com valor superior a 10000 reais é fraude", 15000.0, True, True),
        ("Todos os pix com valor superior a 10000 reais é fraude", 9000.0, False, None),
    ])
    def test_valor_rules(self, parser, rule_text, valor, expected_match, is_fraud):
        """Test valor-based rules with various thresholds."""
        rule = parser.parse(rule_text)
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': valor,
            'timestamp': '2026-04-25T10:00:00'
        }
        result = rule.matches(transaction)
        if expected_match:
            assert result == is_fraud
        else:
            assert result is None
    
    # Horário Tests (10 tests)
    @pytest.mark.parametrize("rule_text,hour,expected_match,is_fraud", [
        ("Todos os pix depois das 22:00 é fraude", 23, True, True),
        ("Todos os pix depois das 22:00 é fraude", 21, False, None),
        ("Todos os pix antes das 06:00 é fraude", 5, True, True),
        ("Todos os pix antes das 06:00 é fraude", 7, False, None),
        ("Todos os pix depois das 00:00 é fraude", 1, True, True),
        ("Todos os pix depois das 00:00 é fraude", 0, False, None),
        ("Todos os pix antes das 12:00 é fraude", 11, True, True),
        ("Todos os pix antes das 12:00 é fraude", 13, False, None),
        ("Todos os pix depois das 18:00 é fraude", 19, True, True),
        ("Todos os pix depois das 18:00 é fraude", 17, False, None),
    ])
    def test_horario_rules(self, parser, rule_text, hour, expected_match, is_fraud):
        """Test horário-based rules with various times."""
        rule = parser.parse(rule_text)
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': f'2026-04-25T{hour:02d}:00:00'
        }
        result = rule.matches(transaction)
        if expected_match:
            assert result == is_fraud
        else:
            assert result is None
    
    # Canal Tests (10 tests)
    @pytest.mark.parametrize("rule_text,canal,expected_match,is_fraud", [
        ("Todos os pix do app é fraude", "app", True, True),
        ("Todos os pix do app é fraude", "web", False, None),
        ("Todos os pix do web é fraude", "web", True, True),
        ("Todos os pix do web é fraude", "app", False, None),
        ("Todos os pix do api é fraude", "api", True, True),
        ("Todos os pix do api é fraude", "app", False, None),
        ("Todos os pix do app nao é fraude", "app", True, False),
        ("Todos os pix do web nao é fraude", "web", True, False),
        ("Todos os pix do api nao é fraude", "api", True, False),
        ("Todos os pix do app nao é fraude", "web", False, None),
    ])
    def test_canal_rules(self, parser, rule_text, canal, expected_match, is_fraud):
        """Test canal-based rules with various channels."""
        rule = parser.parse(rule_text)
        transaction = {
            'id': 'test-1',
            'canal': canal,
            'sender': {'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        result = rule.matches(transaction)
        if expected_match:
            assert result == is_fraud
        else:
            assert result is None
    
    # Banco Tests (10 tests)
    @pytest.mark.parametrize("rule_text,banco,expected_match,is_fraud", [
        ("Bloquear todas as transacoes do banco 001", 1, True, True),
        ("Bloquear todas as transacoes do banco 001", 23, False, None),
        ("Bloquear todas as transacoes do banco 23", 23, True, True),
        ("Bloquear todas as transacoes do banco 23", 1, False, None),
        ("Bloquear todas as transacoes do banco 152", 152, True, True),
        ("Bloquear todas as transacoes do banco 152", 23, False, None),
        ("Bloquear todas as transacoes do banco 888", 888, True, True),
        ("Bloquear todas as transacoes do banco 888", 152, False, None),
        ("Bloquear todas as transacoes do banco 341", 341, True, True),
        ("Bloquear todas as transacoes do banco 341", 888, False, None),
    ])
    def test_banco_rules(self, parser, rule_text, banco, expected_match, is_fraud):
        """Test banco-based rules with various banks."""
        rule = parser.parse(rule_text)
        transaction = {
            'id': 'test-1',
            'canal': 'app',
            'sender': {'banco': banco, 'cpfSender': '12345678901'},
            'valor': 100.0,
            'timestamp': '2026-04-25T10:00:00'
        }
        result = rule.matches(transaction)
        if expected_match:
            assert result == is_fraud
        else:
            assert result is None
    
    # Combined Conditions Tests (20 tests)
    @pytest.mark.parametrize("rule_text,transaction,expected_match,is_fraud", [
        # CPF + Canal
        ("Todos os pix do app do CPF 12345678901 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        ("Todos os pix do app do CPF 12345678901 é fraude",
         {'canal': 'web', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         False, None),
        # CPF + Valor
        ("Todo pix do CPF 12345678901 com valor superior a 1000 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        ("Todo pix do CPF 12345678901 com valor superior a 1000 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 500.0, 'timestamp': '2026-04-25T10:00:00'},
         False, None),
        # CPF + Horário
        ("Todo pix do CPF 12345678901 depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T23:00:00'},
         True, True),
        ("Todo pix do CPF 12345678901 depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T21:00:00'},
         False, None),
        # Valor + Horário
        ("Todos os pix com valor superior a 1000 reais depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T23:00:00'},
         True, True),
        ("Todos os pix com valor superior a 1000 reais depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 500.0, 'timestamp': '2026-04-25T23:00:00'},
         False, None),
        # Canal + Valor
        ("Todos os pix do app com valor superior a 1000 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        ("Todos os pix do app com valor superior a 1000 reais é fraude",
         {'canal': 'web', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T10:00:00'},
         False, None),
        # Banco + Valor
        ("Bloquear todas as transacoes do banco 001 com valor superior a 5000 reais",
         {'canal': 'app', 'sender': {'banco': 1, 'cpfSender': '12345678901'}, 'valor': 6000.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        ("Bloquear todas as transacoes do banco 001 com valor superior a 5000 reais",
         {'canal': 'app', 'sender': {'banco': 1, 'cpfSender': '12345678901'}, 'valor': 4000.0, 'timestamp': '2026-04-25T10:00:00'},
         False, None),
        # CPF + Banco
        ("Todo pix do CPF 12345678901 do banco 23 é fraude",
         {'canal': 'app', 'sender': {'banco': 23, 'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        ("Todo pix do CPF 12345678901 do banco 23 é fraude",
         {'canal': 'app', 'sender': {'banco': 152, 'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         False, None),
        # Three conditions: CPF + Valor + Horário
        ("Todo pix do CPF 12345678901 com valor superior a 1000 reais depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T23:00:00'},
         True, True),
        ("Todo pix do CPF 12345678901 com valor superior a 1000 reais depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T21:00:00'},
         False, None),
        # Three conditions: Canal + Valor + Horário
        ("Todos os pix do app com valor superior a 1000 reais depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T23:00:00'},
         True, True),
        ("Todos os pix do app com valor superior a 1000 reais depois das 22:00 é fraude",
         {'canal': 'web', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T23:00:00'},
         False, None),
        # Four conditions: CPF + Canal + Valor + Horário
        ("Todos os pix do app do CPF 12345678901 com valor superior a 1000 reais depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T23:00:00'},
         True, True),
        ("Todos os pix do app do CPF 12345678901 com valor superior a 1000 reais depois das 22:00 é fraude",
         {'canal': 'web', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T23:00:00'},
         False, None),
    ])
    def test_combined_conditions(self, parser, rule_text, transaction, expected_match, is_fraud):
        """Test rules with combined conditions."""
        rule = parser.parse(rule_text)
        transaction['id'] = 'test-1'
        if 'receiver' not in transaction:
            transaction['receiver'] = {'cpfReceiver': '98765432100'}
        result = rule.matches(transaction)
        if expected_match:
            assert result == is_fraud
        else:
            assert result is None
    
    # Priority Tests (10 tests)
    @pytest.mark.parametrize("rule1_text,rule2_text,rule1_priority,rule2_priority,transaction,expected_result", [
        # Lower priority fraud rule, higher priority whitelist rule
        ("Todo pix do CPF 12345678901 é fraude", "Todo pix do CPF 12345678901 nao é fraude", 1, 10,
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         False),
        # Higher priority fraud rule, lower priority whitelist rule
        ("Todo pix do CPF 12345678901 é fraude", "Todo pix do CPF 12345678901 nao é fraude", 10, 1,
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True),
        # Same priority, first rule wins
        ("Todo pix do CPF 12345678901 é fraude", "Todo pix do CPF 12345678901 nao é fraude", 5, 5,
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True),
        # Different CPFs, no conflict
        ("Todo pix do CPF 12345678901 é fraude", "Todo pix do CPF 98765432100 nao é fraude", 1, 1,
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True),
        # Different CPFs, no conflict (whitelist)
        ("Todo pix do CPF 12345678901 é fraude", "Todo pix do CPF 98765432100 nao é fraude", 1, 1,
         {'canal': 'app', 'sender': {'cpfSender': '98765432100'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         False),
        # Valor rules with different priorities
        ("Todos os pix com valor superior a 1000 reais é fraude", "Todos os pix com valor superior a 1000 reais nao é fraude", 1, 10,
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T10:00:00'},
         False),
        # Valor rules with different priorities (reverse)
        ("Todos os pix com valor superior a 1000 reais é fraude", "Todos os pix com valor superior a 1000 reais nao é fraude", 10, 1,
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T10:00:00'},
         True),
        # Horário rules with different priorities
        ("Todos os pix depois das 22:00 é fraude", "Todos os pix depois das 22:00 nao é fraude", 1, 10,
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T23:00:00'},
         False),
        # Horário rules with different priorities (reverse)
        ("Todos os pix depois das 22:00 é fraude", "Todos os pix depois das 22:00 nao é fraude", 10, 1,
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T23:00:00'},
         True),
        # Banco rules with different priorities
        ("Bloquear todas as transacoes do banco 001", "Bloquear todas as transacoes do banco 001 nao é fraude", 1, 10,
         {'canal': 'app', 'sender': {'banco': 1, 'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         False),
    ])
    def test_priority_rules(self, parser, evaluator, rule1_text, rule2_text, rule1_priority, rule2_priority, transaction, expected_result):
        """Test rule priority evaluation."""
        rule1 = parser.parse(rule1_text)
        rule1.priority = rule1_priority
        rule2 = parser.parse(rule2_text)
        rule2.priority = rule2_priority
        
        evaluator.add_rules([rule1, rule2])
        
        transaction['id'] = 'test-1'
        if 'receiver' not in transaction:
            transaction['receiver'] = {'cpfReceiver': '98765432100'}
        
        result = evaluator.evaluate_single(transaction)
        assert result == expected_result
        
        evaluator.rules = {}  # Clear rules for next test
    
    # Disabled Rules Tests (5 tests)
    @pytest.mark.parametrize("rule_text,transaction,expected_result", [
        ("Todo pix do CPF 12345678901 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         None),
        ("Todos os pix com valor superior a 1000 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T10:00:00'},
         None),
        ("Todos os pix depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T23:00:00'},
         None),
        ("Todos os pix do app é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         None),
        ("Bloquear todas as transacoes do banco 001",
         {'canal': 'app', 'sender': {'banco': 1, 'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         None),
    ])
    def test_disabled_rules(self, parser, evaluator, rule_text, transaction, expected_result):
        """Test that disabled rules are not evaluated."""
        rule = parser.parse(rule_text)
        rule.enabled = False
        evaluator.add_rule(rule)
        
        transaction['id'] = 'test-1'
        if 'receiver' not in transaction:
            transaction['receiver'] = {'cpfReceiver': '98765432100'}
        
        result = evaluator.evaluate_single(transaction)
        assert result == expected_result
        
        evaluator.rules = {}
    
    # Edge Cases Tests (10 tests)
    @pytest.mark.parametrize("rule_text,transaction,expected_result", [
        # Empty rule
        ("", {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         None),
        # Missing field in transaction
        ("Todo pix do CPF 12345678901 é fraude",
         {'canal': 'app', 'sender': {}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         None),
        # Invalid timestamp
        ("Todos os pix depois das 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': 'invalid'},
         None),
        # Zero valor
        ("Todos os pix com valor superior a 0 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 0.0, 'timestamp': '2026-04-25T10:00:00'},
         None),
        # Very high valor
        ("Todos os pix com valor superior a 100000 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1000000.0, 'timestamp': '2026-04-25T10:00:00'},
         True),
        # Edge hour (midnight)
        ("Todos os pix depois das 00:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T00:00:00'},
         None),
        # Edge hour (23:59)
        ("Todos os pix antes das 00:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T23:59:00'},
         None),
        # CPF with leading zeros
        ("Todo pix do CPF 00000000001 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '00000000001'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True),
        # Case insensitive canal
        ("Todos os pix do app é fraude",
         {'canal': 'APP', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True),
        # Decimal valor
        ("Todos os pix com valor superior a 100.50 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.51, 'timestamp': '2026-04-25T10:00:00'},
         True),
    ])
    def test_edge_cases(self, parser, evaluator, rule_text, transaction, expected_result):
        """Test edge cases and boundary conditions."""
        if rule_text:
            rule = parser.parse(rule_text)
            evaluator.add_rule(rule)
        
        transaction['id'] = 'test-1'
        if 'receiver' not in transaction:
            transaction['receiver'] = {'cpfReceiver': '98765432100'}
        
        result = evaluator.evaluate_single(transaction)
        assert result == expected_result
        
        evaluator.rules = {}
    
    # Multiple Rules Same Transaction Tests (5 tests)
    @pytest.mark.parametrize("rules_texts,transaction,expected_matched_count", [
        # Multiple CPF rules, only one matches
        (["Todo pix do CPF 12345678901 é fraude", "Todo pix do CPF 98765432100 é fraude"],
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         1),
        # Multiple rules, both match (same CPF)
        (["Todo pix do CPF 12345678901 é fraude", "Todos os pix do app do CPF 12345678901 é fraude"],
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         2),
        # Multiple rules, none match
        (["Todo pix do CPF 12345678901 é fraude", "Todos os pix com valor superior a 10000 reais é fraude"],
         {'canal': 'app', 'sender': {'cpfSender': '98765432100'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         0),
        # Mix of fraud and legitimate rules
        (["Todo pix do CPF 12345678901 é fraude", "Todo pix do CPF 12345678901 nao é fraude"],
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         2),
        # Different field rules
        (["Todo pix do CPF 12345678901 é fraude", "Todos os pix do app é fraude", "Todos os pix com valor superior a 1000 reais é fraude"],
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T10:00:00'},
         3),
    ])
    def test_multiple_rules_same_transaction(self, parser, evaluator, rules_texts, transaction, expected_matched_count):
        """Test multiple rules evaluating same transaction."""
        for rule_text in rules_texts:
            rule = parser.parse(rule_text)
            evaluator.add_rule(rule)
        
        transaction['id'] = 'test-1'
        if 'receiver' not in transaction:
            transaction['receiver'] = {'cpfReceiver': '98765432100'}
        
        result = evaluator.evaluate(transaction)
        assert len(result['matched_rules']) == expected_matched_count
        
        evaluator.rules = {}
    
    # Brazilian Portuguese Variations Tests (10 tests)
    @pytest.mark.parametrize("rule_text,transaction,expected_match,is_fraud", [
        # "não" with tilde
        ("Todo pix do CPF 12345678901 não é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, False),
        # "nao" without tilde
        ("Todo pix do CPF 12345678901 nao é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, False),
        # "é uma fraude"
        ("Todo pix do CPF 12345678901 é uma fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        # "é um fraude"
        ("Todo pix do CPF 12345678901 é um fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        # "é suspeita"
        ("Todo pix do CPF 12345678901 é suspeita",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        # "não é suspeita"
        ("Todo pix do CPF 12345678901 não é suspeita",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, False),
        # "bloquear" instead of "é fraude"
        ("Bloquear todas as transacoes do CPF 12345678901",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        # "superior a" vs "maior que"
        ("Todos os pix com valor maior que 1000 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 1500.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        # "abaixo de" vs "menor que"
        ("Todos os pix com valor abaixo de 100 reais é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 50.0, 'timestamp': '2026-04-25T10:00:00'},
         True, True),
        # "após as" vs "depois das"
        ("Todos os pix após as 22:00 é fraude",
         {'canal': 'app', 'sender': {'cpfSender': '12345678901'}, 'valor': 100.0, 'timestamp': '2026-04-25T23:00:00'},
         True, True),
    ])
    def test_portuguese_variations(self, parser, rule_text, transaction, expected_match, is_fraud):
        """Test Brazilian Portuguese language variations."""
        rule = parser.parse(rule_text)
        transaction['id'] = 'test-1'
        if 'receiver' not in transaction:
            transaction['receiver'] = {'cpfReceiver': '98765432100'}
        result = rule.matches(transaction)
        if expected_match:
            assert result == is_fraud
        else:
            assert result is None
