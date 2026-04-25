"""
Natural language parser for fraud detection rules.
"""

import re
from typing import List, Optional, Tuple

from .rule import ActionType, Condition, ConditionType, Operator, Rule


class RuleParser:
    """Parse natural language rules into Rule objects."""

    # Regex patterns for extracting components
    PATTERNS = {
        "cpf": r"(?:cpf|CPF)\s*(?:do|da|do\s+sender|do\s+receiver|do\s+remetente)?\s*(\d{10,11})",
        "valor": r"(?:valor|montante|quantia)\s*(?:superior\s+a|maior\s+que|igual\s+a|menor\s+que|abaixo\s+de)\s*(\d+(?:\.\d{2})?)\s*(?:reais|R\$)?",
        "horario": r"(?:depois\s+das|após\s+as|antes\s+das|antes\s+das)\s*(\d{2}):(\d{2})",
        "canal": r"(?:pix|transação|tudo)\s*(?:do|da|no|na)\s*(app|web|api)",
        "produto": r"(pix|ted|boleto|autenticação|login)",
        "banco": r"(?:do\s+)?(?:banco|Banco)\s*(?:do|da|do\s+sender|do\s+receiver|do\s+remetente)?\s*(\d+)",
        "is_fraud": r"(?:é|é\s+uma|é\s+um|bloquear|bloqueia|bloqueia)\s*(?:fraude|suspeita)",
        "is_not_fraud": r"(?:não|nao)(?:\s+é)?\s*(?:fraude|suspeita)",
    }

    def __init__(self):
        self.rules_count = 0

    def parse(
        self,
        rule_text: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Rule:
        """Parse natural language rule text into Rule object."""
        self.rules_count += 1
        rule_id = f"rule_{self.rules_count}"

        rule_text_lower = rule_text.lower().strip()

        # Extract conditions
        conditions = self._extract_conditions(rule_text_lower)

        # Extract action
        action = self._extract_action(rule_text_lower)

        return Rule(
            id=rule_id,
            name=name or f"Regra {self.rules_count}",
            description=description or rule_text,
            original_text=rule_text,
            conditions=conditions,
            action=action,
        )

    def _extract_conditions(self, text: str) -> List[Condition]:
        """Extract conditions from rule text."""
        conditions = []

        # CPF condition
        cpf_match = re.search(self.PATTERNS["cpf"], text)
        if cpf_match and cpf_match.group(1):
            cpf = cpf_match.group(1)
            # Determine if it's sender or receiver
            if "receiver" in text or "remetente" in text:
                conditions.append(
                    Condition(
                        field=ConditionType.CPF_RECEIVER,
                        operator=Operator.EQUALS,
                        value=cpf,
                    )
                )
            else:
                conditions.append(
                    Condition(
                        field=ConditionType.CPF_SENDER,
                        operator=Operator.EQUALS,
                        value=cpf,
                    )
                )

        # Banco condition
        banco_match = re.search(self.PATTERNS["banco"], text)
        if banco_match and banco_match.group(1):
            banco = banco_match.group(1)
            # Determine if it's sender or receiver
            if "receiver" in text or "remetente" in text:
                conditions.append(
                    Condition(
                        field=ConditionType.BANCO_RECEIVER,
                        operator=Operator.EQUALS,
                        value=int(banco),
                    )
                )
            else:
                conditions.append(
                    Condition(
                        field=ConditionType.BANCO_SENDER,
                        operator=Operator.EQUALS,
                        value=int(banco),
                    )
                )

        # Valor condition
        valor_match = re.search(self.PATTERNS["valor"], text)
        if valor_match and valor_match.group(1):
            valor = float(valor_match.group(1))
            operator_text = valor_match.group(0)

            if "superior" in operator_text or "maior" in operator_text:
                operator = Operator.GREATER_THAN
            elif "menor" in operator_text or "abaixo" in operator_text:
                operator = Operator.LESS_THAN
            else:
                operator = Operator.EQUALS

            conditions.append(
                Condition(field=ConditionType.VALOR, operator=operator, value=valor)
            )

        # Horário condition
        horario_match = re.search(self.PATTERNS["horario"], text)
        if horario_match and horario_match.group(1):
            hour = int(horario_match.group(1))
            operator_text = horario_match.group(0)

            if "depois" in operator_text or "após" in operator_text:
                operator = Operator.AFTER
            else:
                operator = Operator.BEFORE

            conditions.append(
                Condition(field=ConditionType.HORARIO, operator=operator, value=hour)
            )

        # Canal condition (only if it matches "pix do app" pattern)
        canal_match = re.search(self.PATTERNS["canal"], text)
        if canal_match and len(canal_match.groups()) > 0 and canal_match.group(1):
            canal = canal_match.group(1)
            conditions.append(
                Condition(
                    field=ConditionType.CANAL, operator=Operator.EQUALS, value=canal
                )
            )

        # Produto condition (only if it's a simple "Todo pix é fraude" without other specific conditions)
        # Check if text is like "todo pix é fraude" or "toda pix é fraude" without CPF, valor, horario, or canal
        if not conditions:
            produto_pattern = r"^(?:todo|toda)\s+(pix|ted|boleto|autenticação|login)\s+"
            produto_match = re.search(produto_pattern, text)
            if produto_match and produto_match.group(1):
                produto = produto_match.group(1)
                conditions.append(
                    Condition(
                        field=ConditionType.CANAL,
                        operator=Operator.CONTAINS,
                        value=produto,
                    )
                )

        return conditions

    def _extract_action(self, text: str) -> ActionType:
        """Extract action from rule text."""
        # Normalize text to handle encoding issues - Portuguese PT-BR
        normalized_text = text.lower()

        # Check for "não é fraude" or "nao é fraude" or "nao fraude"
        if re.search(r"(?:não|nao)(?:\s+é)?\s*(?:fraude|suspeita)", normalized_text):
            return ActionType.MARK_AS_LEGITIMATE
        elif re.search(r"(?:é|é\s+uma|é\s+um)\s*(?:fraude|suspeita)", normalized_text):
            return ActionType.MARK_AS_FRAUD

        # Default to mark as fraud if not specified
        return ActionType.MARK_AS_FRAUD

    def parse_multiple(self, rule_texts: List[str]) -> List[Rule]:
        """Parse multiple rule texts."""
        return [self.parse(rule_text) for rule_text in rule_texts]
