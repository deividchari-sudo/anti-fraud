"""
Natural language parser for fraud detection rules.
"""

import re
from typing import List, Optional

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

    # Disjunction connector splitting groups (OR). We require word boundaries
    # to avoid matching inside words like "outro".
    OR_SPLIT = re.compile(r"\s+ou\s+", re.IGNORECASE)

    def parse(
        self,
        rule_text: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Rule:
        """Parse natural language rule text into Rule object.

        V2 behavior:
            - The text may contain multiple groups separated by the connector
              ``" ou "`` to express logical OR. Each group becomes an entry in
              ``Rule.condition_groups``. Inside a group, conditions are AND.
            - Tokens ``"exceto"`` or ``"nao sendo"`` immediately before a
              condition phrase mark that condition as ``negated`` (NOT).
        """
        self.rules_count += 1
        rule_id = f"rule_{self.rules_count}"

        rule_text_lower = rule_text.lower().strip()

        # Extract action from the full text first (action keyword may appear
        # only in the last clause).
        action = self._extract_action(rule_text_lower)

        # Split by OR connectors. We strip the action keyword fragment from
        # each piece so the same conditions are not duplicated.
        raw_groups = self.OR_SPLIT.split(rule_text_lower)
        groups: List[List[Condition]] = []
        for raw in raw_groups:
            piece = raw.strip()
            if not piece:
                continue
            group_conditions = self._extract_conditions(piece)
            if group_conditions:
                groups.append(group_conditions)

        # If the text had no OR connector, keep the legacy flat list as well
        # so callers relying on ``rule.conditions`` still work.
        if len(groups) <= 1:
            conditions = groups[0] if groups else self._extract_conditions(rule_text_lower)
            condition_groups: List[List[Condition]] = []
        else:
            # Flatten first group into ``conditions`` for backward compat
            conditions = list(groups[0])
            condition_groups = groups

        return Rule(
            id=rule_id,
            name=name or f"Regra {self.rules_count}",
            description=description or rule_text,
            original_text=rule_text,
            conditions=conditions,
            action=action,
            condition_groups=condition_groups,
        )

    @staticmethod
    def _is_negated(text: str, keyword_start: int) -> bool:
        """Return True if a NOT marker (``exceto``/``nao sendo``) precedes the
        given position in ``text`` within a short window.
        """
        window = text[max(0, keyword_start - 25): keyword_start]
        return bool(re.search(r"\b(exceto|n[aã]o\s+sendo)\b", window))

    def _extract_conditions(self, text: str) -> List[Condition]:
        """Extract conditions from rule text."""
        conditions = []

        # CPF condition
        cpf_match = re.search(self.PATTERNS["cpf"], text)
        if cpf_match and cpf_match.group(1):
            cpf = cpf_match.group(1)
            negated = self._is_negated(text, cpf_match.start())
            # Determine if it's sender or receiver
            if "receiver" in text or "remetente" in text:
                conditions.append(
                    Condition(
                        field=ConditionType.CPF_RECEIVER,
                        operator=Operator.EQUALS,
                        value=cpf,
                        negated=negated,
                    )
                )
            else:
                conditions.append(
                    Condition(
                        field=ConditionType.CPF_SENDER,
                        operator=Operator.EQUALS,
                        value=cpf,
                        negated=negated,
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
            negated = self._is_negated(text, canal_match.start())
            conditions.append(
                Condition(
                    field=ConditionType.CANAL,
                    operator=Operator.EQUALS,
                    value=canal,
                    negated=negated,
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

    def validate(self, rule_text: str) -> dict:
        """Dry-validate a rule text without registering it.

        Returns a structured report with detected conditions, action, groups,
        and warnings. Used by ``POST /rules/validate``.
        """
        warnings: List[str] = []
        try:
            rule = self.parse(rule_text)
            # The validate path consumed an id slot; release it so the next
            # real parse keeps deterministic numbering.
            self.rules_count -= 1
        except Exception as exc:  # pragma: no cover - defensive
            return {"valid": False, "error": str(exc), "warnings": warnings}

        total_conditions = sum(
            len(g) for g in (rule.condition_groups or [rule.conditions])
        )
        if total_conditions == 0:
            warnings.append(
                "Nenhuma condição reconhecida; a regra acionará para qualquer transação."
            )
        return {
            "valid": True,
            "action": rule.action.value,
            "groups": [
                [
                    {
                        "field": c.field.value,
                        "operator": c.operator.value,
                        "value": c.value,
                        "negated": c.negated,
                    }
                    for c in group
                ]
                for group in (rule.condition_groups or [rule.conditions])
            ],
            "warnings": warnings,
        }
