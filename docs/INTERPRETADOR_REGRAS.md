# Interpretador de Regras em Linguagem Natural

## Visão Geral

O Interpretador de Regras é um componente do sistema de detecção de fraude que permite usuários operacionais criarem regras de fraude em linguagem natural em português brasileiro (PT-BR). Isso elimina a necessidade de conhecimento técnico para definir regras de negócio.

## Arquitetura

```
src/rule_engine/
├── __init__.py          # Exportações públicas
├── parser.py            # Parser de linguagem natural
├── rule.py              # Estruturas de dados (Rule, Condition, ActionType)
└── evaluator.py         # Avaliador de regras contra transações
```

### Componentes

#### 1. RuleParser (`parser.py`)

Converte texto em linguagem natural em objetos `Rule` estruturados.

**Padrões Suportados:**
- **CPF**: `cpf do 12345678901` ou `CPF do sender 12345678901`
- **Valor**: `valor superior a 1000 reais`, `valor menor que 100 reais`
- **Horário**: `depois das 22:00`, `antes das 06:00`
- **Canal**: `pix do app`, `pix do web`, `pix do api`
- **Produto**: `todo pix`, `toda ted`

**Ações Suportadas:**
- `é fraude` → Marca como fraude
- `não é fraude` → Marca como legítimo (whitelist)

**Exemplos de Regras:**
```
"Todos os pix do app com valor superior a 1000 reais depois das 22:00 é fraude"
"Todo pix do CPF 12345678901 é fraude"
"Todo pix do CPF 98765432100 não é fraude"
"Todos os pix antes das 06:00 é fraude"
```

#### 2. Rule (`rule.py`)

Estrutura de dados que representa uma regra de fraude.

**Campos:**
- `id`: Identificador único da regra
- `name`: Nome da regra
- `description`: Descrição detalhada
- `original_text`: Texto original em linguagem natural
- `conditions`: Lista de condições a serem avaliadas
- `action`: Ação a executar (MARK_AS_FRAUD ou MARK_AS_LEGITIMATE)
- `enabled`: Se a regra está ativa
- `priority`: Prioridade da regra (maior valor = maior prioridade)

**Tipos de Condição:**
- `CPF_SENDER`: CPF do remetente
- `CPF_RECEIVER`: CPF do destinatário
- `VALOR`: Valor da transação
- `HORARIO`: Horário da transação
- `CANAL`: Canal da transação (app, web, api)
- `BANCO_SENDER`: Banco do remetente
- `BANCO_RECEIVER`: Banco do destinatário

**Operadores:**
- `EQUALS`: Igual a
- `NOT_EQUALS`: Diferente de
- `GREATER_THAN`: Maior que
- `LESS_THAN`: Menor que
- `GREATER_THAN_OR_EQUAL`: Maior ou igual a
- `LESS_THAN_OR_EQUAL`: Menor ou igual a
- `CONTAINS`: Contém
- `AFTER`: Depois de (horário)
- `BEFORE`: Antes de (horário)

#### 3. RuleEvaluator (`evaluator.py`)

Avalia transações contra um conjunto de regras.

**Funcionalidades:**
- Adicionar/remover regras
- Habilitar/desabilitar regras
- Avaliar transação contra todas as regras
- Retornar resultado com regras que deram match
- Suporte a priorização de regras

## Integração com API FastAPI

### Endpoints Disponíveis

#### 1. Criar Regra
```
POST /rules
Query Params:
  - rule_text: Texto da regra em linguagem natural (obrigatório)
  - name: Nome da regra (opcional)
  - description: Descrição da regra (opcional)

Response:
  {
    "rule_id": "rule_1",
    "name": "Regra 1",
    "description": "Texto original",
    "original_text": "Texto original",
    "action": "mark_as_fraud",
    "conditions_count": 3,
    "enabled": true,
    "priority": 0
  }
```

#### 2. Listar Regras
```
GET /rules

Response:
  {
    "total_rules": 5,
    "rules": [...]
  }
```

#### 3. Obter Regra Específica
```
GET /rules/{rule_id}

Response:
  {
    "rule_id": "rule_1",
    "name": "Regra 1",
    "description": "...",
    "original_text": "...",
    "action": "mark_as_fraud",
    "conditions": [...],
    "enabled": true,
    "priority": 0
  }
```

#### 4. Deletar Regra
```
DELETE /rules/{rule_id}

Response:
  {"message": "Rule deleted successfully"}
```

#### 5. Habilitar/Desabilitar Regra
```
POST /rules/{rule_id}/enable
POST /rules/{rule_id}/disable

Response:
  {"message": "Rule enabled/disabled successfully"}
```

#### 6. Avaliar Transação
```
POST /rules/evaluate
Body: {transaction_data}

Response:
  {
    "transaction_id": "...",
    "matched_rules": [...],
    "is_fraud_by_rules": true,
    "is_legitimate_by_rules": false,
    "rule_count": 5
  }
```

### Integração com Endpoint /predict

O endpoint `/predict` foi atualizado para avaliar regras antes de executar o modelo ML:

1. **Primeiro**: Avalia a transação contra todas as regras
2. **Se regra de fraude der match**: Retorna imediatamente com probabilidade 1.0
3. **Se regra de whitelist der match**: Retorna imediatamente com probabilidade 0.0
4. **Se nenhuma regra der match**: Executa o modelo ML normalmente

**Benefícios:**
- Latência reduzida para casos claros de fraude/whitelist
- Controle total por usuários operacionais
- Conformidade regulatória (regras determinísticas)
- Priorização de regras sobre modelo ML

## Exemplos de Uso

### Exemplo 1: Regra Simples de CPF

**Texto:** "Todo pix do CPF 12345678901 é fraude"

**Estrutura Gerada:**
```python
Rule(
    id="rule_1",
    name="Regra 1",
    description="Todo pix do CPF 12345678901 é fraude",
    original_text="Todo pix do CPF 12345678901 é fraude",
    conditions=[
        Condition(
            field=ConditionType.CPF_SENDER,
            operator=Operator.EQUALS,
            value="12345678901"
        )
    ],
    action=ActionType.MARK_AS_FRAUD
)
```

### Exemplo 2: Regra Combinada

**Texto:** "Todos os pix do app com valor superior a 1000 reais depois das 22:00 é fraude"

**Estrutura Gerada:**
```python
Rule(
    conditions=[
        Condition(field=ConditionType.CANAL, operator=Operator.EQUALS, value="app"),
        Condition(field=ConditionType.VALOR, operator=Operator.GREATER_THAN, value=1000.0),
        Condition(field=ConditionType.HORARIO, operator=Operator.AFTER, value=22)
    ],
    action=ActionType.MARK_AS_FRAUD
)
```

### Exemplo 3: Whitelist

**Texto:** "Todo pix do CPF 98765432100 não é fraude"

**Estrutura Gerada:**
```python
Rule(
    conditions=[
        Condition(field=ConditionType.CPF_SENDER, operator=Operator.EQUALS, value="98765432100")
    ],
    action=ActionType.MARK_AS_LEGITIMATE
)
```

## Priorização de Regras

Regras podem ter prioridade definida. Regras com maior prioridade são avaliadas primeiro e podem sobrepor regras de menor prioridade.

**Cenário de Whitelist vs Blacklist:**
```python
# Blacklist: CPF 12345678901 é sempre fraude (prioridade 1)
rule1.priority = 1

# Whitelist: CPF 98765432100 nunca é fraude (prioridade 10)
rule2.priority = 10

# A regra de whitelist tem prioridade maior
```

## Conformidade Regulatória

### Auditabilidade
- Todas as regras são armazenadas com seu texto original
- Cada avaliação registra quais regras deram match
- Histórico completo de decisões baseadas em regras

### Rastreabilidade
- Cada transação pode ser rastreada até a regra específica que a marcou
- Regras podem ser habilitadas/desabilitadas sem código
- Mudanças em regras são imediatas (sem reimplantação)

### BACEN/LGPD
- Regras são determinísticas (sem black-box)
- Explicação clara de por que uma transação foi marcada
- Controle total sobre critérios de fraude

## Performance

### Latência
- Avaliação de regras: <5ms
- Integração com /predict: Adiciona ~2ms ao tempo total
- Regras têm prioridade sobre modelo ML (reduz latência para casos claros)

### Escalabilidade
- Avaliação em memória (sem dependências externas)
- Suporta milhares de regras simultâneas
- Custo zero (sem APIs externas)

## Boas Práticas

### 1. Regras Específicas vs Genéricas
- **Recomendado**: Regras específicas (CPF específico, valor exato)
- **Evitar**: Regras muito genéricas que podem gerar muitos falsos positivos

### 2. Priorização
- Use prioridade alta para whitelists (ex: clientes VIP)
- Use prioridade baixa para regras de risco moderado
- Whitelists devem sempre ter prioridade sobre blacklists

### 3. Teste de Regras
- Use o endpoint `/rules/evaluate` para testar regras antes de habilitar
- Valide regras com transações históricas
- Monitore falsos positivos após implementação

### 4. Manutenção
- Revise regras periodicamente
- Remova regras obsoletas
- Documente o propósito de cada regra

## Limitações

### Linguagem Natural
- Apenas português brasileiro (PT-BR)
- Sintaxe específica deve ser seguida
- Não suporta linguagem natural livre (usa padrões definidos)

### Campos Suportados
Atualmente suporta: CPF, valor, horário, canal, banco
Campos futuros podem ser adicionados conforme necessidade

### Operadores
Operadores complexos (OR, NOT lógico) não são suportados
Apenas AND lógico entre condições

## Testes

### Cobertura de Testes
- 38 testes específicos para o interpretador de regras
- Testes de parser, avaliação de condições, avaliador
- Testes de cenários complexos (whitelist/blacklist)
- Testes de edge cases

### Executar Testes
```bash
python -m pytest tests/test_rule_engine.py -v
```

## V2 — Evolução do Interpretador (entregue)

A versão V2 mantém **100% de retrocompatibilidade** com a V1 e adiciona:

### 1. Persistência de regras

Regras agora sobrevivem a reinícios. Persistência via padrão Repository
(igual ao usado para modelos e perfis):

- `RuleRepository` (interface abstrata)
- `JSONRuleRepository` (default em `data/rules.json`, escrita atômica)
- `InMemoryRuleRepository` (uso em testes / DI)

```python
from src.rule_engine import RuleEvaluator, JSONRuleRepository, RuleAuditLogger

evaluator = RuleEvaluator(
    repository=JSONRuleRepository("data/rules.json"),
    audit_logger=RuleAuditLogger("logs/rule_audit.jsonl"),
)
```

### 2. Operadores lógicos OR e NOT

- **OR**: o conector `" ou "` no texto separa grupos. Cada grupo é AND
  internamente; grupos são unidos por OR.
- **NOT**: tokens `exceto` ou `não sendo` antes de uma condição marcam-na
  como `negated=True` (resultado da comparação é invertido).

```
"Todo pix do app ou todo pix do web é fraude"
"Todo pix exceto do CPF 12345678901 é fraude"
```

Modelo de dados:

- `Rule.condition_groups: List[List[Condition]]` — quando preenchido tem
  precedência sobre `conditions`. Quando vazio, fallback para `conditions`
  (AND simples — comportamento V1 intacto).
- `Condition.negated: bool = False`.

### 3. Audit log BACEN

Cada match registra um JSON-line em `logs/rule_audit.jsonl`:

```json
{"timestamp": "2026-04-26T18:25:00+00:00", "rule_id": "rule_3",
 "rule_name": "Bloqueio CPF VIP", "transaction_id": "tx-99",
 "action": "mark_as_fraud", "is_fraud": true,
 "conditions_matched": 2, "priority": 5}
```

Append-only, imutável, ingerível por ELK/Splunk/BigQuery.

### 4. Métricas operacionais por regra

`RuleMetricsRegistry` mantém, por `rule_id`:

- `hit_count` — número total de matches
- `last_match_at` — ISO timestamp do último match
- `false_positive_count`, `last_false_positive_at` — para feedback humano

Endpoint: `GET /rules/metrics`.

### 5. Detecção de conflitos

`detect_conflicts(rules)` retorna pares de regras com **mesma assinatura
de condições e ações opostas** (uma blacklist e uma whitelist contraditórias).

Endpoint: `GET /rules/conflicts`.

### 6. Dry-run / Simulação

Antes de habilitar uma regra, valide o comportamento contra um lote de
transações sem persistir nem auditar:

```
POST /rules/simulate
{
  "rule_texts": ["Todo pix do CPF 12345678901 é fraude"],
  "transactions": [ { ... }, { ... } ]
}
```

Retorna `flagged_fraud`, `flagged_legitimate`, `untouched`, `hits_by_rule`.

### 7. Validação prévia

Para feedback imediato de UX antes de criar uma regra:

```
POST /rules/validate
{ "rule_text": "Todo pix do CPF 12345678901 é fraude" }
```

Retorna `valid`, `action`, `groups`, `warnings` — não persiste nada.

### 8. Import / Export

```
GET  /rules/export                → JSON com todas as regras
POST /rules/import {rules, replace?}  → cria ou substitui em lote
```

Habilita versionamento via Git, replicação entre ambientes e backup.

### 9. PATCH parcial

```
PATCH /rules/{rule_id}
{ "priority": 99, "enabled": false }
```

Campos editáveis: `enabled`, `priority`, `name`, `description`. Outros
campos exigem deletar e recriar a regra para manter rastreabilidade.

---

### Tabela de endpoints V2

| Método | Path | Descrição |
|---|---|---|
| POST | `/rules/validate` | Dry-parse, sem persistir |
| POST | `/rules/simulate` | Dry-run em lote |
| GET  | `/rules/conflicts` | Lista pares conflitantes |
| GET  | `/rules/export` | Exporta catálogo (JSON) |
| POST | `/rules/import` | Importa catálogo (JSON) |
| GET  | `/rules/metrics` | Métricas por regra |
| PATCH | `/rules/{rule_id}` | Atualiza campos editáveis |

### Cobertura de testes V2

- 25 testes novos em `tests/test_rule_engine_v2.py`
- 148 testes legados (`tests/test_rule_engine.py` e `tests/test_rule_engine_comprehensive.py`) **continuam verdes**
- Suite completa: **293 passed, 5 skipped, 0 errors**

### Conformidade BACEN/LGPD V2

- Audit log JSONL imutável e timestamped
- Persistência atômica via `os.replace` (sem corrupção em crash)
- Detecção automática de conflitos prevenindo decisões ambíguas
- Métricas por regra permitem identificar regras zumbis e calibrar falsos positivos

## Suporte

Para dúvidas ou problemas:
- Consulte a documentação técnica em `src/rule_engine/`
- Execute os testes para validar comportamento
- Entre em contato com a equipe de desenvolvimento
