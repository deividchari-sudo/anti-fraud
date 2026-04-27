# Postman Collection - Fraud Detection API v2.0.0

Este documento contém exemplos de JSON para testar todos os endpoints da API usando Postman.

## Base URL
```
http://localhost:8000
```

> **Nota Windows**: se a porta 8000 estiver bloqueada (`WinError 10013`), use
> `http://127.0.0.1:8001` e ajuste a variável `baseUrl` da collection.

## Collections

### 1. Health Check

#### GET /health

Verifica status da API

**Request**:
```http
GET http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0"
}
```

---

### 2. Predição de Fraude

#### POST /predict

Predição de fraude (transação única) com análise comportamental

**Request**:
```http
POST http://localhost:8000/predict
Content-Type: application/json
```

**Body**:
```json
{
  "payload": {
    "id": "test-id-123",
    "timestamp": "2026-04-25T19:55:11",
    "canal": "web",
    "produto": "pix",
    "jornada": "pix_troco",
    "direcao": "saida",
    "sender": {
      "banco": 152,
      "agencia": "0001",
      "nuConta": 61075434,
      "cpfSender": "75096441908"
    },
    "receiver": {
      "banco": 888,
      "agencia": "3061",
      "nuConta": 628328,
      "cpfReceiver": "69473704016"
    },
    "valor": 1000.0,
    "extra_info": {
      "codigo_barra": null,
      "motivo_acesso": null
    }
  }
}
```

**Response**:
```json
{
  "transaction_id": "test-id-123",
  "fraud_probability": 0.0454,
  "is_fraud": false,
  "confidence": "low",
  "processing_time_ms": 28.25,
  "timestamp": "2026-04-25T19:55:11.539100",
  "explanation": null,
  "behavioral_analysis": {
    "has_profile": false,
    "transaction_count": 0,
    "is_cold_start": true,
    "is_anomaly": true,
    "anomaly_score": 0.5,
    "anomalies": [
      {
        "type": "horario_anomaly",
        "severity": "medium",
        "description": "Transaction at 19:00 is outside habitual hour range",
        "expected_range": "6h - 18h",
        "actual_value": "19:00"
      }
    ]
  }
}
```

---

### 3. Predição em Lote

#### POST /predict/batch

Predição em lote

**Request**:
```http
POST http://localhost:8000/predict/batch
Content-Type: application/json
```

**Body**:
```json
{
  "transactions": [
    {
      "payload": {
        "id": "tx-001",
        "timestamp": "2024-01-15T10:30:00",
        "canal": "mobile",
        "produto": "pix",
        "jornada": "transferencia",
        "direcao": "send",
        "sender": {
          "banco": 260,
          "agencia": "0001",
          "nuConta": 123456,
          "cpfSender": "12345678901"
        },
        "receiver": {
          "banco": 1,
          "agencia": "0002",
          "nuConta": 654321,
          "cpfReceiver": "98765432100"
        },
        "valor": 1000.0,
        "extra_info": {
          "codigo_barra": "",
          "motivo_acesso": ""
        }
      }
    }
  ]
}
```

**Response**:
```json
{
  "results": [
    {
      "transaction_id": "tx-001",
      "fraud_probability": 0.0454,
      "is_fraud": false,
      "confidence": "low",
      "processing_time_ms": 22.5,
      "timestamp": "2024-01-15T10:30:00.022Z",
      "explanation": null,
      "behavioral_analysis": {
        "risk_score": 0.1,
        "profile_status": "normal",
        "is_anomaly": false,
        "cold_start": false
      }
    }
  ],
  "total_transactions": 1,
  "processing_time_ms": 45.12,
  "avg_time_per_transaction": 45.12
}
```

### 4. Predição Segmentada (por Produto e Canal)

#### POST /predict/segmented

Usa modelo especializado para o segmento `(produto, canal)` se existir em `models/`. Fallback automático para o modelo global. Estratégias: `auto` (padrão), `global`, `specialized`.

**Request**:
```http
POST http://localhost:8000/predict/segmented
Content-Type: application/json
```

**Body**:
```json
{
  "payload": {
    "id": "tx-seg-001",
    "timestamp": "2024-01-15T10:30:00",
    "canal": "mobile",
    "produto": "pix",
    "jornada": "transferencia",
    "direcao": "send",
    "sender": {
      "banco": 260,
      "agencia": "0001",
      "nuConta": 123456,
      "cpfSender": "12345678901"
    },
    "receiver": {
      "banco": 1,
      "agencia": "0002",
      "nuConta": 654321,
      "cpfReceiver": "98765432100"
    },
    "valor": 1000.0,
    "extra_info": {
      "codigo_barra": "",
      "motivo_acesso": ""
    }
  },
  "strategy": "auto"
}
```

**Response**:
```json
{
  "transaction_id": "tx-seg-001",
  "fraud_probability": 0.92,
  "is_fraud": true,
  "confidence": "high",
  "processing_time_ms": 18.5,
  "timestamp": "2024-01-15T10:30:00.018Z",
  "segment": "pix_mobile",
  "model_used": "specialized",
  "threshold_used": 0.7,
  "explanation": {
    "base_value": 0.12,
    "fraud_probability": 0.92,
    "top_contributing_features": {
      "valor": 0.35,
      "transacoes_ultimas_1h": 0.28,
      "nova_relacao": 0.19
    },
    "explanation_summary": "Indicadores de fraude: valor, transacoes_ultimas_1h, nova_relacao."
  }
}
```

**Estratégias disponíveis:**
- `auto`: usa especializado se existir, senão global (padrão)
- `global`: sempre modelo global
- `specialized`: exige modelo especializado; retorna 422 se não existir

**Latência esperada:**
- Cache warm: +5ms vs `/predict`
- Cold-load do segmento: +80ms na primeira chamada

---

### 5. Informações do Modelo

#### GET /model/info

Informações do modelo

**Request**:
```http
GET http://localhost:8000/model/info
```

**Response**:
```json
{
  "model_type": "XGBoost",
  "feature_count": 71,
  "threshold": 0.5,
  "top_features": {
    "valor": 0.234,
    "sender_banco": 0.156,
    "receiver_banco": 0.123
  }
}
```

---

### 6. Regras (Rule Engine)

#### POST /rules

Criar nova regra

**Request**:
```http
POST http://localhost:8000/rules
Content-Type: application/json
```

**Body**:
```json
{
  "rule_text": "Todo pix do CPF 12345678901 é fraude",
  "name": "Regra CPF Suspeito",
  "description": "Marca como fraude transações do CPF específico"
}
```

**Response**:
```json
{
  "rule_id": "rule_1",
  "name": "Regra CPF Suspeito",
  "description": "Texto original",
  "original_text": "Todo pix do CPF 12345678901 é fraude",
  "action": "mark_as_fraud",
  "conditions_count": 1,
  "enabled": true,
  "priority": 0
}
```

#### GET /rules

Listar todas as regras

**Request**:
```http
GET http://localhost:8000/rules
```

**Response**:
```json
{
  "total_rules": 5,
  "rules": [
    {
      "rule_id": "rule_1",
      "name": "Regra CPF Suspeito",
      "enabled": true
    }
  ]
}
```

#### GET /rules/{rule_id}

Obter regra específica

**Request**:
```http
GET http://localhost:8000/rules/rule_1
```

**Response**:
```json
{
  "rule_id": "rule_1",
  "name": "Regra CPF Suspeito",
  "description": "Texto original",
  "original_text": "Todo pix do CPF 12345678901 é fraude",
  "action": "mark_as_fraud",
  "conditions": [
    {
      "type": "cpf",
      "operator": "equals",
      "value": "12345678901"
    }
  ],
  "enabled": true,
  "priority": 0
}
```

#### DELETE /rules/{rule_id}

Deletar regra

**Request**:
```http
DELETE http://localhost:8000/rules/rule_1
```

**Response**:
```json
{
  "message": "Rule rule_1 deleted successfully"
}
```

#### POST /rules/{rule_id}/enable

Habilitar regra

**Request**:
```http
POST http://localhost:8000/rules/rule_1/enable
```

**Response**:
```json
{
  "rule_id": "rule_1",
  "enabled": true
}
```

#### POST /rules/{rule_id}/disable

Desabilitar regra

**Request**:
```http
POST http://localhost:8000/rules/rule_1/disable
```

**Response**:
```json
{
  "rule_id": "rule_1",
  "enabled": false
}
```

#### POST /rules/evaluate

Avaliar transação contra regras

**Request**:
```http
POST http://localhost:8000/rules/evaluate
Content-Type: application/json
```

**Body**:
```json
{
  "payload": {
    "id": "test-id-123",
    "timestamp": "2026-04-25T19:55:11",
    "canal": "app",
    "produto": "pix",
    "sender": {
      "cpfSender": "12345678901"
    },
    "valor": 100.0
  }
}
```

**Response**:
```json
{
  "transaction_id": "test-id-123",
  "is_fraud_by_rules": true,
  "is_legitimate_by_rules": false,
  "matched_rules": ["rule_1"]
}
```

---

### 7. Behavioral Profiling

#### GET /profile/{cpf}

Obter perfil comportamental do usuário

**Request**:
```http
GET http://localhost:8000/profile/12345678901
```

**Response**:
```json
{
  "cpf": "hashed_cpf",
  "created_at": "2026-04-25T19:00:00",
  "last_updated": "2026-04-25T19:00:00",
  "transaction_count": 150,
  "is_cold_start": false,
  "statistics": {
    "valor": {
      "mean": 2500.0,
      "std": 1500.0,
      "median": 2000.0,
      "p25": 1500.0,
      "p75": 3000.0,
      "p95": 5000.0,
      "min": 100.0,
      "max": 10000.0
    },
    "hora": {
      "mean": 14.0,
      "std": 3.5,
      "median": 14.0,
      "p25": 11.0,
      "p75": 17.0
    },
    "frequencia": {
      "transactions_per_day_mean": 5.2,
      "transactions_per_day_std": 2.1,
      "days_active": 30
    }
  },
  "destinations": {
    "common_cpfs": {},
    "common_bancos": {}
  },
  "canais": {
    "app": {
      "count": 100,
      "percentage": 0.67
    },
    "web": {
      "count": 40,
      "percentage": 0.27
    },
    "api": {
      "count": 10,
      "percentage": 0.06
    }
  },
  "produtos": {
    "pix": {
      "count": 120,
      "percentage": 0.80
    },
    "ted": {
      "count": 25,
      "percentage": 0.17
    },
    "boleto": {
      "count": 5,
      "percentage": 0.03
    },
    "autenticacao": null
  },
  "temporal_features": {
    "sliding_windows": {
      "window_7_days": {
        "transaction_count": 15,
        "total_amount": 37500.0,
        "avg_amount": 2500.0,
        "max_amount": 5000.0,
        "min_amount": 100.0
      },
      "window_30_days": {
        "transaction_count": 150,
        "total_amount": 375000.0,
        "avg_amount": 2500.0,
        "max_amount": 10000.0,
        "min_amount": 100.0
      },
      "window_90_days": {
        "transaction_count": 150,
        "total_amount": 375000.0,
        "avg_amount": 2500.0,
        "max_amount": 10000.0,
        "min_amount": 100.0
      }
    },
    "trends": {
      "trend_direction": "stable",
      "trend_slope": 0.0,
      "recent_avg_amount": 2500.0,
      "older_avg_amount": 2500.0,
      "change_percent": 0.0
    },
    "seasonal": {
      "day_of_week_distribution": [0.14, 0.15, 0.16, 0.15, 0.14, 0.13, 0.13],
      "hour_of_day_distribution": [0.02, 0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.13, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01],
      "most_active_day": 2,
      "most_active_hour": 9,
      "avg_value_per_day": 2500.0
    }
  },
  "cluster_id": 2
}
```

#### POST /feedback/anomaly

Enviar feedback de analista sobre anomalias

**Request**:
```http
POST http://localhost:8000/feedback/anomaly
Content-Type: application/json
```

**Body**:
```json
{
  "transaction_id": "test-id-123",
  "cpf": "75096441908",
  "is_true_anomaly": true,
  "analyst_id": "analyst_001",
  "notes": "Transação suspeita - valor muito alto para horário",
  "anomaly_type": "valor_anomaly"
}
```

**Response**:
```json
{
  "status": "recorded",
  "transaction_id": "test-id-123",
  "cpf_hashed": true
}
```

---

---

### 8. Rule Engine V2 — Endpoints novos

#### POST /rules/validate

Dry-parse de uma regra (sem persistir). Retorna como o parser interpretou o texto.

**Body**:
```json
{
  "rule_text": "Todo pix do app ou todo pix do web com valor superior a 1000 reais é fraude"
}
```

**Response**:
```json
{
  "valid": true,
  "action": "mark_as_fraud",
  "groups": [
    [{"field": "canal", "operator": "equals", "value": "app", "negated": false}],
    [
      {"field": "valor", "operator": "greater_than", "value": 1000.0, "negated": false},
      {"field": "canal", "operator": "equals", "value": "web", "negated": false}
    ]
  ],
  "warnings": []
}
```

#### POST /rules/simulate

Dry-run em lote. Aceita `rule_texts` (regras candidatas) ou usa o catálogo persistido.
Não atualiza métricas e não grava no audit log.

**Body**:
```json
{
  "rule_texts": ["Todo pix do CPF 12345678901 é fraude"],
  "transactions": [
    {"id": "tx1", "sender": {"cpfSender": "12345678901"}},
    {"id": "tx2", "sender": {"cpfSender": "00000000000"}}
  ]
}
```

**Response**:
```json
{
  "total_transactions": 2,
  "flagged_fraud": 1,
  "flagged_legitimate": 0,
  "untouched": 1,
  "hits_by_rule": {"rule_1": 1},
  "evaluated_rules": 1
}
```

#### GET /rules/conflicts

Detecta pares de regras com mesma assinatura de condições e ações opostas (blacklist vs whitelist).

**Response**:
```json
{
  "total_conflicts": 1,
  "conflicts": [
    {
      "rule_a": {"id": "rule_1", "name": "Bloqueio CPF", "action": "mark_as_fraud", "priority": 0},
      "rule_b": {"id": "rule_2", "name": "Whitelist CPF", "action": "mark_as_legitimate", "priority": 0},
      "reason": "Mesma assinatura de condições com ações opostas",
      "resolution": "Ajuste a prioridade da regra de whitelist para um valor maior, ou remova uma das regras."
    }
  ]
}
```

#### GET /rules/export

Exporta o catálogo de regras como JSON portável.

**Response**:
```json
{
  "version": 2,
  "rules": [
    {
      "id": "rule_1",
      "name": "Regra 1",
      "action": "mark_as_fraud",
      "enabled": true,
      "priority": 0,
      "conditions": [
        {"field": "cpf_sender", "operator": "equals", "value": "12345678901", "negated": false}
      ],
      "condition_groups": []
    }
  ]
}
```

#### POST /rules/import

Importa catálogo (formato de `/rules/export`). Use `replace=true` para substituir tudo.

**Body**:
```json
{
  "rules": [/* mesmo formato de /rules/export */],
  "replace": false
}
```

**Response**:
```json
{"imported": 5, "total_rules": 5}
```

#### GET /rules/metrics

Retorna métricas operacionais por regra.

**Response**:
```json
{
  "rule_1": {
    "hit_count": 12,
    "last_match_at": "2026-04-27T11:32:08+00:00",
    "false_positive_count": 0,
    "last_false_positive_at": null
  }
}
```

#### PATCH /rules/{rule_id}

Atualiza campos editáveis: `enabled`, `priority`, `name`, `description`.

**Body**:
```json
{"priority": 99, "enabled": false}
```

**Response**:
```json
{
  "rule_id": "rule_1",
  "name": "Regra 1",
  "enabled": false,
  "priority": 99,
  "description": "..."
}
```

#### Sintaxe estendida V2

- **OR** entre cláusulas: `" ou "`
  - `"Todo pix do app ou todo pix do web é fraude"`
- **NOT** em condição: `"exceto"` ou `"não sendo"`
  - `"Todo pix exceto do CPF 12345678901 é fraude"`

---

## Importando para Postman

### Método 1: Importar como Collection

1. No Postman, clique em "Import"
2. Cole este arquivo ou copie os exemplos
3. Selecione "Postman Collection"
4. Clique em "Import"

### Método 2: Criar Collection Manual

1. Crie uma nova collection chamada "Fraud Detection API v2.0.0"
2. Adicione cada endpoint como um request
3. Use os exemplos de JSON acima como body
4. Configure a variável de ambiente `baseUrl` = `http://localhost:8000`

---

## Variáveis de Ambiente

Crie as seguintes variáveis de ambiente no Postman:

| Variável | Valor | Descrição |
|----------|-------|-----------|
| baseUrl | http://localhost:8000 | URL base da API |
| cpf_sender | 12345678901 | CPF de teste para remetente |
| cpf_receiver | 98765432100 | CPF de teste para destinatário |

---

## Testes de Carga

### Script de Teste no Postman

Adicione o seguinte script na aba "Tests" de cada request:

```javascript
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has required fields", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property("transaction_id");
});
```

---

## Cenários de Teste

### Cenário 1: Transação Normal

**Endpoint**: POST /predict

**CPF**: 12345678901 (histórico normal)
**Valor**: 100.0
**Horário**: 10:00 (horário comercial)
**Resultado esperado**: is_fraud = false

### Cenário 2: Transação Suspeita

**Endpoint**: POST /predict

**CPF**: 99999999999 (novo usuário)
**Valor**: 50000.0 (valor alto)
**Horário**: 23:00 (horário noturno)
**Resultado esperado**: is_fraud = true, is_cold_start = true

### Cenário 3: Money Mule

**Endpoint**: POST /predict

**CPF**: 11111111111 (muitas conexões)
**Valor**: 1000.0
**Destino**: CPF com muitas conexões
**Resultado esperado**: is_anomaly = true (grafo)

---

## Notas Importantes

1. **CPF Hashing**: Todos os CPFs são hasheados antes de serem armazenados (conformidade LGPD)
2. **Cold Start**: Usuários com < 30 transações são marcados como cold start
3. **Latência**: Behavioral profiling adiciona ~2ms à latência total
4. **Feedback**: O endpoint `/feedback/anomaly` permite que analistas refinam o modelo

---

## Suporte

Para dúvidas ou problemas, consulte:
- `README.md` do projeto
- `docs/INTERPRETADOR_REGRAS.md` (Rule Engine V2 detalhado)
- `docs/COMO_FUNCIONA.md` (visão não-técnica)
- Equipe de Engenharia de Dados — Banco Brasileiro
