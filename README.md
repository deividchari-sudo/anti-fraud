# Fraud Detection API - Anti-Fraud v3

Sistema de detecção de fraude em tempo real para transações bancárias brasileiras (PIX, TED, Boleto, Autenticação) com baixa latência e conformidade regulatória (BACEN/LGPD).

## Versão

**Versão Atual**: 2.0.0
**Data de Lançamento**: 27/04/2026
**Última Atualização**: 27/04/2026

### Mudanças na Versão 2.0.0 (Rule Engine V2)

Evolução completa do interpretador de regras em linguagem natural, mantendo retrocompatibilidade total com V1.

**Persistência de regras**
- `RuleRepository` (ABC) + `JSONRuleRepository` (escrita atômica via `os.replace`)
- `InMemoryRuleRepository` para DI em testes
- Regras sobrevivem a reinícios da API (`data/rules.json`)

**Operadores lógicos OR e NOT**
- `Rule.condition_groups`: lista de grupos AND, conectados por OR
- `Condition.negated`: inverte resultado (NOT)
- Parser detecta `" ou "` como separador e `"exceto"`/`"não sendo"` como NOT

**Audit log BACEN**
- `RuleAuditLogger` em `logs/rule_audit.jsonl` (append-only JSONL)
- Cada match grava `timestamp`, `rule_id`, `transaction_id`, `action`, `priority`
- Imutável e timestamped — ingerível por ELK/Splunk/BigQuery

**Métricas operacionais por regra**
- `RuleMetricsRegistry`: `hit_count`, `last_match_at`, `false_positive_count`
- Identifica regras zumbis e ajuda calibração contra falsos positivos

**Detecção de conflitos**
- `detect_conflicts()` identifica pares com mesma assinatura e ações opostas
- Endpoint `GET /rules/conflicts`

**Novos endpoints REST**
- `POST /rules/validate` — dry-parse sem persistir
- `POST /rules/simulate` — dry-run em lote contra dataset
- `GET /rules/conflicts` — pares conflitantes
- `GET /rules/export` / `POST /rules/import` — portabilidade JSON
- `GET /rules/metrics` — métricas por regra
- `PATCH /rules/{rule_id}` — update parcial (priority, enabled, name, description)

**Cobertura de testes**
- +25 testes V2 em `tests/test_rule_engine_v2.py`
- Suite completa: **293 passed, 5 skipped, 0 errors** (era 268)
- Zero regressão nos 148 testes legados do rule engine

### Histórico de Sprints (1.6 → 1.9)

O histórico completo das 4 sprints (Ensemble, Stacking, Deep Learning, Federated+RL),
com métricas e sign-off da squad, está consolidado em
[`docs/CONCLUSAO_FINAL.md`](docs/CONCLUSAO_FINAL.md).

Resumo:

| Versão | Sprint | Entregas principais |
|---|---|---|
| 1.9.0 | Sprint 4 | Federated Learning (FedAvg+DP) + RL Threshold Adaptativo |
| 1.8.0 | Sprint 3 | AutoEncoder zero-day + SimpleGraphSAGE |
| 1.7.0 | Sprint 2 | Stacking (XGB+LGB+Cat→LR) + Open Finance + Re-treino agendado |
| 1.6.0 | Sprint 1 | Ensemble (XGB+LGB) + BaseFraudModel ABC + DI |
| 1.5.0 | Dataset | Expansão para 100k amostras (1% fraude) |
| 1.4.0 | — | DBSCAN, silhouette, auto-contamination, Ensemble IF |
| 1.3.0 | — | Behavioral Profiling V2 (LGPD hashing, SQLite, online learning) |
| 1.2.0 | — | Interpretador de Regras V1 (DSL português) |
| 1.1.0 | — | Repository Pattern + DI + SHAP + audit log |

## Arquitetura

```
┌─────────────────┐
│   Cliente API   │
└────────┬────────┘
         │ HTTP/JSON
         ▼
┌─────────────────┐
│  FastAPI (API)  │
│  - main.py      │
│  - config.py    │
└────────┬────────┘
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌─────────────────────┐
│ Rule Engine     │      │ Behavioral Profile  │
│ - parser.py     │      │ - service.py        │
│ - rule.py       │      │ - anomaly_detector.py│
│ - evaluator.py  │      │ - temporal_features  │
└────────┬────────┘      │ - clustering.py      │
         │              │ - isolation_forest   │
         │              │ - online_learning      │
         │              │ - graph_features      │
         │              └──────────┬────────────┘
         │                         │
         │                         ▼
         │              ┌─────────────────────┐
         │              │  UserProfile Repo   │
         │              │  (SQLite + Cache)    │
         │              └─────────────────────┘
         │
         ▼
┌─────────────────┐
│Feature Engineer │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ML Model (XGBoost)│
│  - SHAP Explainer│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Repositories   │
└─────────────────┘
```

## Stack Tecnológica

- **Linguagem**: Python 3.13
- **Framework API**: FastAPI 0.104+
- **ML Framework**: XGBoost 2.0+
- **Processamento de Dados**: Pandas, NumPy
- **Testes**: Pytest, Pytest-Cov
- **ASGI Server**: Uvicorn 0.24+
- **Configuration**: Pydantic Settings
- **Explicabilidade**: SHAP 0.44+
- **Persistência**: Joblib 1.3+, SQLite 3
- **Rule Engine**: DSL em Português (Regex-based)
- **Behavioral Profiling**: 
  - Scikit-learn (K-means, Isolation Forest)
  - NetworkX (análise de grafo)
  - Cryptography (SHA-256 hashing)

## Instalação

```bash
# Clone o repositório
git clone https://github.com/your-org/anti-fraud-v3-wf.git
cd anti-fraud-v3-wf

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt

# Treine o modelo
python train_model.py
```

## Treinamento do Modelo

```bash
python train_model.py
```

Este script:
1. Carrega o dataset expandido (100.000 amostras)
2. Extrai 71 features
3. Aplica SMOTE para balanceamento
4. Treina modelo XGBoost
5. Salva modelo em `models/fraud_model.pkl`

## Execução da API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

API estará disponível em http://localhost:8000

## Endpoints

### GET /health

Verifica status da API

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0"
}
```

### POST /predict

Predição de fraude (transação única) com explicação SHAP

**Priorização**: Regras são avaliadas antes do modelo ML. Se uma regra der match, o resultado é retornado imediatamente sem executar o modelo.

**Request**:
```json
{
  "payload": {
    "id": "test-id-123",
    "timestamp": "2026-04-16T09:55:11",
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

**Response (Rule-based)**:
```json
{
  "transaction_id": "test-id-123",
  "fraud_probability": 1.0,
  "is_fraud": true,
  "confidence": "high",
  "processing_time_ms": 5.2,
  "timestamp": "2026-04-25T19:01:29.539100",
  "explanation": {
    "type": "rule_based",
    "matched_rules": ["rule_1"],
    "reason": "Transaction matched one or more fraud rules"
  }
}
```

**Response (ML-based)**:
```json
{
  "transaction_id": "test-id-123",
  "fraud_probability": 0.0454,
  "is_fraud": false,
  "confidence": "low",
  "processing_time_ms": 28.25,
  "timestamp": "2026-04-25T19:01:29.539100",
  "explanation": null
}
```

**Nota**: 
- O campo `explanation` é preenchido quando `is_fraud = true`
- Se regras derem match, `explanation.type = "rule_based"`
- Se ML der match, `explanation` contém SHAP values

### POST /predict/batch

Predição em lote

**Response**:
```json
{
  "results": [...],
  "total_transactions": 2,
  "processing_time_ms": 45.12,
  "avg_time_per_transaction": 22.56
}
```

### GET /model/info

Informações do modelo

**Response**:
```json
{
  "model_type": "XGBoost",
  "feature_count": 71,
  "threshold": 0.5,
  "top_features": {...}
}
```

## Endpoints de Regras

O sistema possui um interpretador de regras em linguagem natural (português brasileiro) para usuários operacionais.

### POST /rules

Criar nova regra

**Request**:
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

### GET /rules

Listar todas as regras

**Response**:
```json
{
  "total_rules": 5,
  "rules": [...]
}
```

### GET /rules/{rule_id}

Obter regra específica

**Response**:
```json
{
  "rule_id": "rule_1",
  "name": "Regra CPF Suspeito",
  "description": "Texto original",
  "original_text": "Todo pix do CPF 12345678901 é fraude",
  "action": "mark_as_fraud",
  "conditions": [...],
  "enabled": true,
  "priority": 0
}
```

### DELETE /rules/{rule_id}

Deletar regra

**Response**:
```json
{
  "message": "Rule rule_1 deleted successfully"
}
```

### POST /rules/{rule_id}/enable

Habilitar regra

**Response**:
```json
{
  "rule_id": "rule_1",
  "enabled": true
}
```

### POST /rules/{rule_id}/disable

Desabilitar regra

**Response**:
```json
{
  "rule_id": "rule_1",
  "enabled": false
}
```

### POST /rules/evaluate

Avaliar transação contra regras

**Request**:
```json
{
  "payload": {
    "id": "test-id-123",
    "timestamp": "2026-04-16T09:55:11",
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

## Sintaxe de Regras

### Condições Suportadas

- **CPF**: `Todo pix do CPF 12345678901 é fraude`
- **Valor**: `Todos os pix com valor superior a 1000 reais é fraude`
- **Horário**: `Todos os pix depois das 22:00 é fraude`
- **Canal**: `Todos os pix do app é fraude`
- **Banco**: `Bloquear todas as transacoes do banco 001`

### Ações Suportadas

- **Fraude**: `é fraude`, `é suspeita`, `bloquear`
- **Legítimo (Whitelist)**: `não é fraude`, `nao é fraude`

### Combinações

- `Todo pix do CPF 12345678901 com valor superior a 1000 reais depois das 22:00 é fraude`
- `Todos os pix do app do CPF 98765432100 não é fraude`

Para mais detalhes, veja [Documentação do Interpretador de Regras](docs/INTERPRETADOR_REGRAS.md).

## Endpoints de Behavioral Profiling

### GET /profile/{cpf}

Obter perfil comportamental do usuário

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
  "destinations": {...},
  "canais": {...},
  "produtos": {...},
  "temporal_features": {
    "sliding_windows": {...},
    "trends": {...},
    "seasonal": {...}
  },
  "cluster_id": 2
}
```

### POST /feedback/anomaly

Enviar feedback de analista sobre detecção de anomalias

**Request**:
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

## Behavioral Profiling

O sistema inclui análise de perfil comportamental para detecção de anomalias baseada no histórico de transações do usuário.

### Funcionalidades

**1. Perfis de Usuário**
- Estatísticas de valor (média, desvio padrão, percentis)
- Padrões de horário (hora do dia, dia da semana)
- Frequência de transações
- Distribuição de destinos, canais e produtos
- Cold start detection (< 30 transações)

**2. Detecção de Anomalias**
- Anomalias de valor (z-score, percentil)
- Anomalias de horário (fora do padrão habitual)
- Anomalias de destino (novo CPF/banco)
- Anomalias de canal (canal não habitual)

**3. Features Temporais**
- Janelas deslizantes (7, 30, 90 dias)
- Análise de tendência (crescente/decrescente)
- Features sazonais (dia da semana, hora do dia)
- Padrões de frequência ao longo do tempo

**4. Clustering de Usuários**
- K-means com 5 clusters comportamentais
- Segmentação automática de usuários
- Detecção de anomalias relativa ao cluster
- Descrições interpretáveis (ex: "high_value_night_user")

**5. Isolation Forest**
- Detecção de outliers multivariados
- 14 features extraídas de transações
- Integração com contexto de perfil do usuário

**6. Aprendizado Online**
- Perfis adaptativos com médias exponenciais móveis
- Threshold adaptativo com feedback de analistas
- Detecção de concept drift (mudanças de padrão)
- Forgetting factor para ponderar dados antigos

**7. Features de Grafo**
- Análise de rede de conexões entre CPFs
- Features: degree, clustering coefficient, PageRank
- Detecção de money mules (high-degree nodes)
- Detecção de transações circulares

### Scripts de Backfill

**Gerar Perfis do Dataset**
```bash
python scripts/backfill_profiles.py --min-transactions 1
```

**Gerar Perfil Global (Cold Start)**
```bash
python scripts/generate_global_profile.py
```

**Treinar Clustering**
```bash
python -c "
from src.user_profile import UserProfileService
from src.repositories import SQLiteUserProfileRepository

repo = SQLiteUserProfileRepository()
service = UserProfileService(repo)
result = service.train_clustering()
print(result)
"
```

**Treinar Isolation Forest**
```bash
python -c "
from src.user_profile import UserProfileService
from src.repositories import SQLiteUserProfileRepository
import json

repo = SQLiteUserProfileRepository()
service = UserProfileService(repo)

# Load transactions from dataset
with open('dataset_transacoes_expanded.csv', 'r') as f:
    # Parse and load transactions
    pass

result = service.train_multivariate_anomaly_detector(transactions)
print(result)
"
```

## Métricas Atuais

| Métrica | Valor | Meta | Status |
|---------|-------|------|--------|
| Latência | 28ms | <100ms | ✅ |
| AUC-ROC | 0.8395 | >0.90 | ⚠️ |
| F1-Score (threshold 0.5) | 0.1375 | >0.85 | ⚠️ |
| F1-Score (threshold ótimo 0.889) | 0.2778 | >0.85 | ⚠️ |
| Recall (Fraude) | 0.5950 | >0.85 | ⚠️ |
| Dataset | 100.000 amostras | 100.000+ | ✅ |
| Explicabilidade | SHAP | Obrigatório | 
| Logging Auditável | 100% | 100% | 

## Features Implementadas (71 total)

### Features Básicas
- `valor`, `sender_banco`, `sender_agencia`, `sender_conta`
- `receiver_banco`, `receiver_agencia`, `receiver_conta`
- `canal`, `produto`, `jornada`, `direcao`

### Features Temporais
- `hora_do_dia`, `dia_da_semana`, `fim_de_semana`, `horario_noturno`
- `hora_sin`, `hora_cos`, `dia_semana_sin`, `dia_semana_cos` (encoding cíclico)
- `dia_do_mes`, `mes_do_ano`, `inicio_mes`, `fim_mes`

### Features de Valor
- `valor_log`, `valor_maior_1000`, `valor_maior_5000`, `valor_maior_10000`
- `valor_zscore`
- `faixa_valor_baixa`, `faixa_valor_media`, `faixa_valor_alta`, `faixa_valor_muito_alta`

### Features de Produto/Canal
- `is_pix`, `is_ted`, `is_boleto`, `is_autenticacao`
- `is_app`, `is_web`, `is_api`
- `is_login`, `is_transferencia`, `is_pix_troco`, `is_pix_saque`, `is_estorno`

### Features Comportamentais
- `transacoes_ultimas_1h`, `transacoes_ultimas_24h`
- `valor_total_ultimas_24h`, `valor_medio_ultimas_24h`
- `nova_relacao`, `dispositivo_distinto`

### Features de Geolocalização
- `cross_border`, `mesmo_banco`, `banco_diferente_sender`

### Features de Rede
- `grau_sender`, `grau_receiver`

### Features de Padrões de Fraude
- `horario_atipico`, `transacao_fora_horario_comercial`
- `valor_atipico`, `multiplos_dispositivos`

## Conformidade Regulatória

### BACEN (Banco Central do Brasil)

**Exigências Atendidas**:
- Detecção de fraude em tempo real
- Rastreabilidade de decisões (logs/audit.log)
- Auditoria algorítmica (SHAP values)
- Latência <100ms
- Taxa de detecção >95% (em progresso)
- Detecção de money mules e transações circulares

### LGPD (Lei Geral de Proteção de Dados)

**Exigências Atendidas**:
- Proteção de dados sensíveis (CPF) com SHA-256 + salt
- CPFs não armazenados em plaintext
- Consentimento explícito
- Direito à explicação (SHAP)
- Minimização de dados
- Hashing reversível apenas para analistas autorizados
- Log de acesso a dados sensíveis para auditoria

## Logging Auditável

Todas as predições são logadas em `logs/audit.log` com:

- transaction_id
- timestamp
- fraud_probability
- is_fraud
- threshold
- explanation (SHAP values quando is_fraud = true)

## Configuration Management

Configurações centralizadas em `config.py`:

```python
from config import settings

# Acessar configurações
model_path = settings.model_path
api_port = settings.api_port
threshold = settings.threshold
```

Variáveis de ambiente suportadas via `.env`:
```bash
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/fraud_model.pkl
THRESHOLD=0.5
```

## Repository Pattern

Implementado em `src/repositories.py`:

- `TransactionRepository`: Abstração para acesso a dados
- `CSVTransactionRepository`: Implementação CSV
- `ModelRepository`: Abstração para persistência de modelo
- `JoblibModelRepository`: Implementação Joblib

Benefícios:
- Baixo acoplamento
- Testabilidade (mock fácil)
- Dependency Injection

## Testes

```bash
# Executar todos os testes
pytest

# Executar com coverage
pytest --cov=src --cov-report=html

# Executar testes específicos
pytest tests/test_model.py -v
pytest tests/test_api.py -v
pytest tests/test_performance.py -v
pytest tests/test_rule_engine.py -v
pytest tests/test_rule_engine_comprehensive.py -v
pytest tests/test_rule_engine_v2.py -v
```

**Resultado atual**: 293 passed, 5 skipped (V2.0.0)

## Performance

| Operação | Latência Média | Limite | Status |
|----------|----------------|--------|--------|
| Feature Extraction | ~15ms | <50ms | 
| Predição (single) | ~13ms | <100ms | 
| Predição (batch) | ~22ms/transação | <100ms | 
| Total (single) | 28ms | <100ms | 

## Estrutura de Diretórios

```
anti-fraud-v3-wf/
├── src/                          # Código fonte
│   ├── __init__.py
│   ├── models.py                 # Modelos Pydantic
│   ├── crypto.py                 # Hashing de CPFs (LGPD)
│   ├── feature_engineering.py    # Feature engineering
│   ├── base_model.py             # BaseFraudModel (ABC) - lógica compartilhada
│   ├── model.py                  # Modelo XGBoost + SHAP
│   ├── ml_mixins.py              # ThresholdTuningMixin + SHAPExplainerMixin
│   ├── ensemble_model.py         # EnsembleFraudModel (XGBoost + LightGBM)
│   ├── stacking_model.py         # StackingFraudModel (XGB + LGB + CatBoost -> LR)
│   ├── autoencoder_anomaly.py    # AutoEncoderAnomalyDetector (zero-day)
│   ├── simple_graph_sage.py      # SimpleGraphSAGE (graph embeddings)
│   ├── federated_aggregator.py   # FederatedAggregator (FedAvg + DP)
│   ├── rl_threshold.py           # EpsilonGreedyThresholdSelector (RL)
│   ├── open_finance_features.py  # OpenFinanceFeatureExtractor (9 features)
│   ├── repositories.py           # Repository Pattern (SQLite + JSON + Ensemble + Anomaly)
│   ├── rule_engine/              # Interpretador de Regras V2
│   │   ├── __init__.py
│   │   ├── parser.py             # Parser de linguagem natural (OR/NOT)
│   │   ├── rule.py               # Estruturas de dados (negated, condition_groups)
│   │   ├── evaluator.py          # Avaliador com persistência, métricas, audit
│   │   ├── repository.py         # JSONRuleRepository / InMemoryRuleRepository
│   │   ├── audit.py              # RuleAuditLogger (BACEN JSONL)
│   │   ├── metrics.py            # RuleMetricsRegistry
│   │   └── conflicts.py          # detect_conflicts()
│   ├── user_profile/             # Behavioral Profiling
│   │   ├── __init__.py
│   │   ├── models.py             # Modelos Pydantic para perfil
│   │   ├── service.py           # UserProfileService
│   │   ├── anomaly_detector.py   # Detecção de anomalias
│   │   ├── temporal_features.py # Features temporais
│   │   ├── clustering.py        # Clustering de usuários
│   │   ├── isolation_forest.py  # Isolation Forest
│   │   ├── online_learning.py   # Aprendizado online
│   │   └── graph_features.py     # Features de grafo
│   └── main.py                   # API FastAPI
├── scripts/                      # Scripts de backfill e utilitários
│   ├── backfill_profiles.py      # Gerar perfis do dataset
│   └── generate_global_profile.py # Gerar perfil global
├── tests/                        # Suíte de testes
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   ├── test_api.py
│   ├── test_performance.py
│   ├── test_rule_engine.py       # 38 testes do rule engine
│   ├── test_rule_engine_comprehensive.py  # 110 testes abrangentes
│   ├── test_user_profile_service.py
│   ├── test_anomaly_detector.py
│   └── test_json_repository.py
├── data/                         # Dados
│   ├── user_profiles.db          # SQLite para perfis
│   ├── user_profiles.json        # JSON para perfis (legado)
│   ├── global_profile.json       # Perfil global (cold start)
│   └── anomaly_feedback.json     # Feedback de analistas
├── models/                       # Modelos treinados
│   ├── fraud_model.pkl
│   └── feature_names.json
├── logs/                         # Logs de auditoria
│   └── audit.log
├── .github/workflows/            # CI/CD
│   └── ci.yml
├── dataset_transacoes.csv        # Dataset original (10k)
├── dataset_transacoes_expanded.csv  # Dataset expandido (100k)
├── generate_expanded_dataset.py
├── train_model.py
├── config.py                     # Configuration Management
├── requirements.txt
├── pytest.ini
├── README.md
├── AGENTS.md
└── docs/                         # Documentação
    ├── COMO_FUNCIONA.md          # Explicação não-técnica para áreas de negócio
    ├── INTERPRETADOR_REGRAS.md   # Documentação V2 do interpretador de regras
    ├── CONCLUSAO_FINAL.md        # Histórico das Sprints 1-4
    └── POSTMAN_COLLECTION.md     # Exemplos de payload por endpoint
```

## Documentação

- [Como Funciona (não-técnica)](docs/COMO_FUNCIONA.md) — explicação acessível para áreas de negócio
- [Interpretador de Regras V2](docs/INTERPRETADOR_REGRAS.md) — DSL em PT-BR, persistência, OR/NOT, audit, métricas
- [Conclusão das Sprints 1-4](docs/CONCLUSAO_FINAL.md) — histórico técnico completo da evolução
- [Postman Collection](docs/POSTMAN_COLLECTION.md) — exemplos de payload para todos os endpoints

## Próximos Passos

### Concluídos na V1.3.0 ✅
- ✅ Implementar features de rede (graph analysis)
- ✅ Implementar clustering de usuários
- ✅ Implementar features temporais (janelas deslizantes)
- ✅ Implementar Isolation Forest para outliers multivariados
- ✅ Implementar aprendizado online com modelos adaptativos
- ✅ Implementar hashing de CPFs para conformidade LGPD
- ✅ Otimizar backend com SQLite

### Próximos Passos Futuros
1. Aumentar dataset para 100k+ amostras
2. Adicionar dados externos (score de crédito)
3. Implementar autenticação na API
4. Implementar Redis para cache distribuído (escala horizontal)
5. Implementar RNN/LSTM para padrões sequenciais
6. Deploy em produção com Kubernetes
7. Validação com dados reais em produção
8. Treinamento de analistas para feedback loop

## Licença

Confidencial - Uso interno

## Suporte

Equipe de Engenharia de Dados - Banco Brasileiro
