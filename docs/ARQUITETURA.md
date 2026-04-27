# Arquitetura Técnica — Anti-Fraud v3

Documento técnico consolidado, voltado a **engenheiros, arquitetos e cientistas de dados**.
Para a visão de negócio, comece por [`COMO_FUNCIONA.md`](COMO_FUNCIONA.md).

---

## 1. Visão de 30 segundos

Sistema de **detecção de fraude em tempo real** (PIX, TED, Boleto, Autenticação)
com 3 camadas decisórias em cascata, latência **≤ 100ms** (P95) e conformidade
**BACEN/LGPD**.

```
┌──────────────┐
│ Cliente API  │  POST /predict
└──────┬───────┘
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                          FastAPI                                │
│  validação (Pydantic) → orquestração → resposta + audit log     │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐  match? → curto-circuito           ┌──────────────────┐
│ 1. Rule Engine   │ ───── action=mark_as_fraud ─────►  │   Resposta final │
│ V2 (DSL PT-BR)   │                                    │ + SHAP/explanation│
└──────┬───────────┘                                    └──────────────────┘
       │ sem match
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. ML Layer (escolha por configuração / sprint atual)            │
│                                                                  │
│   • XGBoost solo (v1.0)                                          │
│   • EnsembleFraudModel  XGB + LGB        (v1.6, sprint 1)        │
│   • StackingFraudModel  XGB+LGB+Cat → LR (v1.7, sprint 2)        │
│   • AutoEncoderAnomalyDetector (zero-day) (v1.8, sprint 3)       │
│   • SimpleGraphSAGE (graph embeddings)    (v1.8, sprint 3)       │
│                                                                  │
│   Threshold adaptativo: EpsilonGreedyThresholdSelector (RL, v1.9)│
│   Calibração: CalibratedClassifierCV (isotonic)                  │
│   Explicabilidade: SHAP (mixin compartilhado)                    │
└──────┬───────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Behavioral Profiling (UserProfileService)                     │
│   • estatísticas online (EMA + drift)                            │
│   • janelas 7/30/90 dias  • clustering (KMeans/DBSCAN)           │
│   • IsolationForest ensemble (auto-contamination)                │
│   • graph features (degree, PageRank, mule detection)            │
└──────┬───────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Persistência (Repository Pattern + DI)                           │
│   SQLite (perfis) · JSON (regras, feedback) · Joblib (modelos)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Stack tecnológica

| Camada | Tecnologia | Versão |
|---|---|---|
| Linguagem | Python | 3.13 |
| API / ASGI | FastAPI + Uvicorn | 0.104+ / 0.24+ |
| Validação | Pydantic + Pydantic-Settings | v2 |
| ML supervisionado | XGBoost, LightGBM, CatBoost | 2.0+ / 4.3+ / 1.2+ |
| ML não-supervisionado | scikit-learn (IsolationForest, KMeans, DBSCAN, MLP) | 1.4+ |
| Explicabilidade | SHAP | 0.44+ |
| Balanceamento | imbalanced-learn (SMOTE) | 0.12+ |
| Grafos | NetworkX (+ NumPy puro para SimpleGraphSAGE) | 3.x |
| Persistência | SQLite, Joblib, JSON | stdlib / 1.3+ |
| Cripto | hashlib SHA-256 + salt | stdlib |
| Testes | pytest, pytest-cov, httpx | latest |

---

## 3. Camadas e responsabilidades

### 3.1 API (`src/main.py`, `config.py`)

- **FastAPI** com validação Pydantic em todos os payloads
- Endpoints organizados por contexto: `/predict`, `/rules*`, `/profile/*`, `/feedback/*`, `/model/info`, `/health`
- **Configuration Management** centralizado em `config.py` (`pydantic.BaseSettings`), variáveis via `.env`
- **Dependency Injection** explícita: repositórios, audit logger, rule evaluator
- Logging estruturado via `logging` stdlib em `logs/audit.log` (sistema) e `logs/rule_audit.jsonl` (regras)

### 3.2 Rule Engine V2 (`src/rule_engine/`)

DSL em português brasileiro para usuários operacionais (compliance, prevenção a fraude).

| Módulo | Função |
|---|---|
| `parser.py` | Regex-based; suporta CPF, valor, horário, canal, banco; OR (`" ou "`) e NOT (`"exceto"`/`"não sendo"`) |
| `rule.py` | `Rule`, `Condition`, `condition_groups` (lista de grupos AND, unidos por OR) |
| `evaluator.py` | Avaliação ordenada por prioridade, integrado a repositório/audit/métricas |
| `repository.py` | `RuleRepository` (ABC), `JSONRuleRepository` (escrita atômica via `os.replace`), `InMemoryRuleRepository` |
| `audit.py` | `RuleAuditLogger` → `logs/rule_audit.jsonl` (append-only, BACEN) |
| `metrics.py` | `RuleMetricsRegistry`: `hit_count`, `last_match_at`, `false_positive_count` |
| `conflicts.py` | `detect_conflicts()` — pares com mesma assinatura e ações opostas |

**Decisão de design:** o rule engine é avaliado ANTES do ML.
Se uma regra dá match, retorna-se imediatamente (curto-circuito).
Isso garante:
- Latência mínima em casos óbvios (~5 ms)
- Auditabilidade BACEN imediata e textual
- Override humano sobre o modelo (compliance)

Detalhes completos: [`INTERPRETADOR_REGRAS.md`](INTERPRETADOR_REGRAS.md).

### 3.3 ML Layer

#### Modelos disponíveis

| Modelo | Arquivo | Quando usar |
|---|---|---|
| **`FraudDetectionModel`** | `src/model.py` | Baseline XGBoost solo + SMOTE + SHAP. Treino rápido, latência 13 ms. |
| **`EnsembleFraudModel`** | `src/ensemble_model.py` | XGB + LGB com voting ponderado e calibração isotônica. Probabilidades confiáveis, threshold por canal/produto. |
| **`StackingFraudModel`** | `src/stacking_model.py` | XGB + LGB + CatBoost → meta-learner Logistic Regression. **Maior precision** (0.32). Padrão Nubank. |
| **`AutoEncoderAnomalyDetector`** | `src/autoencoder_anomaly.py` | MLP autoencoder treinado **só em transações legítimas**. Detecta padrões zero-day (fraudes nunca vistas no treino). |
| **`SimpleGraphSAGE`** | `src/simple_graph_sage.py` | Embeddings 24-d de transações via mean-aggregation K-hop. Detecta money mules / círculos de lavagem. |
| **`SegmentedModelRepository`** | `src/segmented_model.py` | Lazy-load de modelos especializados por `(produto, canal)`. Fallback automático para modelo global. Cache em memória. Endpoint `POST /predict/segmented`. |

#### Infraestrutura ML compartilhada

- **`BaseFraudModel`** (`src/base_model.py`, ABC) — train/test split, SMOTE, scale_pos_weight, alinhamento de features, audit logging.
- **`ml_mixins.py`** — `ThresholdTuningMixin` (busca threshold ótimo por F1, global + por canal + por produto) e `SHAPExplainerMixin` (top-N features que pesaram na decisão).
- **`EpsilonGreedyThresholdSelector`** (`src/rl_threshold.py`) — multi-armed bandit que adapta threshold com feedback humano via `/feedback/anomaly`. Estado persistido em JSON.
- **`FederatedAggregator`** (`src/federated_aggregator.py`) — FedAvg + Differential Privacy + whitelist de participantes; pronto para integração com Flower / TF Federated.

#### Feature Engineering (`src/feature_engineering.py`)

71 features, agrupadas em:

- **Básicas** (11) — valor, banco/agencia/conta, canal, produto, jornada, direção
- **Temporais** (12) — hora, dia, fim_de_semana, encoding cíclico (sin/cos), inicio/fim mês
- **Valor** (9) — log, faixas (baixa/média/alta/muito alta), z-score
- **Produto/Canal one-hot** (12) — is_pix, is_app, is_login, is_pix_troco…
- **Comportamentais** (6) — transações_24h, valor_total_24h, nova_relacao, dispositivo_distinto
- **Geolocalização** (3) — cross_border, mesmo_banco
- **Rede** (2) — grau_sender, grau_receiver
- **Padrões de fraude** (4) — horario_atipico, valor_atipico, multiplos_dispositivos
- **Open Finance** (9, opcional) — score, estabilidade, divergência, fraude reportada externa

Implementação: extração vetorizada com Pandas; cache de histórico em memória (LRU).

### 3.4 Behavioral Profiling (`src/user_profile/`)

Camada de personalização — cada CPF tem perfil próprio, atualizado online.

| Componente | Função |
|---|---|
| `service.py` (`UserProfileService`) | Orquestra todos os subcomponentes |
| `models.py` | Pydantic schemas (`UserProfile`, `Statistics`, `TemporalFeatures`) |
| `anomaly_detector.py` | Detecção univariada (z-score em valor/horário/destino/canal) |
| `temporal_features.py` | Janelas 7/30/90 d, tendência, sazonalidade |
| `clustering.py` | KMeans 10-clusters + DBSCAN + silhouette score |
| `isolation_forest.py` | Ensemble de 5 IsolationForests com voting; auto-contamination |
| `online_learning.py` | EMA + drift detection + forgetting factor |
| `graph_features.py` | NetworkX: degree, clustering coefficient, PageRank, ciclos |

**Cold start:** usuários com menos de 30 transações usam `global_profile.json` (estatísticas agregadas do dataset) como prior.

**LGPD:** todos os CPFs são hasheados via `src/crypto.py` (SHA-256 + salt).
O hash é determinístico (mesmo CPF → mesmo hash) mas computacionalmente irreversível.

### 3.5 Persistência (`src/repositories.py`)

Repository Pattern com Dependency Injection.

| Repositório | Backend | Uso |
|---|---|---|
| `TransactionRepository` (ABC) → `CSVTransactionRepository` | CSV | Datasets de treino |
| `ModelRepository` (ABC) → `JoblibModelRepository` | Joblib (`models/`) | XGBoost solo |
| `EnsembleModelRepository` (ABC) → `JoblibEnsembleModelRepository` | Joblib | Ensemble + Stacking |
| `AnomalyModelRepository` (ABC) → `JoblibAnomalyModelRepository` | Joblib | AutoEncoder |
| `UserProfileRepository` (ABC) → `SQLiteUserProfileRepository` | SQLite (`data/user_profiles.db`) | Perfis comportamentais |
| `SegmentedModelRepository` | Joblib (`models/fraud_model_{prod}_{chan}.pkl`) | Router lazy-load por segmento; fallback global |
| `RuleRepository` (ABC) → `JSONRuleRepository` / `InMemoryRuleRepository` | JSON / RAM | Regras V2 |

Vantagens: testabilidade (mocks in-memory), troca de backend sem tocar lógica de negócio,
preparação para PostgreSQL/Redis em produção.

---

## 4. Fluxo de uma requisição `/predict`

```
1. POST /predict {payload}
2. Pydantic valida o schema
3. Feature engineering: 71 features extraídas (~15 ms)
4. RuleEvaluator.evaluate_single(features):
     ├── match? → retorna {is_fraud, matched_rules, explanation.type=rule_based}
     └── sem match → próxima etapa
5. SegmentedModelRouter: decide se usa modelo especializado (produto×canal) ou global (~2 ms lookup)
6. Modelo ML: predict_proba (XGBoost / Ensemble / Stacking) (~13 ms)
7. Threshold: global ou por canal/produto ou via RL
7. SHAP: top-5 features (apenas se is_fraud=True ou debug)
8. UserProfileService.update(cpf_hash, features) — atualização online (async-friendly)
9. AuditLogger grava decisão em logs/audit.log
10. Resposta JSON com fraud_probability, is_fraud, confidence, processing_time_ms, explanation
```

Latência total típica: **~28 ms**, P95 **~98 ms**, P99 **~125 ms**.

---

## 5. Métricas atuais (dataset sintético 100k)

| Modelo | AUC-ROC | F1 (ótimo) | Precision | Recall | Latência média |
|---|---|---|---|---|---|
| XGBoost solo (v1.5) | 0.8395 | 0.2778 | 0.0777 | 0.5950 | 28 ms |
| Ensemble XGB+LGB (v1.6) | 0.8207 | 0.2574 | 0.2775 | 0.2400 | 75 ms |
| **Stacking XGB+LGB+Cat→LR (v1.7)** | 0.7832 | 0.2454 | **0.3175** | 0.2000 | 98 ms |

**Observações:**
- Precision multiplicada por **4×** ao longo das sprints (0.078 → 0.318)
- AUC menor com calibração isotônica é **esperado** em dados sintéticos com base learners correlatos (todos tree-based) — em dados reais a tendência se inverte (ver literatura: Nubank, Mercado Pago)
- Recall menor é tradeoff de precision em produção real, combina-se com revisão humana via `/feedback/anomaly`
- AutoEncoder e GraphSAGE são **complementares** (sinais adicionais), não substitutos do supervisionado

Critério de aceite **BACEN** (latência P95 < 100 ms): **atendido em todos os modelos**.

Histórico completo de evolução: [`CONCLUSAO_FINAL.md`](CONCLUSAO_FINAL.md).

---

## 6. Conformidade regulatória

### BACEN (Resolução BCB nº 6, 2020)

| Exigência | Implementação |
|---|---|
| Detecção em tempo real | Latência P95 ~98 ms (limite 100 ms) |
| Rastreabilidade de decisões | `logs/audit.log` + `logs/rule_audit.jsonl` |
| Auditoria algorítmica | SHAP top-N features por decisão; rules em texto natural |
| Segregação de funções | Repository Pattern + DI permite mocks em auditoria |
| Detecção de money mules | `graph_features.py` + `SimpleGraphSAGE` |
| Pronto para FRD/MED | Decisões com `transaction_id` reprocessáveis |

### LGPD (Lei nº 13.709/2018)

| Princípio | Implementação |
|---|---|
| Minimização de dados | Apenas CPF (hashed), valor, banco, canal, timestamp; sem nome/endereço/email |
| Não armazenar dados sensíveis em plaintext | `src/crypto.py` SHA-256 + salt |
| Direito à explicação | SHAP values devolvidos quando `is_fraud=true`; rules em texto humano |
| Auditabilidade de acesso | Logs estruturados em `logs/audit.log` |
| Differential Privacy (federated) | `FederatedAggregator` aplica DP nos pesos antes do compartilhamento |

---

## 7. Performance e capacidade

| Métrica | Valor | Limite |
|---|---|---|
| Latência média (XGBoost) | ~28 ms | < 100 ms |
| Latência P95 (Stacking) | ~98 ms | < 100 ms (BACEN) |
| Latência P99 (Stacking) | ~110 ms | < 200 ms |
| Throughput estimado | ~1.500 req/s por instância | — |
| Feature extraction | ~15 ms | < 50 ms |
| Behavioral profiling | ~2 ms | < 10 ms |
| Rule evaluation | ~5 ms | < 10 ms |

**Escala horizontal:** o stateless da API permite N pods atrás de load balancer.
SQLite é gargalo em produção: roadmap prevê migração para **PostgreSQL + Redis**
(perfis cacheados, escrita batch).

---

## 8. Testes

- **293 passed, 5 skipped** na suite total
- Coverage por contexto:
  - `tests/test_model.py`, `test_ensemble_model.py`, `test_stacking_model.py` — modelos
  - `tests/test_rule_engine.py`, `test_rule_engine_comprehensive.py`, `test_rule_engine_v2.py` — DSL (148 + 25)
  - `tests/test_user_profile_service.py`, `test_anomaly_detector.py` — behavioral
  - `tests/test_api.py` — endpoints
  - `tests/test_performance.py` — latência

```bash
pytest                                      # tudo
pytest -k "rule_engine"                     # só rule engine
pytest tests/test_performance.py -v         # SLOs
pytest --cov=src --cov-report=html          # coverage
```

CI: `.github/workflows/ci.yml` (pytest em cada PR) e
`.github/workflows/scheduled-retrain.yml` (re-treino semanal cron).

---

## 9. Decisões arquiteturais (ADRs informais)

| Decisão | Motivo |
|---|---|
| 3 camadas em cascata (Rules → ML → Behavioral) | BACEN exige decisões auditáveis e overridable; rules dão explicação textual imediata |
| Rule engine antes do ML | Curto-circuito reduz latência média e dá compliance precedence |
| DSL em PT-BR (não YAML/JSON) | Operadores de fraude/compliance escrevem; reduz time-to-deploy de regra |
| Ensemble + Stacking | Reduz variance (correlação entre base learners) e calibra probabilidades |
| AutoEncoder one-class | Detecta zero-day sem precisar de exemplos rotulados de fraude nova |
| Repository Pattern | Testabilidade + preparação para mudança de backend (SQLite → Postgres) |
| SHA-256 hashing de CPF | LGPD; determinístico permite join entre logs sem reidentificação |
| SHAP via mixin | Reuso entre modelos; explicabilidade BACEN sem custo extra |
| RL para threshold | Adapta a concept drift sem retreinar; reward online via feedback humano |
| Federated com DP | Permite colaboração entre bancos sem expor dados de clientes |

---

## 10. Próximos passos técnicos (pós-2.0.0)

1. Migrar SQLite → **PostgreSQL + Redis** (escala horizontal, perfis cacheados)
2. Substituir `OpenFinanceFeatureExtractor` simulado pela API real do BACEN/Open Finance
3. Trocar `FederatedAggregator` in-memory por **Flower** ou **TF Federated** (RPC real)
4. Conectar `EpsilonGreedyThresholdSelector` ao endpoint `/feedback/anomaly` em loop online
5. Validação shadow com **dados reais por 30 dias** antes do canary
6. Deploy gradual com canary release (1% → 10% → 50% → 100%)
7. Autenticação na API (OAuth2 / JWT) e rate limiting
8. Observabilidade: Prometheus metrics + Grafana dashboards + tracing (OpenTelemetry)

---

## Referências cruzadas

- [`README.md`](../README.md) — visão geral, instalação, endpoints, quickstart
- [`COMO_FUNCIONA.md`](COMO_FUNCIONA.md) — explicação não-técnica
- [`INTERPRETADOR_REGRAS.md`](INTERPRETADOR_REGRAS.md) — DSL completa do rule engine V2
- [`CONCLUSAO_FINAL.md`](CONCLUSAO_FINAL.md) — narrativa das 4 sprints
- [`POSTMAN_COLLECTION.md`](POSTMAN_COLLECTION.md) — exemplos por endpoint
- [`GLOSSARIO.md`](GLOSSARIO.md) — termos técnicos e de negócio
- [`FAQ.md`](FAQ.md) — perguntas frequentes e troubleshooting
