# Fraud Detection API - Anti-Fraud v3

Sistema de detecção de fraude em tempo real para transações bancárias brasileiras (PIX, TED, Boleto, Autenticação) com baixa latência e conformidade regulatória (BACEN/LGPD).

## Versão

**Versão Atual**: 1.1.0  
**Data de Lançamento**: 25/04/2026  
**Última Atualização**: 25/04/2026

### Mudanças na Versão 1.1.0

- Adicionado Configuration Management (config.py)
- Implementado Repository Pattern (src/repositories.py)
- Implementado Dependency Injection
- Adicionado SHAP Explainer para explicabilidade
- Implementado Logging Auditável
- Atualizados testes para suportar explicações
- Melhorias arquiteturais (DDD, CDD)

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
- **Persistência**: Joblib 1.3+

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
1. Carrega o dataset expandido (60.000 amostras)
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
  "version": "1.1.0"
}
```

### POST /predict

Predição de fraude (transação única) com explicação SHAP

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

**Response**:
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

**Nota**: O campo `explanation` é preenchido apenas quando `is_fraud = true` e contém SHAP values.

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

## Métricas Atuais

| Métrica | Valor | Meta | Status |
|---------|-------|------|--------|
| Latência | 28ms | <100ms | 
| AUC-ROC | 0.8307 | >0.90 | 
| F1-Score | 0.2412 | >0.85 | 
| Dataset | 60.000 amostras | 100.000+ | 
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

### LGPD (Lei Geral de Proteção de Dados)

**Exigências Atendidas**:
- Proteção de dados sensíveis (CPF)
- Consentimento explícito
- Direito à explicação (SHAP)
- Minimização de dados

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
```

**Resultado**: 26 passed, 5 skipped

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
│   ├── feature_engineering.py    # Feature engineering
│   ├── model.py                  # Modelo XGBoost + SHAP
│   ├── repositories.py           # Repository Pattern
│   └── main.py                   # API FastAPI
├── tests/                        # Suíte de testes
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   ├── test_api.py
│   ├── test_performance.py
│   └── test_bdd_features.feature
├── models/                       # Modelos treinados
│   ├── fraud_model.pkl
│   └── feature_names.json
├── logs/                         # Logs de auditoria
│   └── audit.log
├── .github/workflows/            # CI/CD
│   └── ci.yml
├── dataset_transacoes.csv        # Dataset original (10k)
├── dataset_transacoes_expanded.csv  # Dataset expandido (60k)
├── generate_expanded_dataset.py
├── train_model.py
├── config.py                     # Configuration Management
├── requirements.txt
├── pytest.ini
├── README.md
├── AGENTS.md
└── docs/                         # Documentação
    ├── DOCUMENTACAO_COMPLETA.md
    └── RELATORIO_TECNICO.md
```

## Documentação

- [Documentação Completa](docs/DOCUMENTACAO_COMPLETA.md) - Detalhamento completo do sistema
- [Relatório Técnico](docs/RELATORIO_TECNICO.md) - Aspectos técnicos e arquiteturais
- [Análise de Dataset e Proposta de Modelo](analise_dataset_proposta_modelo.md)
- [Validação Final PM](validacao_final_pm.md)

## Próximos Passos

1. Aumentar dataset para 100k+ amostras
2. Implementar features de rede (graph analysis)
3. Adicionar dados externos (score de crédito)
4. Implementar autenticação na API
5. Deploy em produção com Kubernetes

## Licença

Confidencial - Uso interno

## Suporte

Equipe de Engenharia de Dados - Banco Brasileiro
