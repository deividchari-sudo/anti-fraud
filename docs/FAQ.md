# FAQ e Troubleshooting

Perguntas frequentes e soluções para problemas comuns. Dividido por **audiência**.

- [Para quem é de Negócio / Compliance / Jurídico](#para-quem-é-de-negócio--compliance--jurídico)
- [Para quem é de Dados / Modelagem](#para-quem-é-de-dados--modelagem)
- [Para quem vai operar a API](#para-quem-vai-operar-a-api)
- [Troubleshooting](#troubleshooting)

---

## Para quem é de Negócio / Compliance / Jurídico

### O que esse sistema faz, em uma frase?

Detecta fraude em transações financeiras (PIX, TED, Boleto, Login) **antes** que
o dinheiro saia da conta, em menos de 100 milissegundos, com decisão auditável e
explicável (BACEN/LGPD).

### Como ele decide?

Em três camadas, em cascata:
1. **Regras** escritas em português (criadas por analistas de fraude).
2. **Modelo de IA** treinado em 100 mil transações históricas.
3. **Perfil comportamental** de cada cliente (online, adapta no tempo).

Detalhes não-técnicos com analogias: [`COMO_FUNCIONA.md`](COMO_FUNCIONA.md).

### Como o BACEN é atendido?

- Latência P95 abaixo de 100 ms (exigência da Resolução BCB nº 6).
- **Toda decisão fica registrada** em `logs/audit.log` (sistema) e `logs/rule_audit.jsonl` (regras), com `transaction_id`, timestamp, motivo.
- **Decisões são explicáveis** via SHAP (top features que pesaram) e regras em texto natural.
- Detecta money mules e padrões circulares (lavagem) por análise de grafo.

### E a LGPD?

- CPFs **nunca** ficam em texto puro: passam por hash SHA-256 + salt.
- Coletamos só o mínimo necessário (CPF, valor, banco, canal, timestamp) — sem nome, endereço ou e-mail.
- O cliente tem **direito à explicação** (art. 20): respondemos com SHAP values e regra disparada.
- Logs registram acessos a dados sensíveis para auditoria.
- Em modo Federated (futuro), aplicamos Differential Privacy nos pesos compartilhados.

### Posso saber por que uma transação foi bloqueada?

Sim. Ao chamar `/predict`, a resposta inclui o campo `explanation`:
- Se foi por **regra**: `{"type": "rule_based", "matched_rules": [...], "reason": "..."}` — texto legível por humano.
- Se foi por **modelo**: `{"shap_values": {...}}` — top features e contribuição numérica.

### Quem cria as regras?

Operadores de fraude / compliance. As regras são escritas em **português natural**, por exemplo:
- `"Todo pix do CPF 12345678901 é fraude"`
- `"Todos os pix com valor superior a 10000 reais depois das 22:00 é suspeita"`

Detalhes em [`INTERPRETADOR_REGRAS.md`](INTERPRETADOR_REGRAS.md).

### O que acontece se errar?

- **Falso positivo** (legítimo bloqueado): cliente reclama → analista marca via `/feedback/anomaly` → modelo aprende.
- **Falso negativo** (fraude que passou): registrado para análise pós-evento → modelo retreina semanalmente; PIX tem **MED** (Mecanismo Especial de Devolução) por até 80 dias.

### Vocês usam dados reais?

Hoje (v2.0.0) **não** — usamos um dataset sintético de 100k transações para
desenvolvimento e testes. Em produção, a calibração será refeita com dados
reais durante o piloto shadow (30 dias).

### Posso ver as métricas?

Sim, três caminhos:
- `/health` para status da API
- `/model/info` para info do modelo carregado
- `/rules/metrics` para `hit_count` por regra
- `logs/audit.log` e `logs/rule_audit.jsonl` para auditoria

---

## Para quem é de Dados / Modelagem

### Qual modelo está em produção?

A configuração padrão usa **XGBoost solo** (baseline, latência mínima).
Existem outros modelos prontos para troca via repositório:
- `EnsembleFraudModel` (XGB + LGB calibrado)
- `StackingFraudModel` (XGB + LGB + Cat → LR)
- `AutoEncoderAnomalyDetector` (zero-day)
- `SimpleGraphSAGE` (graph embeddings)

Comparativo de métricas e latências em [`ARQUITETURA.md §5`](ARQUITETURA.md#5-métricas-atuais-dataset-sintético-100k).

### Por que o AUC do Stacking é menor que o do XGBoost solo?

Em **dataset sintético** (100k samples), os base learners (XGB/LGB/Cat) são todos
tree-based e altamente correlatos, o que limita o ganho do stacking. Em
**produção com dados reais e diversos**, o stacking tipicamente supera o ensemble simples
(literatura: Nubank, Mercado Pago).
A precision **subiu 4×** ao longo das sprints (0.078 → 0.318), reduzindo falsos positivos
— que é o que importa em operação real combinada com revisão humana.

### Como retreinar?

```bash
# XGBoost solo
python train_model.py

# Ensemble
python train_ensemble.py

# Stacking
python train_stacking.py

# Re-treino agendado com drift detection
python scripts/scheduled_retrain.py
```

GitHub Actions workflow `scheduled-retrain.yml` roda **toda segunda às 03:00 UTC**.

### Como funciona a explicabilidade (SHAP)?

`SHAPExplainerMixin` calcula valores Shapley (decomposição por feature da
contribuição na probabilidade). Devolvemos as **top-N features** quando
`is_fraud=true`. Custo: ~5-10 ms a mais.

Para ML por baixo de stacking, calculamos SHAP no **base learner XGBoost**
(meta-learner LR é facilmente explicável por outros meios).

### Como tunamos threshold?

`ThresholdTuningMixin.tune_threshold()` busca o ótimo F1 em três níveis:
1. **Por produto** (pix, ted, boleto, autenticação) — maior precedência
2. **Por canal** (app, web, api)
3. **Global**

Em runtime, `EpsilonGreedyThresholdSelector` adapta dinamicamente com base no
feedback dos analistas (`/feedback/anomaly`).

### Como detectar concept drift?

`scripts/scheduled_retrain.py` compara métricas com a run anterior:
- AUC drop > 0.05 → drift
- F1 drop > 0.10 → drift

Resultados em `logs/retrain_history.json`. `online_learning.py` também detecta
desvios em estatísticas de cada CPF.

### O que cada feature significa?

Lista completa categorizada (71 features) em [`ARQUITETURA.md §3.3`](ARQUITETURA.md#feature-engineering-srcfeature_engineeringpy).

### Posso adicionar features minhas?

Sim. Estenda `src/feature_engineering.py` ou implemente um extractor à parte
(como `OpenFinanceFeatureExtractor`). Depois retreine com `train_*.py`.

---

## Para quem vai operar a API

### Como subir localmente?

```bash
git clone https://github.com/<org>/anti-fraud-v3-wf.git
cd anti-fraud-v3-wf
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows PowerShell
# .venv/bin/activate                                   # Linux/macOS
pip install -r requirements.txt
python train_model.py                                  # gera models/fraud_model.pkl
uvicorn src.main:app --host 127.0.0.1 --port 8001 --reload
```

Swagger: http://127.0.0.1:8001/docs

### Como testar com curl?

```bash
# Health
curl http://127.0.0.1:8001/health

# Predict
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"payload":{"id":"tx-1","timestamp":"2026-04-27T10:00:00","canal":"app","produto":"pix","sender":{"cpfSender":"12345678901","banco":152},"receiver":{"cpfReceiver":"98765432100","banco":888},"valor":100.0}}'
```

Mais exemplos: [`POSTMAN_COLLECTION.md`](POSTMAN_COLLECTION.md).

### Como criar uma regra rápido?

```bash
# Criar
curl -X POST http://127.0.0.1:8001/rules \
  -H "Content-Type: application/json" \
  -d '{"rule_text":"Todo pix do CPF 12345678901 é fraude","name":"Bloqueio CPF teste"}'

# Listar
curl http://127.0.0.1:8001/rules

# Métricas
curl http://127.0.0.1:8001/rules/metrics
```

### Como rodar a suite de testes?

```bash
pytest                                  # tudo (293 passed, 5 skipped)
pytest -k "rule_engine" -v              # só rule engine
pytest --cov=src --cov-report=html      # coverage
```

### Onde ficam os logs?

| Arquivo | Conteúdo |
|---|---|
| `logs/audit.log` | Decisões do modelo ML (transaction_id, prob, is_fraud, SHAP) |
| `logs/rule_audit.jsonl` | Matches de regras (BACEN, append-only) |
| `logs/retrain_history.json` | Histórico de re-treinos com métricas |

---

## Troubleshooting

### `[WinError 10013]` ao subir na porta 8000 (Windows)

A porta está bloqueada por proxy/firewall corporativo. Use **8001** ou outra:

```bash
uvicorn src.main:app --host 127.0.0.1 --port 8001 --reload
```

E ajuste a `baseUrl` na collection do Postman.

### `ModuleNotFoundError: No module named 'src'`

Você está rodando de dentro de `src/` ou não criou o ambiente virtual.
Execute sempre da raiz do projeto e ative o `.venv`:

```bash
.venv\Scripts\Activate.ps1     # Windows
source .venv/bin/activate      # Linux/macOS
```

### `FileNotFoundError: models/fraud_model.pkl`

O modelo ainda não foi treinado:

```bash
python train_model.py
```

### `pytest` falha em `test_ensemble_model.py` com "array is read-only"

Já corrigido na branch atual. Se aparecer em fork antigo, atualize:

```python
labels = df["fraudResult"].values.copy()   # adicionar .copy()
np.random.shuffle(labels)
```

### Treino do modelo é muito lento

- Para iteração rápida, reduza `n_estimators` no XGBoost (script `train_model.py`)
- Use o dataset original (10k) em vez do expandido (100k): `--dataset dataset_transacoes.csv`
- Em CI, use `pytest -k "not slow"` para pular testes pesados

### `DeprecationWarning: datetime.utcnow()`

Já corrigido em `src/repositories.py` e `src/user_profile/service.py`.
Se aparecer em outro arquivo seu, troque por:

```python
from datetime import datetime, timezone
datetime.now(timezone.utc)
```

### A API responde, mas todas as predições retornam `is_fraud=false`

Verifique:
1. Se o modelo carregou: `GET /model/info` deve retornar `feature_count: 71`
2. Se há regras: `GET /rules` (regras de blacklist forçam `is_fraud=true`)
3. Threshold: `GET /model/info` mostra threshold atual; valores muito altos
   (próximos de 1.0) tornam o sistema permissivo demais.

### Onde reportar bugs?

- Issues no GitHub do projeto
- Equipe de Engenharia de Dados — Banco Brasileiro

---

## Aprendendo mais

| Quero entender… | Vá para |
|---|---|
| O que o sistema faz, sem termos técnicos | [`COMO_FUNCIONA.md`](COMO_FUNCIONA.md) |
| Arquitetura técnica completa | [`ARQUITETURA.md`](ARQUITETURA.md) |
| Termos que vi e não conhecia | [`GLOSSARIO.md`](GLOSSARIO.md) |
| Sintaxe das regras em PT-BR | [`INTERPRETADOR_REGRAS.md`](INTERPRETADOR_REGRAS.md) |
| Histórico de versões | [`CONCLUSAO_FINAL.md`](CONCLUSAO_FINAL.md) |
| Exemplos de payload por endpoint | [`POSTMAN_COLLECTION.md`](POSTMAN_COLLECTION.md) |
