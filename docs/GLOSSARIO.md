# Glossário — Termos Técnicos e de Negócio

Glossário pensado para **leitores mistos** (negócio, compliance, jurídico, dados, engenharia).
Cada termo tem uma explicação curta de uma frase ("o que é") e onde ele aparece no projeto.

---

## Negócio e regulatório

**BACEN (Banco Central do Brasil)** — Autoridade monetária brasileira; publicou a Resolução BCB nº 6 (2020) que define exigências de detecção de fraude em tempo real, auditabilidade e latência. _Aparece em:_ todas as decisões precisam ser auditáveis e responder em < 100 ms.

**LGPD (Lei nº 13.709/2018)** — Lei Geral de Proteção de Dados Pessoais brasileira; obriga minimização de dados, direito à explicação e proteção de informação sensível. _Aparece em:_ CPFs são hasheados; logs registram acessos; SHAP devolve a explicação.

**MED (Mecanismo Especial de Devolução)** — Procedimento do PIX que permite ao banco solicitar a devolução de valores transferidos a contas suspeitas em até 80 dias. _Aparece em:_ contexto de mitigação pós-fraude.

**PIX** — Sistema de pagamentos instantâneos do BACEN (24×7, sub-segundos). _Aparece em:_ produto principal monitorado pelo sistema.

**TED** — Transferência Eletrônica Disponível; transferência interbancária em horário comercial. _Aparece em:_ produto secundário monitorado.

**Boleto** — Forma de cobrança com código de barras; pode ser fraudado por troca do beneficiário. _Aparece em:_ produto monitorado.

**Money mule (laranja)** — Conta usada por terceiros para receber e repassar valores fraudulentos rapidamente, dificultando o rastreamento. _Aparece em:_ detecção via análise de grafo (degree alto + repasse rápido).

**Cold start** — Cliente sem histórico suficiente (< 30 transações) para gerar perfil comportamental confiável. _Aparece em:_ usa-se um perfil global agregado como prior.

**Conta laranja** — sinônimo de money mule.

**Whitelist** — Lista de exceções; transações que casam com regra de whitelist são marcadas como **legítimas** sem passar pelo modelo. _Aparece em:_ rule engine — `"... não é fraude"`.

**Blacklist** — Lista de bloqueio; transações que casam são marcadas como **fraude** automaticamente. _Aparece em:_ rule engine — `"... é fraude"`.

**Falso positivo (FP)** — Transação legítima classificada como fraude (incomoda o cliente). _Aparece em:_ métrica de feedback dos analistas; otimizamos para reduzir.

**Falso negativo (FN)** — Fraude que passou como legítima (perda financeira). _Aparece em:_ pior cenário; recall mede a fração de fraudes capturadas.

---

## Métricas de modelo

**AUC-ROC** — Área sob a curva ROC; mede a capacidade do modelo de **separar** fraude de não-fraude em qualquer threshold. **0.5 = aleatório, 1.0 = perfeito**. Nossa baseline: 0.8395 (XGBoost).

**F1-Score** — Média harmônica entre precision e recall. Útil em datasets desbalanceados (poucas fraudes). **0 = ruim, 1 = perfeito**. Nosso atual: 0.32 (Stacking).

**Precision** — Dos alertas de fraude, **quantos são fraude real** (P = TP / (TP + FP)). Importante para reduzir incômodo ao cliente.

**Recall (sensibilidade)** — Das fraudes que existem, **quantas o modelo pegou** (R = TP / (TP + FN)). Importante para evitar prejuízo.

**Threshold** — Valor de corte da probabilidade que decide fraude vs legítimo (ex: 0.5 ou 0.889). _Aparece em:_ `ThresholdTuningMixin` busca o ótimo por F1, e o `RL Threshold Selector` adapta online.

**Latência P95 / P99** — 95% (ou 99%) das requisições respondem em até esse tempo. BACEN exige P95 < 100 ms.

---

## Machine Learning

**XGBoost** — Algoritmo de **gradient boosting** com árvores; padrão de mercado em fraude tabular. Usado no Itaú, Nubank, Inter. _Aparece em:_ `src/model.py`.

**LightGBM** — Implementação alternativa de boosting com leaf-wise growth; mais rápida que XGBoost. _Aparece em:_ ensemble e stacking.

**CatBoost** — Boosting com tratamento nativo de variáveis categóricas e ordered boosting. _Aparece em:_ stacking model.

**Ensemble** — Combinação de **vários modelos** (votação ponderada) para reduzir variância. _Aparece em:_ `EnsembleFraudModel` (XGB + LGB).

**Stacking** — Ensemble onde um **meta-learner** aprende a combinar as predições dos base learners. _Aparece em:_ `StackingFraudModel` (XGB+LGB+Cat → Logistic Regression).

**Calibração (CalibratedClassifierCV / isotonic)** — Ajuste para que `predict_proba` reflita probabilidade real (ex: 0.7 = 70% real). _Aparece em:_ ensemble e stacking.

**SMOTE (Synthetic Minority Over-sampling Technique)** — Gera amostras sintéticas da classe minoritária (fraude) para balancear o treino. _Aparece em:_ `BaseFraudModel.prepare_train_test()`.

**SHAP (SHapley Additive exPlanations)** — Técnica de **explicabilidade**; mostra quanto cada feature contribuiu para a decisão. Exigida por BACEN/LGPD. _Aparece em:_ `SHAPExplainerMixin`, retornado em `/predict` quando `is_fraud=true`.

**Feature engineering** — Transformação dos dados brutos em entradas numéricas que o modelo entende. Nosso sistema gera 71 features. _Aparece em:_ `src/feature_engineering.py`.

**One-hot encoding** — Transforma categoria em colunas binárias (`is_pix`, `is_app`…).

**Encoding cíclico (sin/cos)** — Converte variáveis circulares (hora, dia da semana) em pares (sin, cos) para preservar a continuidade.

**Z-score** — Quantos desvios-padrão um valor está da média. _Aparece em:_ feature `valor_zscore` e detecção de anomalias univariadas.

**Concept drift** — Mudança gradual no padrão dos dados ao longo do tempo (golpistas inventam técnicas novas). _Aparece em:_ `online_learning.py` detecta drift; `scheduled_retrain.py` re-treina semanalmente.

---

## Anomalia e não-supervisionado

**Isolation Forest** — Algoritmo que isola pontos anômalos com poucos cortes aleatórios; rápido e multivariado. _Aparece em:_ `user_profile/isolation_forest.py` (ensemble de 5 modelos com voting).

**KMeans** — Clustering particional; agrupa CPFs em **10 clusters comportamentais** (ex: "high_value_night_user"). _Aparece em:_ `user_profile/clustering.py`.

**DBSCAN** — Clustering por densidade; descobre clusters de forma arbitrária e marca outliers automaticamente. _Aparece em:_ alternativa ao KMeans.

**Silhouette score** — Métrica de qualidade de clustering (-1 a 1). _Aparece em:_ validação automática após cada treino.

**AutoEncoder** — Rede neural treinada para **reconstruir** sua entrada; treinada **só em transações legítimas**, fraudes geram alta reconstruction error → "zero-day detection". _Aparece em:_ `autoencoder_anomaly.py`.

**Auto-contamination** — Heurística que encontra automaticamente o parâmetro `contamination` ótimo do Isolation Forest. _Aparece em:_ `isolation_forest.py`.

**Cold start (ML)** — Modelo ou perfil sem dados suficientes para predição confiável.

---

## Análise de Grafo

**Grafo de transações** — Cada CPF é um nó; cada transferência é uma aresta direcional ponderada pelo valor.

**Degree (grau)** — Quantidade de conexões de um nó. Money mules tipicamente têm degree alto.

**PageRank** — Algoritmo do Google; mede importância de um nó na rede. Alto PageRank em conta nova é suspeito.

**Clustering coefficient** — Mede quão "fechada" é a vizinhança de um nó (triângulos). Padrões circulares (A→B→C→A) têm clustering coefficient alto.

**GraphSAGE** — Algoritmo de Graph Neural Network que produz **embeddings** vetoriais de nós, generalizando para nós novos. _Aparece em:_ `simple_graph_sage.py` (implementação NumPy puro, sem PyTorch Geometric).

**Embedding** — Representação vetorial densa (ex: 24 dimensões) de uma entidade complexa.

---

## Aprendizado distribuído e adaptativo

**Online learning** — Modelo que se atualiza continuamente, transação a transação, sem precisar re-treinar do zero. _Aparece em:_ `online_learning.py` (EMA + drift).

**EMA (Exponential Moving Average)** — Média ponderada que dá mais peso aos dados recentes. Usado em estatísticas online.

**Forgetting factor** — Hiperparâmetro que controla a velocidade com que dados antigos perdem importância na EMA.

**Federated Learning** — Vários bancos treinam um modelo conjunto **sem trocar dados de clientes** — só pesos do modelo. _Aparece em:_ `FederatedAggregator` com FedAvg.

**FedAvg** — Algoritmo padrão de Federated Learning: cada participante treina localmente, agregador faz média ponderada dos pesos.

**Differential Privacy (DP)** — Adiciona ruído controlado aos pesos antes do compartilhamento, protegendo dados individuais. _Aparece em:_ FederatedAggregator.

**Reinforcement Learning (RL) — multi-armed bandit** — Algoritmo que aprende a escolher a melhor ação (ex: melhor threshold) com base em rewards. _Aparece em:_ `EpsilonGreedyThresholdSelector` adapta o threshold ao feedback dos analistas.

**Epsilon-greedy** — Estratégia RL: com probabilidade `ε` explora opção aleatória, senão escolhe a melhor conhecida. `epsilon_decay` reduz `ε` com o tempo.

---

## Engenharia de software

**FastAPI** — Framework web Python moderno baseado em type hints; gera Swagger automaticamente.

**Pydantic** — Biblioteca de validação de dados com type hints; usado nos schemas dos payloads.

**Repository Pattern** — Padrão arquitetural que isola acesso a dados atrás de uma interface (ABC). _Aparece em:_ `src/repositories.py` e `rule_engine/repository.py`.

**Dependency Injection (DI)** — Receber dependências como parâmetros em vez de instanciá-las internamente; facilita testes (mock fácil).

**ABC (Abstract Base Class)** — Classe Python que define interface abstrata. _Aparece em:_ `BaseFraudModel`, `RuleRepository`, `ModelRepository`.

**Mixin** — Classe pequena para compartilhar funcionalidade via herança múltipla. _Aparece em:_ `ThresholdTuningMixin`, `SHAPExplainerMixin`.

**JSONL (JSON Lines)** — Formato com um JSON por linha, append-only. _Aparece em:_ `logs/rule_audit.jsonl`.

**SQLite** — Banco relacional em arquivo único; usado para perfis. Em produção será trocado por PostgreSQL.

**Joblib** — Biblioteca Python para serializar modelos sklearn/xgboost.

**Salt (criptografia)** — String secreta concatenada antes do hash para impedir ataques de rainbow table. _Aparece em:_ `crypto.py` no hashing de CPF.

**SHA-256** — Função hash criptográfica; produz 256 bits, computacionalmente irreversível.

**Atomic write (`os.replace`)** — Substituição atômica de arquivo; garante que persistência não corrompe em caso de crash. _Aparece em:_ `JSONRuleRepository`.

**Audit log** — Registro imutável e timestamped de decisões; exigido por BACEN.

**Canary release** — Deploy gradual (1% → 10% → 50% → 100%) para detectar problemas antes de afetar todos os clientes.

**SLO (Service Level Objective)** — Meta interna de qualidade (ex: P95 < 100ms).

---

## Roles da Squad

**[PM]** — Product Manager; foca em KPIs, compliance e narrativa.

**[Arquiteto]** — Define DDD (Domain-Driven Design), camadas, decisões técnicas.

**[Especialista de Dados]** — Cientista de dados; modelos ML, métricas estatísticas.

**[Backend]** — Implementação Python/FastAPI, integração, persistência.

**[QA]** — BDD, testes, garantia de auditoria algorítmica.

**[Documentador]** — Documentação técnica e de negócio (este papel).

---

## Onde encontrar mais

- Conceitos de negócio e analogias: [`COMO_FUNCIONA.md`](COMO_FUNCIONA.md)
- Arquitetura técnica: [`ARQUITETURA.md`](ARQUITETURA.md)
- Rule engine em detalhes: [`INTERPRETADOR_REGRAS.md`](INTERPRETADOR_REGRAS.md)
- Histórico de evolução: [`CONCLUSAO_FINAL.md`](CONCLUSAO_FINAL.md)
- Perguntas frequentes: [`FAQ.md`](FAQ.md)
