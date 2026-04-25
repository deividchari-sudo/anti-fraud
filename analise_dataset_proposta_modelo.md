# Análise de Dataset e Proposta de Modelo ML

## 1. Análise Exploratória dos Dados

### 1.1 Estrutura do Dataset
- **Total de transações**: 10.000
- **Classes**: 
  - Não-fraude (0): 9.900 (99%)
  - Fraude (1): 100 (1%)
- **Desbalanceamento**: Severo (99:1)

### 1.2 Features Disponíveis

#### Features Categóricas
- **canal**: web, api, app
- **produto**: autenticacao, financeiro_generico, pix, ted, boleto
- **jornada**: login_perfil, transferencia, pix_troco, pix_saque, estorno
- **direcao**: saida (apenas saída no dataset)

#### Features Numéricas
- **valor**: valor da transação (0.0 para autenticação)
- **sender.banco**: código do banco origem
- **sender.agencia**: agência origem
- **sender.nuConta**: número conta origem
- **receiver.banco**: código banco destino (null para autenticação)
- **receiver.agencia**: agência destino (null para autenticação)
- **receiver.nuConta**: número conta destino (null para autenticação)

#### Features Temporais
- **timestamp**: data/hora da transação (formato ISO 8601)

#### Features Identificadores
- **id**: UUID único da transação
- **cpfSender**: CPF do remetente
- **cpfReceiver**: CPF do destinatário (null para autenticação)

### 1.3 Análise de Padrões Potenciais de Fraude

#### Hipóteses a Validar
1. **Valor da transação**: Valores muito altos podem indicar fraude
2. **Horário**: Transações em horários atípicos (madrugada)
3. **Canal**: Alguns canais podem ter maior incidência de fraude
4. **Produto**: PIX pode ter maior risco que TED
5. **Comportamento do usuário**: Múltiplas transações em curto período
6. **Novas relações**: Primeira transação entre dois CPFs
7. **Padrões geográficos**: Transações para bancos/regiões atípicas

## 2. Proposta de Feature Engineering

### 2.1 Features Derivadas

#### Features Temporais
- **hora_do_dia**: 0-23 (feature cíclica)
- **dia_da_semana**: 0-6 (feature cíclica)
- **fim_de_semana**: binário
- **horario_noturno**: binário (22h-6h)

#### Features de Valor
- **valor_log**: log(valor + 1) para normalização
- **valor_zscore**: normalização por z-score
- **faixa_valor**: categorização (baixo, médio, alto, muito alto)

#### Features de Comportamento
- **transacoes_ultimas_1h**: número de transações do mesmo CPF nas últimas 1h
- **transacoes_ultimas_24h**: número de transações do mesmo CPF nas últimas 24h
- **valor_total_ultimas_24h**: soma dos valores das últimas 24h
- **media_valor_ultimas_24h**: média dos valores das últimas 24h
- **nova_relacao**: binário (primeira transação entre sender e receiver)

#### Features de Rede
- **grau_sender**: número de destinatários únicos do sender
- **grau_receiver**: número de remetentes únicos do receiver
- **banco_receiver_comum**: binário (receiver é banco frequente)

### 2.2 Encoding de Features Categóricas
- **One-Hot Encoding**: canal, produto, jornada (baixa cardinalidade)
- **Target Encoding**: sender.banco, receiver.banco (alta cardinalidade)
- **Label Encoding**: direcao (binário)

## 3. Proposta de Modelo de Machine Learning

### 3.1 Modelo Principal: Gradient Boosting

**Algoritmo**: XGBoost ou LightGBM

**Justificativa**:
- **Latência**: Inferência rápida (< 5ms por transação)
- **Performance**: Excelente em dados tabulares
- **Robustez**: Lida bem com features categóricas e numéricas
- **Interpretabilidade**: Feature importance disponível
- **Escalabilidade**: Suporta milhões de previsões

**Hiperparâmetros Sugeridos**:
```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': 99,  # para lidar com desbalanceamento
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

### 3.2 Alternativa: Isolation Forest (Anomaly Detection)

**Justificativa**:
- **Não supervisionado**: Não depende de labels de fraude
- **Velocidade**: Inferência extremamente rápida (< 2ms)
- **Detecção de novidades**: Identifica padrões atípicos

**Uso**: Como modelo secundário para ensemble

### 3.3 Estratégia de Treinamento

#### Divisão dos Dados
- **Treinamento**: 70% (7.000 transações)
- **Validação**: 15% (1.500 transações)
- **Teste**: 15% (1.500 transações)

#### Estratificação
- Stratified split para manter proporção de fraudes

#### Cross-Validation
- 5-fold stratified cross-validation
- Early stopping para evitar overfitting

### 3.4 Tratamento de Desbalanceamento

#### Técnicas a Aplicar
1. **Class Weighting**: scale_pos_weight no XGBoost
2. **SMOTE**: Synthetic Minority Over-sampling Technique (apenas no treino)
3. **Threshold Tuning**: Ajustar threshold de decisão para maximizar F1-Score

## 4. Métricas de Avaliação

### 4.1 Métricas Principais
- **F1-Score**: > 0.85 (equilíbrio precision/recall)
- **AUC-ROC**: > 0.90 (capacidade de discriminação)
- **Precision**: > 0.80 (minimizar falsos positivos)
- **Recall**: > 0.90 (maximizar detecção de fraudes)

### 4.2 Métricas Secundárias
- **Confusion Matrix**: análise detalhada de erros
- **Precision-Recall Curve**: para threshold selection
- **SHAP Values**: interpretabilidade do modelo

## 5. Considerações de Arquitetura

### 5.1 Pipeline de Inferência
```
Transação → Feature Engineering (real-time) → Modelo ML → Score → Decisão
```

### 5.2 Latência Estimada
- **Feature Engineering**: ~30ms
- **Inferência Modelo**: ~5ms
- **Total**: ~35ms (b abaixo do limite de 100ms)

### 5.3 Escalabilidade
- **Batch Processing**: Para treinamento e re-treinamento
- **Real-time Processing**: Para inferência (FastAPI + Redis cache)
- **Model Versioning**: MLflow para rastreabilidade

### 5.4 Monitoramento
- **Drift Detection**: Monitorar distribuição de features
- **Performance Monitoring**: Alertas se métricas caírem
- **Audit Trail**: Logs de todas as decisões (compliance BACEN)

## 6. Plano de Implementação

### Fase 1: Protótipo (1 semana)
1. Feature engineering básico
2. Treinamento modelo XGBoost baseline
3. Avaliação de métricas

### Fase 2: Otimização (1 semana)
1. Feature engineering avançado
2. Tuning de hiperparâmetros
3. Ensemble com Isolation Forest

### Fase 3: Produção (1 semana)
1. API FastAPI para inferência
2. Pipeline de re-treinamento
3. Monitoramento e alertas

## 7. Riscos e Mitigações

### Risco 1: Overfitting devido ao dataset pequeno
**Mitigação**: Cross-validation rigoroso, regularização, early stopping

### Risco 2: Concept drift (padrões de fraude mudam)
**Mitigação**: Re-treinamento semanal, monitoramento de drift

### Risco 3: Alta taxa de falsos positivos
**Mitigação**: Threshold tuning, feedback loop com analistas

## 8. Próximos Passos

1. Implementar feature engineering
2. Treinar modelo baseline XGBoost
3. Avaliar métricas contra KPIs do PM
4. Se aprovado, prosseguir para implementação da API
