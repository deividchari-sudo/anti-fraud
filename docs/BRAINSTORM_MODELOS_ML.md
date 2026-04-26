# Brainstorm: Modelos ML para Detecção de Fraude - Sistema Financeiro Brasileiro

**Autores**: [PM] + [Especialista de Dados]
**Data**: 26/04/2026
**Versão de referência**: 1.5.0

---

## 1. Contexto de Mercado (Brasil 2024-2025)

- Perdas com fraude: 0,35–0,40% do PIB (R$ 10+ bi em 2024)
- Custo de resposta: R$ 4,49 gastos para cada R$ 1,00 perdido
- PIX: 5+ bi transações/mês (BACEN), maior vetor de fraude atual
- Regulação: BACEN (Res. 6) + LGPD + MED (Mecanismo Especial de Devolução)
- Tendências: Open Finance, biometria comportamental, federated learning

---

## 2. Modelos em Uso pelos Grandes Bancos Brasileiros

### 2.1 Modelos Tabulares Clássicos

| Modelo | Uso típico | Pontos Fortes | Limitações |
|--------|-----------|---------------|------------|
| Logistic Regression | Itaú, BB (camada 1) | Explicável, rápido, regulatório | Não captura não-linearidades |
| Random Forest | Bradesco, Caixa | Robusto, feature importance | Custo computacional alto |
| XGBoost ⭐ | Itaú, Nubank, Inter | SOTA tabular, SHAP nativo | Tuning complexo |
| LightGBM | Nubank, Stone | Mais rápido que XGBoost | Sensível a desbalanceamento |
| CatBoost | C6, Mercado Pago | Categóricas nativas | Menos comunidade |

### 2.2 Deep Learning

| Modelo | Uso típico | Pontos Fortes | Limitações |
|--------|-----------|---------------|------------|
| TabNet | PicPay (P&D) | Atenção interpretável | Treino lento |
| AutoEncoder | Itaú (anomalia) | Não-supervisionado | FP altos |
| LSTM/GRU | BB (sequencial) | Padrões temporais | Latência >50ms |
| TabTransformer | Banco Inter | Self-attention | Caro em produção |

### 2.3 Graph Neural Networks (GNN) — Tendência

| Modelo | Uso típico | Pontos Fortes | Limitações |
|--------|-----------|---------------|------------|
| GraphSAGE | Nubank, Bradesco (P&D) | Money mules, redes | Latência <100ms difícil |
| GAT | Itaú (P&D) | Foco em conexões críticas | Memória alta |
| PinSage / TigerGraph | Mercado Pago | Real-time graph queries | Stack proprietária |

### 2.4 Não-Supervisionados

| Modelo | Uso típico | Pontos Fortes | Limitações |
|--------|-----------|---------------|------------|
| Isolation Forest ⭐ | Inter, Stone | Multivariado, rápido | Sem contexto temporal |
| DBSCAN | C6 | Outliers automáticos | Sensível a eps |
| HDBSCAN | Nubank | Densidade variável | Complexo |
| One-Class SVM | Caixa | Fronteira clara | Não escala |

### 2.5 Ensemble / Stacking — SOTA

| Modelo | Uso típico | Pontos Fortes |
|--------|-----------|---------------|
| XGBoost + IF + Rules | Itaú, Bradesco | Cobre supervisionado + anomalia + compliance |
| Stacking LR + XGB + LGB + CatBoost | Nubank | F1 0.92+ |
| Ensemble GNN + XGBoost | Mercado Pago | Rede + transação |

---

## 3. Comparativo Nosso Modelo (v1.5.0) vs Mercado

| Aspecto | Nosso v1.5.0 | Itaú/Nubank | Gap |
|---------|--------------|-------------|-----|
| AUC-ROC | 0.8395 | 0.92–0.96 | -8 a -12 pp |
| F1-Score | 0.2778 | 0.85–0.92 | Significativo |
| Recall (fraude) | 0.5950 | 0.85–0.95 | -25 a -35 pp |
| Latência | 28ms | 50–150ms | Melhor |
| Explicabilidade | SHAP + Rules | SHAP/LIME | Equivalente |
| Graph Features | NetworkX | GNN dedicado | -2 gerações |
| Online Learning | EMA + drift | Streaming completo | -1 geração |
| Volume treino | 100k | 50–500M | -3 ordens |
| Dataset | Sintético | Real produção | Crítico |

### Comparativo por Banco

| Banco | Stack | F1 estimado |
|-------|-------|-------------|
| Nubank | XGBoost + LightGBM stacking + GNN P&D | ~0.91 |
| Itaú | Rules + XGBoost + AutoEncoder + LSTM | 0.89–0.93 |
| Bradesco | Rules + Random Forest + GraphSAGE P&D | 0.85–0.90 |
| Inter | XGBoost + IF + biometria | 0.83–0.88 |
| Mercado Pago | Ensemble + GNN real-time | 0.90+ |
| **Nosso v1.5.0** | Rules + XGBoost + IF + Graph + Online | 0.28 (sintético) |

---

## 4. Análise de Gaps

### Crítico (bloqueia produção)

1. Dataset sintético — métricas não refletem produção
2. Recall 59% — 41% das fraudes passariam, inaceitável
3. Falta calibração de probabilidades

### Médio (limita qualidade)

4. Sem ensemble supervisionado (LightGBM/CatBoost)
5. Graph features simples — NetworkX != GNN
6. Sem features de Open Finance

### Pontos Fortes

7. Rules + ML híbrido (padrão BACEN)
8. SHAP nativo (acima da média)
9. Latência 28ms (melhor que Stone, Inter ~80ms)
10. Behavioral profiling com online learning (diferencial)

---

## 5. Roadmap de Evolução

### Sprint 1 — Quick Wins (1 semana)

| Ação | Esperado |
|------|----------|
| LightGBM no ensemble com XGBoost | AUC +1.5pp, F1 +5pp |
| Calibração CalibratedClassifierCV (isotonic) | Recall@P=0.95 +10pp |
| Threshold tuning por canal/produto | F1 +3pp |
| Stratified K-Fold + early stopping | Reduz overfitting |

### Sprint 2 — Médio Prazo (2-4 semanas)

| Ação | Esperado |
|------|----------|
| Stacking XGBoost + LightGBM + CatBoost → LR meta | AUC 0.88+, F1 0.40+ |
| Features de Open Finance | F1 +10pp |
| Re-treino semanal automático | Combate concept drift |

### Sprint 3 — Estratégico (1-3 meses)

| Ação | Esperado |
|------|----------|
| GraphSAGE (PyTorch Geometric) | Detecção de redes +30% |
| TabNet/TabTransformer | AUC 0.92+ |
| AutoEncoder para anomalias zero-day | Padrões emergentes |

### Sprint 4 — P&D (3-6 meses)

- Federated Learning entre instituições
- LLM para análise textual (Open Finance)
- Reinforcement Learning para threshold dinâmico

---

## 6. Conclusão da Squad

**[PM]**: Arquitetura correta. Modelo 1 geração atrás. Métricas inválidas sem dados reais.

**[Especialista de Dados]**: Stack sólida. Falta ensemble supervisionado (quick win óbvio). Graph features precisam evoluir para GNN.

### Próximo passo recomendado

Implementar **Sprint 1**: LightGBM ensemble + calibração + threshold tuning.
Estimativa: **F1 0.28 → 0.38–0.45** sem mudar dados.
