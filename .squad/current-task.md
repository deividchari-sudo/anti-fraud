# Squad Task — 2026-04-27 12:02

## Solicitação Original
"precisamos que nosso modelo de machine learning analise individualmente segregando por produto e canal"

## Classificação
- Categoria: ML Experiment
- Workflow: `/ml-experiment`
- Agente Líder: Especialista de Dados

## Status
- [x] 1. [Especialista de Dados] Define hipótese e experimento → C: PM (KPIs), Arquiteto (infra/latência)
- [x] 2. [Especialista de Dados] Feature engineering e treino — Abordagem B (model-level segmentation) selecionada
- [x] 3. [QA] Valida qualidade do modelo
- [x] 4. [Arquiteto] Valida deployability — API não-breaking (/predict/segmented)
- [x] 5. [Backend] Integra modelo — SegmentedModelRepository + SegmentedPredictionService + endpoint
- [x] 6. [QA] Valida regressão — 306 passed, 5 skipped, 0 failures
- [ ] 7. [Documentador] Atualiza docs — README, ARQUITETURA.md, POSTMAN_COLLECTION.md

## Decisões e Apontamentos
### Passo 1 — Hipótese e Experimento (Especialista de Dados)
- **Hipótese:** Modelos segmentados por (produto × canal) superam o modelo global em F1-Score e reduzem falsos positivos, pois padrões de fraude são heterogêneos entre PIX/mobile (muito fraude por takeover) vs TED/web (phishing de alto valor).
- **Segmentos:** Produto {PIX, Cartão, TED, Logins} × Canal {mobile, web, atm, api, ivr} = até 20 combinações. Agrupar raros (ex: TED/ivr).
- **Abordagem A (Feature-based):** Adicionar `product_id` e `channel_id` como features categóricas com target encoding/embedding no modelo global. Menor custo infra, mantém um único artefato.
- **Abordagem B (Model-level):** Treinar modelos especializados por segmento. Maior acurácia potencial, mas custo de deploy N modelos, cold-start para segmentos raros, e latência de routing.
- **Experimento proposto:** Baseline (modelo global atual) vs Abordagem A (features produto/canal) vs Abordagem B (modelos por top-5 segmentos).
- **Métricas:** AUC-ROC, F1-Score, Precision@k, Recall, Falsos Positivos por segmento, Latência p99.
- **Dataset:** Split temporal 70/15/15. Estratificar por produto/canal para garantir representatividade.
- **Prazo:** 3-5 dias para experimento inicial (Abordagem A é quick-win, Abordagem B é follow-up).
