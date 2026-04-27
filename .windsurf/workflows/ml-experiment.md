---
description: Executar experimento de ML (novo modelo, feature engineering, tuning)
---
# Squad: Experimento de ML

## 1. [Especialista de Dados] Define hipótese e experimento
- **R**: Especialista de Dados
- **C**: PM (KPIs), Arquiteto (infra/latência)
- Hipótese: "Adicionar feature X melhora F1-Score em Y%"
- Métricas de sucesso: AUC-ROC, F1, Precision, Recall, Latência
- Dataset: treino/validação/teste split (temporal se possível)
- Prazo de execução.

## 2. [Especialista de Dados] Feature engineering e treino
- **R**: Especialista de Dados
- **C**: Arquiteto (infraestrutura de dados)
- Implemente notebook ou script de experimento em `experiments/`.
- Versione modelo com `BaseFraudModel.save_model()`.
- Compare baseline vs experimento com métricas estatísticas (teste t ou bootstrap).

## 3. [QA] Valida qualidade do modelo
- **R**: QA
- **C**: Especialista de Dados
- Verifique overfitting (train vs validation gap < 5%).
- Verifique fairness (métricas por segmento de valor/horário).
- Valide explicabilidade (SHAP/LIME disponível?).
- Sign-off ou bloqueios.

## 4. [Arquiteto] Valida deployability
- **R**: Arquiteto
- **C**: Backend (API), Especialista de Dados
- O modelo cabe em memória? Tempo de load < 2s?
- Impacto na latência do endpoint `/predict`?
- Necessita alteração de schema ou contrato de API?

## 5. [Backend] Integra modelo
- **R**: Backend
- **C**: Arquiteto, Especialista de Dados
- Substitua ou adicione modelo no ensemble/stacking.
- Atualize `model/info` endpoint se necessário.
- Adicione testes de contrato para o novo modelo.

## 6. [PM] Avalia resultado de negócio
- **R**: PM
- **C**: Especialista de Dados
- O ganho de métricas justifica o deploy?
- Impacto estimado em falsos positivos (clientes impactados)?
- Decisão: deploy, mais experimentos, ou descartar.
