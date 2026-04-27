---
description: Verificar conformidade BACEN e LGPD
---
# Squad: Compliance Check

## 1. [PM] Lista requisitos regulatórios aplicáveis
- **R**: PM
- **C**: Arquiteto (implementação), QA (evidências)
- BACEN: Res. 6 (latência, auditabilidade, explicabilidade)?
- LGPD: dados sensíveis, finalidade, tempo de retenção, direito ao esquecimento?
- Novo regulamento ou atualização de norma?

## 2. [Backend] Audita código e dados
- **R**: Backend
- **C**: Arquiteto, PM
- Verifique hashing de PPF/identificadores: `hashlib.sha256`?
- Verifique tempo de retenção: dados são expurgados após N dias?
- Verifique audit trail: toda decisão de fraude é logada em `logs/audit.jsonl`?
- Verifique consentimento: APIs de opt-out implementadas?

## 3. [QA] Valida evidências
- **R**: QA
- **C**: Backend, PM
- Gere relatório de evidências: logs, testes, screenshots.
- Valide que não há log de dados sensíveis em texto plano.
- Valide que explicabilidade (SHAP/LIME) está disponível para toda decisão.
- Valide latência: <100ms para detecção síncrona?

## 4. [Documentador] Gera relatório de compliance
- **R**: Documentador
- **C**: PM, QA
- Documento formal com: norma, requisito, evidência, status (conforme/não conforme), ação corretiva.
- Armazene em `docs/compliance/` com data e versão.
- Atualize `docs/FAQ.md` com respostas sobre LGPD/BACEN.
