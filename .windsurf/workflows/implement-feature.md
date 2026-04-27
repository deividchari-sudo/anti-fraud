---
description: Implementar nova feature end-to-end (PM → Arquiteto → Dados → Backend → QA → Documentador)
---
# Squad: Implementação de Feature

Siga a matriz RACI do projeto. Cada passo deve indicar quem é Responsible (R) e quem é Consulted (C).

## 1. [PM] Define escopo e critérios de aceite
- **R**: PM
- **C**: Arquiteto (viabilidade técnica), Especialista de Dados (impacto nos modelos)
- Descreva a feature em termos de negócio: qual problema resolve, qual KPI afeta, qual volume esperado.
- Liste critérios de aceite mínimos (BDD-style: "Dado... Quando... Então...").
- Valide compliance: dados sensíveis? LGPD? BACEN?

## 2. [Arquiteto] Desenha arquitetura
- **R**: Arquiteto
- **C**: PM (prioridades de negócio), Backend (implementabilidade)
- Defina camadas afetadas (API, domínio, infra, dados).
- Identifique contratos (schemas Pydantic, interfaces de repositório).
- Avalie latência: <100ms para detecção síncrona?
- Decida se precisa de migration, novo repositório, ou novo módulo.

## 3. [Especialista de Dados] Valida impacto em ML/Features
- **R**: Especialista de Dados
- **C**: Arquiteto (latência), PM (KPIs)
- Liste features novas/modificadas. Impactam os modelos existentes?
- Se necessário, proponha experimento A/B ou re-treino.
- Documente decisão de não-impactar se for o caso.

## 4. [Backend] Implementa
- **R**: Backend
- **C**: Arquiteto (padrões), Especialista de Dados (features)
- Implemente seguindo DDD e Repository Pattern.
- Adicione testes unitários e de integração.
- Garanta logging auditável (JSONL, trace_id).
- Mantenha compatibilidade com Pydantic v2.

## 5. [QA] Valida
- **R**: QA
- **C**: Todos (critérios de aceite)
- Execute testes existentes: `python -m pytest -q`
- Verifique cobertura de testes da nova feature.
- Valide latência com `tests/test_performance.py`.
- Valide audit trail: decisões são rastreáveis?
- Sign-off ou bloqueios devem ser listados.

## 6. [Documentador] Atualiza docs
- **R**: Documentador
- **C**: Backend (APIs), PM (mensagem de negócio)
- Atualize README.md se houver novo endpoint ou comando.
- Atualize docs técnicos afetados (ARQUITETURA, INTERPRETADOR_REGRAS, etc).
- Atualize Postman collection se houver novo endpoint.
- Registre decisões em FAQ.md se resolvem problema recorrente.
