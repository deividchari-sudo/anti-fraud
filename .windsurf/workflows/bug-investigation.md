---
description: Investigar e corrigir bug com análise de causa raiz
---
# Squad: Bug Investigation

## 1. [QA] Reproduz e caracteriza
- **R**: QA
- **C**: Backend (contexto técnico)
- Descreva sintomas, ambiente, frequência.
- Crie teste mínimo que reproduz o bug.
- Classifique: crítico (bloqueia produção), alto, médio, baixo.

## 2. [Backend + Arquiteto] Análise de causa raiz
- **R**: Backend (investigação técnica)
- **C**: Arquiteto (design), Especialista de Dados (se envolver modelo)
- Use logs (`logs/app.log`, `logs/audit.jsonl`) para rastrear.
- Identifique commit que introduziu (git bisect se necessário).
- Distinga: bug de código, bug de modelo, bug de infra, bug de config.

## 3. [Backend] Implementa correção mínima
- **R**: Backend
- **C**: Arquiteto (valida se a correção é no lugar certo)
- Prefira upstream fix sobre workaround downstream.
- Adicione teste de regressão.
- Se alterar regra de negócio, consulte PM.

## 4. [QA] Valida correção
- **R**: QA
- **C**: Backend
- O teste de reprodução passa?
- Testes existentes continuam passando?
- Performance não degradou?

## 5. [Documentador] Atualiza FAQ/troubleshooting
- **R**: Documentador
- **C**: QA (detalhes técnicos), PM (impacto de negócio)
- Se bug é recorrente ou non-obvious, adicione em `docs/FAQ.md`.
- Registre causa raiz para auditoria futura.
