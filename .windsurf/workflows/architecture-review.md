---
description: Revisar arquitetura após mudança significativa
---
# Squad: Architecture Review

## 1. [Arquiteto] Identifica escopo da mudança
- **R**: Arquiteto
- **C**: PM (prioridade), Backend (implementação proposta)
- Liste arquivos/modulos afetados.
- Avalie se quebra contratos existentes (backward compatibility).
- Decida: evolução gradual ou breaking change com migration.

## 2. [Arquiteto] Revisa ADRs (Architecture Decision Records)
- **R**: Arquiteto
- **C**: Backend, Especialista de Dados
- A decisão contradiz algum ADR existente em `docs/ARQUITETURA.md`?
- Se sim, registre novo ADR ou deprecie o antigo com justificativa.
- Avalie trade-offs: performance vs manutenibilidade vs compliance.

## 3. [Especialista de Dados] Revisa impacto em modelos
- **R**: Especialista de Dados
- **C**: Arquiteto
- Mudança de schema de features? Modelos precisam re-treino?
- Avalie drift: a mudança altera distribuição de dados de entrada?

## 4. [QA] Revisa testes e cobertura
- **R**: QA
- **C**: Arquiteto, Backend
- Testes existentes cobrem a mudança?
- Precisa de teste de contrato, carga, ou caos?

## 5. [Documentador] Atualiza ARQUITETURA.md
- **R**: Documentador
- **C**: Arquiteto
- Atualize diagrama, stack, camadas, ADRs.
- Atualize `docs/GLOSSARIO.md` se houver novo termo.
