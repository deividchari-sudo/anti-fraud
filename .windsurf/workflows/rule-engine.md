---
description: Evoluir Rule Engine (nova regra, validação, simulação, conflitos)
---
# Squad: Rule Engine Evolution

## 1. [PM] Define regra de negócio
- **R**: PM
- **C**: Arquiteto (viabilidade de parser), QA (cobertura de teste)
- Regra em linguagem natural: ex: "Bloquear PIX acima de 50.000 entre 00h e 06h exceto para clientes VIP"
- Ação esperada: `block`, `review`, `allow`?
- Prioridade relativa às regras existentes.
- Volume estimado de matching (evitar regras com 50%+ match).

## 2. [Backend] Implementa parser e regra
- **R**: Backend
- **C**: Arquiteto (padrão Rule), QA (casos de borda)
- Adicione/ajuste regex no `parser.py` se necessário.
- Crie regra com `RuleParser.parse(natural_language)`.
- Garanta que `Condition` suporta operadores necessários (`>`, `<`, `in`, `not in`, etc.).
- Teste com `RuleEvaluator.evaluate_single(transaction)`.

## 3. [QA] Valida lógica e conflitos
- **R**: QA
- **C**: Backend, PM
- Execute `POST /rules/{id}/simulate` com batch de transações.
- Verifique `GET /rules/conflicts` — a nova regra conflita com existentes?
- Teste casos de borda: horário limite, valor exato, CPF inválido.
- Verifique que audit log registra quem criou a regra.

## 4. [Backend] Deploy com feature flag
- **R**: Backend
- **C**: Arquiteto, PM
- Regra inicialmente `enabled: false` ou com `dry_run: true`.
- Monitore métricas de hit (`GET /rules/{id}/metrics`) por 24-48h.
- Depois de validado, habilite (`PATCH /rules/{id}`).

## 5. [Documentador] Atualiza docs de regras
- **R**: Documentador
- **C**: Backend (sintaxe exata), PM (exemplos de negócio)
- Atualize `docs/INTERPRETADOR_REGRAS.md` com nova sintaxe.
- Adicione exemplo no `docs/FAQ.md`.
- Atualize Postman collection se houver novo endpoint.
