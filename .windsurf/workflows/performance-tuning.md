---
description: Otimizar latência, throughput ou consumo de recursos
---
# Squad: Performance Tuning

## 1. [Arquiteto] Identifica gargalo
- **R**: Arquiteto
- **C**: Backend (métricas), Especialista de Dados (modelo)
- Colete métricas: latência p50/p99, throughput req/s, CPU, memória.
- Identifique camada gargalo: API, modelo, banco, serialização, I/O?
- Defina meta: latência p99 < 100ms, throughput > 1000 req/s?

## 2. [Backend + Especialista de Dados] Otimiza
- **R**: Backend (infra), Especialista de Dados (modelo)
- **C**: Arquiteto
- Modelo: quantização, pruning, batch inference, ONNX?
- API: async, connection pooling, cache (Redis), compressão?
- Dados: índices, denormalização, materialized views?
- Serialização: orjson vs json, Pydantic v2 model_dump_json?

## 3. [QA] Valida regressão
- **R**: QA
- **C**: Backend, Arquiteto
- Execute `tests/test_performance.py` — baseline vs otimizado.
- Verifique que otimização não quebrou funcionalidade: `python -m pytest -q`.
- Verifique que precisão do modelo não degradou (trade-off velocidade vs acurácia).

## 4. [Documentador] Registra decisão
- **R**: Documentador
- **C**: Arquiteto
- Atualize `docs/ARQUITETURA.md` § Performance e ADRs.
- Registre trade-off aceito (ex: "aceitamos -0.5% AUC para -40ms latência").
- Atualize `docs/FAQ.md` com dicas de troubleshooting de performance.
