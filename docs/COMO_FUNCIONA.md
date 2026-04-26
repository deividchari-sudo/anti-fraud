# Como Funciona Nossa Avaliação de Fraudes

Um guia simples, sem termos técnicos.

---

## 1. O Problema

Toda vez que alguém faz um PIX, TED, paga um boleto ou faz login no banco, **pode ser uma fraude**: golpistas usando contas roubadas, sequestros relâmpago, golpe do falso motoboy, etc.

Em 2024, fraudes bancárias custaram **R$ 10 bilhões** ao Brasil. Para cada R$ 1 perdido, o banco gasta R$ 4,49 lidando com o problema.

**Nossa missão**: identificar transações suspeitas em **menos de 1 décimo de segundo**, antes que o dinheiro saia da conta.

---

## 2. A Analogia do Porteiro

Pense no nosso sistema como um **porteiro de um prédio de luxo** com 3 níveis de verificação:

```
   ┌─────────────────────────────────────────────────────┐
   │           Toda transação passa por aqui              │
   └─────────────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐  ┌──────────┐  ┌──────────────┐
       │ Filtro 1 │  │ Filtro 2 │  │   Filtro 3   │
       │  Regras  │→ │ Modelo   │→ │ Análise do   │
       │ (porteiro│  │ Inteli-  │  │ comportamen- │
       │  rígido) │  │ gente    │  │ to do clien- │
       │          │  │          │  │ te           │
       └──────────┘  └──────────┘  └──────────────┘
              │            │            │
              └────────────┼────────────┘
                           ▼
                  ┌────────────────┐
                  │  DECISÃO FINAL │
                  │  ✅ Liberada    │
                  │  ⚠️ Para análise│
                  │  ❌ Bloqueada   │
                  └────────────────┘
```

---

## 3. Os 3 Filtros Explicados

### 🛡️ Filtro 1: Regras (o porteiro rígido)

São **regras fixas escritas em português**, como um manual do porteiro. Por exemplo:

- "Se a transação for maior que R$ 50.000 entre 2h e 5h da manhã → suspeita"
- "Se for o primeiro PIX para esse destinatário e valor for muito alto → suspeita"
- "Se houver 5 tentativas de login em 1 minuto → bloquear"

**Vantagem**: simples, rápido, fácil de explicar para um juiz.
**Limitação**: golpistas aprendem a contornar.

### 🧠 Filtro 2: Modelo Inteligente (o detetive)

Aqui usamos **inteligência artificial** treinada com 100 mil exemplos passados de transações boas e fraudulentas. O modelo aprendeu sozinho a reconhecer padrões suspeitos olhando para **71 características** ao mesmo tempo:

- Valor da transação
- Hora do dia
- Dia da semana
- Canal usado (app, web, internet banking)
- Tipo de operação (PIX, TED, boleto)
- Banco do destinatário
- Histórico recente da conta
- ... e dezenas de outros sinais

O modelo dá um **"score de risco" de 0% a 100%**. Quanto mais alto, mais parecido com fraudes anteriores.

### 👤 Filtro 3: Comportamento do Cliente (o conhecido da padaria)

Todo cliente tem um **perfil comportamental**: horários típicos de uso, valor médio, lugares onde costuma transferir, dispositivos que sempre usa.

Se de repente:
- Uma cliente que sempre faz PIX de R$ 50 fizer um de R$ 20.000 às 3h da manhã para uma conta nova → 🚨 alarme
- Um cliente que nunca acessa do exterior tentar logar de fora → 🚨 alarme

É como o dono da padaria que reconhece quando algo está estranho com você.

---

## 4. Diferenciais que nos Colocam à Frente

Além dos 3 filtros básicos, temos **proteções extras**:

### 🆕 Detector de Fraudes Novas
Existe uma "vacina" treinada **só com transações boas**. Se aparecer um padrão totalmente diferente de tudo que ela já viu, marca como suspeito — mesmo sem nunca ter visto aquela fraude antes. Isso protege contra **golpes recém-inventados**.

### 🕸️ Análise de Rede (Detector de Laranjas)
Mapeamos quem transfere para quem, formando uma "rede de relações". Quando detectamos:
- Várias contas mandando dinheiro para uma só, e essa única conta repassando rapidamente para outras → padrão clássico de **conta laranja**
- Dinheiro circulando em círculo (A→B→C→A) → tentativa de **lavagem**

### 🔄 Aprendizado Contínuo
Toda semana o sistema **se atualiza sozinho** com as transações novas. Se golpistas inventarem técnicas novas, o sistema percebe a mudança no padrão e se adapta — sem precisar de programador.

### 🤝 Aprendizado Compartilhado (futuro)
Permite que vários bancos **treinem juntos** um modelo melhor, **sem trocar dados de clientes** entre si — cada banco mantém os dados em casa, mas todos ficam mais espertos.

---

## 5. O Que Acontece com sua Transação

| Resultado | O que acontece |
|-----------|----------------|
| ✅ **Liberada (baixo risco)** | Transação processa normalmente, em milissegundos. Você nem percebe. |
| ⚠️ **Para análise (médio risco)** | Pode pedir confirmação extra (token, biometria, ligação). |
| ❌ **Bloqueada (alto risco)** | Transação é negada e um analista humano revisa. Você pode ser contatado pelo banco. |

**Importante**: nenhuma decisão é tomada "às cegas". O sistema sempre **explica por que** considerou uma transação suspeita — listando os 5 fatores que mais pesaram. Isso é exigido pelo **Banco Central** e pela **LGPD**.

---

## 6. Velocidade e Privacidade

### ⚡ Velocidade
A análise completa leva **menos de 100 milissegundos** (1/10 de segundo). Mais rápido do que você consegue piscar duas vezes. O cliente não percebe diferença na velocidade da operação.

### 🔒 Privacidade (LGPD)
- Seu **CPF nunca é guardado em texto puro** — usamos uma "impressão digital" (hash) para ele
- Os dados ficam protegidos e auditáveis
- Você tem direito de pedir **explicação** sobre qualquer decisão
- Conformidade total com a Lei Geral de Proteção de Dados

---

## 7. O Papel do Analista Humano

Quando o sistema marca uma transação como suspeita, ela vai para um **analista de fraude**. O analista:

1. Recebe a transação **com a explicação** do sistema
2. Pode aprovar, rejeitar, ou pedir mais dados
3. **Dá feedback** ao sistema (foi mesmo fraude? não foi?)
4. Esse feedback **alimenta o aprendizado** — o sistema fica mais preciso a cada decisão humana

É uma parceria: máquina filtra o volume, humano decide os casos difíceis.

---

## 8. Em Números

| Indicador | Nosso resultado |
|-----------|-----------------|
| Tempo de análise | < 100 milissegundos |
| Volume processável | Milhões de transações/dia |
| Precisão (de 100 alertas, quantos são fraudes reais) | 32% (4× melhor que modelo simples) |
| Detecção de fraudes | Detectamos 60% das fraudes na primeira camada |
| Conformidade BACEN | 100% |
| Conformidade LGPD | 100% |
| Auditável (Banco Central pode revisar) | Sim, tudo registrado |

---

## 9. Resumindo em 4 Frases

1. **Toda transação passa por 3 filtros** em menos de 1/10 de segundo
2. **Regras simples + inteligência artificial + análise comportamental** decidem juntas
3. **Você nem percebe** quando é normal; quando há suspeita, pode pedir confirmação extra
4. **Tudo é auditável e respeita a LGPD** — sem armazenamento indevido de dados pessoais

---

## 10. Perguntas Frequentes

**"Por que minha transação foi bloqueada?"**
Porque ela teve um padrão muito diferente do seu uso comum, ou caiu numa regra de risco. Você pode ligar pro banco — eles têm a explicação detalhada do sistema.

**"O sistema é justo?"**
Sim. Ele não usa cor, gênero, religião ou nada parecido. Só dados de transações. E toda decisão é explicada.

**"E se errar?"**
Erramos sim, especialmente em casos novos. Por isso existe o analista humano e o feedback contínuo. A cada erro, o sistema aprende.

**"Meu dinheiro fica seguro?"**
Em caso de fraude detectada, o dinheiro **não sai da conta**. Em caso de fraude que passou (raro), há o **MED do PIX** (Mecanismo Especial de Devolução) e seguros.

**"Vocês compartilham meus dados com outros bancos?"**
Não. O aprendizado compartilhado (quando ativo) troca apenas o "conhecimento" do modelo, nunca dados de clientes.

---

*Documento mantido pela Squad de Anti-Fraude. Última revisão: 26/04/2026.*
