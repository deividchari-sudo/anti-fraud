Feature: Detecção de Fraude em Tempo Real
  Como sistema de detecção de fraude
  Quero analisar transações em tempo real
  Para identificar atividades fraudulentas e proteger clientes

  Scenario: Detecção de fraude em transação PIX de alto valor
    Given uma transação PIX de R$ 10.000
    And o canal é "app"
    And o horário é 02:00 (madrugada)
    When a transação é analisada pelo modelo
    Then a probabilidade de fraude deve ser alta (> 0.7)
    And a transação deve ser marcada como suspeita

  Scenario: Detecção de fraude em múltiplas transações em curto período
    Given um cliente realiza 10 transações em 5 minutos
    And o valor total é R$ 50.000
    When as transações são analisadas
    Then pelo menos uma deve ser marcada como suspeita
    And o sistema deve alertar sobre comportamento anômalo

  Scenario: Transação legítima durante horário comercial
    Given uma transação TED de R$ 1.000
    And o canal é "web"
    And o horário é 10:00 (horário comercial)
    And o cliente tem histórico positivo
    When a transação é analisada
    Then a probabilidade de fraude deve ser baixa (< 0.3)
    And a transação deve ser aprovada

  Scenario: Login de autenticação com troca de senha
    Given uma tentativa de login com troca de senha
    And o CPF é do titular da conta
    When a autenticação é analisada
    Then a probabilidade de fraude deve ser muito baixa (< 0.1)
    And a autenticação deve ser permitida

  Scenario: Latência de predição deve ser inferior a 100ms
    Given uma transação típica
    When a predição é realizada
    Then o tempo de processamento deve ser menor que 100ms

  Scenario: Predição em lote para múltiplas transações
    Given 100 transações para análise
    When as predições são realizadas em lote
    Then todas as transações devem ser processadas
    And o tempo médio por transação deve ser menor que 100ms

  Scenario: Modelo não treinado retorna erro apropriado
    Given o modelo ML não foi carregado
    When uma requisição de predição é feita
    Then a API deve retornar status 503
    And a mensagem deve indicar que o modelo não está disponível

  Scenario: Validação de dados de entrada
    Given uma requisição com campos obrigatórios faltando
    When a predição é solicitada
    Then a API deve retornar status 422
    E os campos inválidos devem ser listados

  Scenario: Compatibilidade com diferentes canais
    Given transações dos canais "web", "app" e "api"
    When as transações são analisadas
    Then todas devem ser processadas corretamente
    E as features específicas de cada canal devem ser extraídas

  Scenario: Auditoria de decisões
    Given uma transação foi analisada
    When a decisão é tomada
    Then o ID da transação deve ser registrado
    E a probabilidade de fraude deve ser registrada
    E o timestamp da decisão deve ser registrado
