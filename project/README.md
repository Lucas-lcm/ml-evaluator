# ML Evaluator

Ferramenta didática em Streamlit para que pessoas **sem formação técnica**
percorram um ciclo completo de aprendizado supervisionado: carregar dados,
escolher o que prever, treinar um modelo, ler as métricas e avaliar casos novos
com uma medida explícita de confiança.

Implementa as especificações `project/spec/spec_1_v1.md` a `spec_4_v1.md`.

## Como executar

```bash
pip install -r project/requirements.txt
streamlit run project/app.py
```

Testes:

```bash
pytest project/tests
```

Para publicar no Streamlit Community Cloud, siga [`DEPLOY.md`](DEPLOY.md).

## Arquitetura

Separação estrita entre domínio e apresentação. Nada em `scr/core` importa
Streamlit, o que torna toda a regra de negócio testável como Python comum.

```
ml_evaluator/                 Raiz do repositório (o que vai para o GitHub)
  .gitignore
  .python-version             Python 3.11
  .streamlit/config.toml      Config lida pelo Streamlit Cloud
  requirements.txt            Cópia da de project/, exigida na raiz pelo Cloud
project/
  app.py                      Ponto de entrada e roteamento (st.navigation)
  DEPLOY.md                   Passo a passo de publicação
  requirements.txt
  .streamlit/config.toml      Tema escuro, menu de dev oculto, limite de upload
  assets/                     Ilustrações SVG e favicon (funcionam offline)
  notebooks/                  Notebook Colab com o mesmo fluxo em código
  scr/
    core/                     Domínio — sem Streamlit
      config.py               Limites, sementes e constantes
      schema.py               Contratos entre camadas (dataclasses)
      formatting.py           Formatação numérica pt-BR
      data_loader.py          Ingestão csv/txt/xlsx/json e detecção de separador
      model_registry.py       Catálogo de modelos e explicações
      metrics.py              Cálculo e glossário das métricas
      trainer.py              Pré-processamento, split e treino
      predictor.py            Predição individual e confiança
    ui/                       Apresentação — só desenha
      state.py                Mapa de st.session_state e navegação de etapas
      compat.py               Compatibilidade com a API de layout do Streamlit
      components.py           Componentes reutilizáveis
      pages/
        treinamento.py        Etapas 1 a 3
        predicao.py           Etapas 4 e 5
        explicacao.py         Referência de conceitos e glossário
  tests/                      Testes do domínio (pytest)
```

## Fluxo do usuário

Três telas, navegáveis pelo nome na barra lateral, com cinco etapas encadeadas e
retorno permitido em todas elas.

**Como funciona** *(tela inicial)*

Referência conceitual: ciclo do aprendizado supervisionado, vocabulário, como
ler cada métrica e a tabela comparativa dos modelos. É a abertura porque quem
chega pela primeira vez não sabe o que é alvo, treino ou métrica — pedir que ele
escolha uma coluna antes disso é pedir uma decisão sem o vocabulário para
tomá-la. Um botão "Começar" leva ao upload.

**Dados e treinamento**

1. Carregar o arquivo e conferir a tabela, com perfil de cada coluna.
2. Definir, nesta ordem, o problema a resolver, a coluna alvo, o modelo e as
   colunas de entrada. Cada modelo é explicado em três frases: o que faz, como
   funciona e quando usar.
3. Ler o resultado: acurácia geral e desempenho por categoria (precisão,
   revocação e F1), com botão que leva direto à avaliação de novos casos.

**Avaliar novos casos**

4. Preencher um formulário gerado a partir das colunas usadas no treino.
5. Ler a previsão junto da confiança e dos avisos de extrapolação.

## Catálogo de modelos

Reduzido na spec 3 a quatro famílias de viés indutivo claramente distinto.

| Problema | Modelos |
| --- | --- |
| Classificação | Regressão Logística, Floresta Aleatória, Árvore de Decisão, K Vizinhos (KNN) |
| Regressão | Regressão Linear, Floresta Aleatória, Árvore de Decisão |

Cada entrada declara nome completo, as três frases de explicação e uma fábrica
que devolve sempre uma instância nova.

## Notebook para estudo (spec 4)

`notebooks/ML_Evaluator_Fluxo_Supervisionado.ipynb` refaz as cinco etapas em
código, pronto para abrir no Google Colab: carga, seleção de X e y, definição do
tipo de problema, escolha do modelo (mesma lista da ferramenta) e tabela de
métricas, mais matriz de confusão ou gráfico de resíduos e uma predição com
medida de confiança. Usa conjuntos embutidos no scikit-learn, sem download.

## Como a confiança é calculada

| Tarefa | Origem | Leitura |
| --- | --- | --- |
| Classificação | `predict_proba` da classe vencedora | Probabilidade que o modelo atribui à categoria prevista |
| Regressão (comitês) | Discordância entre as árvores combinada ao erro residual | Intervalo de 95% em torno da previsão |
| Regressão (demais) | Desvio-padrão dos resíduos no conjunto de teste | Intervalo de 95% em torno da previsão |

Em regressão o escore de confiança é a largura do intervalo comparada à
amplitude do alvo: uma faixa estreita diante de um alvo muito variável indica
previsão útil.

## Decisões de segurança

- Extensão, tamanho e MIME validados antes de qualquer parsing; o limite de
  25 MB é aplicado no servidor (`.streamlit/config.toml`) **e** no domínio
  (`scr/core/config.py`).
- Formatos executáveis (`.pkl`, `.joblib`) recusados por princípio —
  desserializar objeto arbitrário equivale a execução remota de código.
- `eval`, `exec` e `pickle.load` não aparecem em nenhum ponto do código.
- Nomes de arquivo têm componentes de caminho removidos (defesa contra path
  traversal) e nomes de coluna são higienizados de caracteres de controle.
- Tetos de linhas e colunas protegem a sessão contra exaustão de memória.
- Nenhum segredo é necessário; se algum for adicionado no futuro, deve vir de
  `st.secrets` ou variável de ambiente, nunca do código.
- Os dados enviados permanecem apenas em memória, no escopo da sessão.

## Notas pedagógicas

- **Nenhuma métrica isolada.** A spec 2 pede o painel completo justamente
  porque acurácia esconde desequilíbrio de classes e R² esconde erros grandes em
  poucos casos. Cada métrica vem com sua explicação e com a indicação de se
  maior ou menor é melhor.
- **Vazamento de dados.** Imputação, padronização e codificação vivem dentro do
  `Pipeline`, portanto são aprendidas só no treino. É o erro mais comum e mais
  invisível em projetos iniciantes.
- **Extrapolação.** Os limites dos campos vêm dos dados de treino; informar
  valor fora da faixa ou categoria inédita gera aviso explícito, porque nenhuma
  métrica de teste cobre esse cenário.
- **Vazamento de alvo.** Colunas que praticamente reproduzem a resposta são
  detectadas (informação mútua normalizada em classificação, correlação de
  Pearson em regressão) e desmarcadas por padrão, com aviso em destaque. Foi a
  causa de um resultado de 100% de acurácia observado em teste: a métrica estava
  correta, os dados é que continham a resposta.
- **Invalidação honesta.** Mudar alvo, modelo ou colunas descarta o resultado
  anterior. Exibir métrica calculada com outra configuração seria a pior falha
  possível de uma ferramenta de ensino.
- **Descartes comunicados.** Colunas constantes, vazias ou de cardinalidade
  altíssima são removidas com o motivo visível, e não em silêncio.

## Verificação das métricas

`tests/test_paridade_sklearn.py` compara os números da ferramenta com o fluxo
canônico feito à mão (`train_test_split` → `fit` → `accuracy_score`) e exige
igualdade. Também checa que a acurácia é a média ponderada das revocações por
classe, que o F1 é a média harmônica de precisão e revocação, e que dois treinos
com a mesma configuração devolvem exatamente o mesmo resultado.

## Compatibilidade de versões do Streamlit

A partir da versão 1.49 o Streamlit trocou `use_container_width=True` por
`width="stretch"` e passou a validar o parâmetro `width` — `None` deixou de ser
aceito em `st.image`. `scr/ui/compat.py` detecta a versão instalada e expõe
`STRETCH` e `IMAGE_AUTO_WIDTH`; o resto da interface nunca escreve o parâmetro
diretamente. Assim a mesma cópia do projeto roda em máquinas de aluno com
versões diferentes.

## Publicação

`DEPLOY.md` traz o passo a passo completo. Dois pontos que costumam surpreender:

- O Streamlit lê `.streamlit/config.toml` a partir do **diretório de trabalho**,
  não do diretório do script. Como o Cloud executa da raiz do repositório e o
  desenvolvimento acontece dentro de `project/`, há duas cópias idênticas do
  arquivo — e `tests/test_deploy_config.py` falha se elas divergirem.
- O menu de opções de desenvolvedor some com `client.toolbarMode = "viewer"`.
  O botão "Manage app" do Cloud é outra coisa e só aparece para o dono do app.

## Casos de borda tratados

CSV brasileiro (`;` com vírgula decimal), arquivos em latin-1, colunas duplicadas
ou sem nome, linhas malformadas, alvo com valores ausentes, classe rara que
impede estratificação, modelo que não converge, categoria desconhecida na
predição, base menor que o mínimo estatístico e métricas indefinidas — todos com
mensagem em pt-BR e sem derrubar a interface.
