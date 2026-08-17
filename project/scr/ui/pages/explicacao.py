"""Página inicial — Como funciona.

Tela de abertura e de referência conceitual: explica o ciclo do aprendizado supervisionado, o
vocabulário que aparece nas outras telas e como ler cada métrica. Não executa
nada — é o lugar onde o aprendiz volta quando um termo da interface não fez
sentido.

O conteúdo é declarado como dados (listas de tuplas) em vez de blocos de
markdown soltos, para que o glossário de métricas venha do mesmo dicionário que
a etapa 3 usa. Explicação divergente entre a tela de resultado e a tela de
referência seria pior do que não ter a referência.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd
import streamlit as st

from scr.core import model_registry
from scr.core.metrics import METRIC_GLOSSARY
from scr.ui import components, state
from scr.ui.compat import STRETCH

_CICLO: Sequence[Tuple[str, str]] = (
    (
        "1. Dados históricos",
        "Uma tabela de casos já resolvidos. Cada linha é um caso, cada coluna é uma "
        "informação sobre ele, e uma dessas colunas é a resposta que se quer prever.",
    ),
    (
        "2. Separação em treino e teste",
        "Parte dos registros é escondida do modelo. Sem essa reserva não há como saber "
        "se ele aprendeu ou apenas decorou.",
    ),
    (
        "3. Treino",
        "O modelo procura padrões que ligam as colunas de entrada à resposta. Tudo que "
        "ele aprende vem exclusivamente da parte de treino.",
    ),
    (
        "4. Avaliação",
        "O modelo responde os casos escondidos e comparamos com a resposta verdadeira. "
        "É daí que saem as métricas da etapa 3.",
    ),
    (
        "5. Uso em casos novos",
        "Só depois de avaliado o modelo é usado para responder situações inéditas — "
        "sempre acompanhado de uma medida de confiança.",
    ),
)

_VOCABULARIO: Sequence[Tuple[str, str]] = (
    (
        "Aprendizado supervisionado",
        "Aprender a partir de exemplos em que a resposta certa já é conhecida.",
    ),
    (
        "Alvo",
        "A coluna que se quer prever. Também chamada de variável dependente ou y.",
    ),
    (
        "Colunas de entrada",
        "As informações usadas como pista para chegar à resposta. Também chamadas de "
        "variáveis, atributos, features ou X.",
    ),
    (
        "Classificação",
        "Prever a qual categoria o caso pertence: aprovado ou reprovado, tipo A, B ou C.",
    ),
    (
        "Regressão",
        "Prever um número em escala contínua: preço, temperatura, tempo de entrega.",
    ),
    (
        "Conjunto de treino e conjunto de teste",
        "As duas partes em que os dados são divididos. O teste nunca participa do "
        "aprendizado; é a prova final.",
    ),
    (
        "Sobreajuste (decorar os dados)",
        "Quando o modelo memoriza os casos de treino em vez de aprender o padrão. "
        "Aparece como desempenho ótimo no treino e ruim no teste.",
    ),
    (
        "Vazamento de alvo",
        "Quando uma coluna de entrada já contém a resposta, direta ou indiretamente. "
        "A métrica sobe para perto de 100% e o modelo falha na vida real. A ferramenta "
        "desmarca automaticamente as colunas suspeitas na etapa 2.",
    ),
    (
        "Extrapolação",
        "Pedir uma previsão para um caso fora da faixa de valores que o modelo viu no "
        "treino. Nenhuma métrica de teste cobre essa situação.",
    ),
    (
        "Valores ausentes",
        "Células vazias na tabela. São preenchidas automaticamente pela mediana "
        "(colunas numéricas) ou pelo valor mais frequente (colunas de texto).",
    ),
)

_METRICAS_PRINCIPAIS: Sequence[Tuple[str, str]] = (
    ("Acurácia", "accuracy"),
    ("Precisão", "precision"),
    ("Revocação", "recall"),
    ("F1", "f1"),
    ("R²", "r2"),
    ("Erro absoluto médio (MAE)", "mae"),
    ("Raiz do erro quadrático médio (RMSE)", "rmse"),
    ("Erro percentual absoluto médio (MAPE)", "mape"),
)


def _glossary_frame(entries: Sequence[Tuple[str, str]], first_column: str) -> pd.DataFrame:
    """Converte pares termo/definição em tabela.

    Args:
        entries: Sequência de pares ``(termo, definição)``.
        first_column: Rótulo da primeira coluna.

    Returns:
        DataFrame de duas colunas pronto para exibição.
    """
    return pd.DataFrame(
        [{first_column: termo, "O que significa": definicao} for termo, definicao in entries]
    )


def _model_catalog_frame() -> pd.DataFrame:
    """Reúne o catálogo de modelos em uma tabela comparativa.

    Returns:
        DataFrame com problema, modelo e as três frases de explicação.
    """
    linhas: List[dict] = []
    rotulos = {"classification": "Classificação", "regression": "Regressão"}

    for task_type, rotulo in rotulos.items():
        for spec in model_registry.list_models(task_type):
            linhas.append(
                {
                    "Problema": rotulo,
                    "Modelo": spec.display_name,
                    "O que faz": spec.summary,
                    "Como funciona": spec.how_it_works,
                    "Quando usar": spec.when_to_use,
                }
            )

    return pd.DataFrame(linhas)


def render() -> None:
    """Renderiza a página de referência conceitual."""
    state.init_state()

    components.render_page_title(
        "Como funciona",
        "Esta é a tela de abertura. Ela explica os conceitos e o vocabulário usados nas "
        "outras telas — volte aqui sempre que um termo não fizer sentido.",
        logo="sklearn",
    )

    esquerda, _ = st.columns([1, 2])
    with esquerda:
        if st.button("Começar: carregar meus dados", type="primary", **STRETCH, key="botao_comecar"):
            state.switch_to_training_page()

    st.divider()

    st.subheader("O ciclo do aprendizado supervisionado")
    components.render_illustration("fluxo")
    for titulo, texto in _CICLO:
        with st.container(border=True):
            st.markdown(f"**{titulo}**")
            st.write(texto)

    st.divider()

    st.subheader("Vocabulário")
    st.dataframe(
        _glossary_frame(_VOCABULARIO, "Termo"),
        **STRETCH,
        hide_index=True,
        column_config={
            "Termo": st.column_config.TextColumn("Termo", width="medium"),
            "O que significa": st.column_config.TextColumn("O que significa", width="large"),
        },
    )

    st.divider()

    st.subheader("Como ler as métricas")
    st.write(
        "Nenhuma métrica sozinha descreve um modelo. Em classificação, a acurácia é uma "
        "média que pode esconder um desempenho ruim na categoria menos frequente — por "
        "isso a etapa 3 mostra precisão, revocação e F1 para cada categoria separadamente."
    )
    st.dataframe(
        _glossary_frame(
            [(rotulo, METRIC_GLOSSARY[chave]) for rotulo, chave in _METRICAS_PRINCIPAIS],
            "Métrica",
        ),
        **STRETCH,
        hide_index=True,
        column_config={
            "Métrica": st.column_config.TextColumn("Métrica", width="medium"),
            "O que significa": st.column_config.TextColumn("O que significa", width="large"),
        },
    )

    st.divider()

    st.subheader("Os modelos disponíveis")
    st.write(
        "Cada família de modelo carrega uma suposição diferente sobre como os dados se "
        "comportam. Não existe um melhor modelo universal: a escolha depende do problema, "
        "do volume de dados e de quanto interpretar o resultado importa."
    )
    st.dataframe(
        _model_catalog_frame(),
        **STRETCH,
        hide_index=True,
        column_config={
            "Problema": st.column_config.TextColumn("Problema", width="small"),
            "Modelo": st.column_config.TextColumn("Modelo", width="medium"),
        },
    )

    st.divider()

    st.subheader("Quer ver o código por trás disto?")
    st.write(
        "O notebook 'ML_Evaluator_Fluxo_Supervisionado.ipynb', na pasta 'notebooks' do "
        "projeto, refaz exatamente este fluxo em Python, passo a passo, e pode ser aberto "
        "no Google Colab. É o mesmo caminho que esta ferramenta percorre por baixo dos panos."
    )
