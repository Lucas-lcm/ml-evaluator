"""Componentes visuais reutilizáveis.

Cada função aqui recebe apenas contratos de `scr.core.schema` e desenha. Nenhuma
delas calcula métrica, treina ou lê arquivo — isso mantém a interface fina e
substituível, e concentra o que precisa de teste automatizado na camada de
domínio.

Diretriz visual (spec 2): textos em pt-BR e ausência de ícones. A hierarquia é
construída com títulos, espaçamento e tabelas, não com símbolos decorativos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import streamlit as st

from scr.core.model_registry import ModelSpec
from scr.ui.compat import IMAGE_AUTO_WIDTH, STRETCH

from scr.core import config
from scr.core.data_loader import profile_columns
from scr.core.formatting import format_number as core_format_number
from scr.core.formatting import format_integer, format_percentage
from scr.core.schema import (
    ClassMetricRow,
    ColumnSpec,
    Dataset,
    MetricRow,
    PredictionResult,
    TrainingResult,
)

# --------------------------------------------------------------------------- #
# Cabeçalhos e trilha de etapas
# --------------------------------------------------------------------------- #


ASSETS_DIR: Path = Path(__file__).resolve().parents[2] / "assets"
"""Pasta das ilustrações. Ficam versionadas no repositório para que a
ferramenta funcione em sala de aula sem internet."""

_LOGOS: Mapping[str, str] = {
    "pandas": "pandas_dados.svg",
    "sklearn": "sklearn_modelo.svg",
    "fluxo": "fluxo_supervisionado.svg",
}


def render_illustration(name: str, *, width: Any = IMAGE_AUTO_WIDTH) -> None:
    """Exibe uma ilustração do repositório, se ela existir.

    A ausência do arquivo nunca derruba a tela: a ilustração é apoio visual,
    não conteúdo. Um repositório clonado sem a pasta ``assets`` continua
    funcionando.

    Args:
        name: Chave da ilustração (``"pandas"``, ``"sklearn"``, ``"fluxo"``).
        width: Largura aceita pelo ``st.image`` da versão instalada — um inteiro
            em pixels, ``"stretch"``, ``"content"``. O padrão resolve a diferença
            entre as versões via :mod:`scr.ui.compat`; **não** passe ``None``,
            que a partir do Streamlit 1.49 é rejeitado.
    """
    filename = _LOGOS.get(name)
    if filename is None:
        return
    path = ASSETS_DIR / filename
    if not path.exists():
        return

    try:
        st.image(str(path), width=width)
    except Exception:
        # Última linha de defesa: se a versão instalada rejeitar o valor de
        # largura, a ilustração ainda aparece com o padrão da biblioteca. Uma
        # imagem decorativa jamais deve derrubar a página.
        st.image(str(path))


def render_page_title(title: str, description: str, *, logo: Optional[str] = None) -> None:
    """Desenha o título da página com a ilustração correspondente ao lado.

    Args:
        title: Título da página.
        description: Frase de contexto.
        logo: Chave da ilustração a exibir à direita, se houver.
    """
    if logo is None:
        st.title(title)
        st.write(description)
        return

    esquerda, direita = st.columns([3, 1])
    with esquerda:
        st.title(title)
        st.write(description)
    with direita:
        render_illustration(logo)


def render_step_header(title: str, description: str) -> None:
    """Desenha o título e a linha de orientação de uma etapa.

    Args:
        title: Título da etapa.
        description: Frase curta explicando o que se espera do usuário aqui.
    """
    st.subheader(title)
    st.caption(description)


def render_progress(current: int, total: int) -> None:
    """Mostra o avanço do fluxo como barra de progresso.

    Args:
        current: Etapa atual (base 1).
        total: Número total de etapas do fluxo.
    """
    st.progress(current / total, text=f"Progresso: etapa {current} de {total}")


def render_warnings(messages: Sequence[str], *, title: str = "Observações sobre os dados") -> None:
    """Exibe avisos não fatais de forma discreta e recolhível.

    Avisos são informação pedagógica: mostram ao aprendiz o que a ferramenta
    decidiu por ele. Ficam recolhidos para não competir com o conteúdo
    principal, mas nunca são omitidos.

    Args:
        messages: Mensagens a exibir.
        title: Rótulo do bloco recolhível.
    """
    if not messages:
        return
    with st.expander(f"{title} ({len(messages)})", expanded=False):
        for message in messages:
            st.write(f"- {message}")


# --------------------------------------------------------------------------- #
# Tabela e perfil de dados
# --------------------------------------------------------------------------- #


def render_dataset_summary(dataset: Dataset) -> None:
    """Resume o arquivo carregado em números grandes e legíveis.

    Args:
        dataset: Tabela carregada.
    """
    first, second, third, fourth = st.columns(4)
    first.metric("Linhas", format_integer(dataset.n_rows))
    second.metric("Colunas", str(dataset.n_cols))
    third.metric("Separador", _describe_separator(dataset.detected_separator))
    fourth.metric("Codificação", dataset.detected_encoding or "não se aplica")


def _describe_separator(separator: Optional[str]) -> str:
    """Traduz o separador detectado para um rótulo legível.

    Args:
        separator: Caractere separador, ou ``None`` para formatos não textuais.

    Returns:
        Nome do separador em pt-BR.
    """
    if separator is None:
        return "não se aplica"
    return {
        ",": "vírgula",
        ";": "ponto e vírgula",
        "\t": "tabulação",
        "|": "barra vertical",
        ":": "dois-pontos",
    }.get(separator, separator)


def render_table(dataset: Dataset) -> None:
    """Exibe a tabela carregada com paginação implícita do Streamlit.

    Args:
        dataset: Tabela carregada.
    """
    st.dataframe(dataset.frame, **STRETCH, height=380)


def render_column_profile(dataset: Dataset) -> None:
    """Mostra tipo, ausências e cardinalidade de cada coluna.

    Este é o momento de *exploração de dados* do fluxo: antes de escolher o
    alvo, o aprendiz precisa ver quais colunas estão furadas e quais têm
    variedade suficiente para serem previstas.

    Args:
        dataset: Tabela carregada.
    """
    profile = profile_columns(dataset.frame)
    frame = pd.DataFrame(
        [
            {
                "Coluna": column,
                "Tipo": info["tipo"],
                "Valores ausentes": info["ausentes"],
                "Ausentes (%)": info["percentual_ausente"],
                "Valores distintos": info["distintos"],
            }
            for column, info in profile.items()
        ]
    )
    st.dataframe(
        frame,
        **STRETCH,
        hide_index=True,
        column_config={
            "Ausentes (%)": st.column_config.ProgressColumn(
                "Ausentes (%)",
                help="Proporção de células vazias na coluna.",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            )
        },
    )


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #

_REGRESSION_HIGHLIGHTS: Sequence[str] = ("r2", "mae", "rmse")


def render_metrics(result: TrainingResult) -> None:
    """Apresenta o desempenho do modelo treinado.

    Em classificação (spec 3) a visão principal é a acurácia geral seguida da
    tabela por categoria com precisão, revocação e F1. É a leitura honesta: a
    média sozinha esconde o modelo que acerta a classe majoritária e erra a
    minoritária. As demais métricas continuam disponíveis, recolhidas.

    Args:
        result: Resultado do treino concluído.
    """
    if result.task_type == "classification":
        _render_classification_metrics(result)
    else:
        _render_regression_metrics(result)

    st.caption(
        "Todos os números vêm dos registros separados antes do treino, que o modelo nunca "
        "viu. É essa separação que impede um resultado otimista demais."
    )


def _render_classification_metrics(result: TrainingResult) -> None:
    """Desenha acurácia geral e o desempenho por categoria.

    Args:
        result: Resultado do treino de classificação.
    """
    by_key = {row.key: row for row in result.metrics}
    accuracy = by_key.get("accuracy")

    primeira, segunda, terceira = st.columns(3)
    if accuracy is not None:
        primeira.metric("Acurácia geral", accuracy.formatted, help=accuracy.explanation)
    segunda.metric(
        "Registros de treino",
        format_integer(result.n_train),
        help="Quantidade de registros usados para o modelo aprender.",
    )
    terceira.metric(
        "Registros de teste",
        format_integer(result.n_test),
        help="Registros separados antes do treino, usados para medir o desempenho.",
    )

    st.markdown("**Desempenho em cada categoria**")
    if result.per_class:
        st.dataframe(
            per_class_to_frame(result.per_class),
            **STRETCH,
            hide_index=True,
            column_config={
                "Categoria": st.column_config.TextColumn("Categoria", width="medium"),
                "Precisão": st.column_config.ProgressColumn(
                    "Precisão", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Revocação": st.column_config.ProgressColumn(
                    "Revocação", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "F1": st.column_config.ProgressColumn(
                    "F1", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Casos no teste": st.column_config.NumberColumn("Casos no teste", width="small"),
            },
        )
    else:
        st.info("Não foi possível calcular o desempenho por categoria neste conjunto de teste.")

    st.caption(
        "Precisão: entre os casos apontados como desta categoria, quantos realmente eram. "
        "Revocação: entre os casos que eram desta categoria, quantos o modelo encontrou. "
        "F1: equilíbrio entre as duas."
    )

    with st.expander("Outras métricas calculadas", expanded=False):
        st.dataframe(
            metrics_to_frame(result.metrics),
            **STRETCH,
            hide_index=True,
        )


def _render_regression_metrics(result: TrainingResult) -> None:
    """Desenha o painel de métricas de regressão.

    Args:
        result: Resultado do treino de regressão.
    """
    by_key = {row.key: row for row in result.metrics}
    colunas = st.columns(len(_REGRESSION_HIGHLIGHTS))
    for coluna, key in zip(colunas, _REGRESSION_HIGHLIGHTS):
        row = by_key.get(key)
        if row is not None:
            coluna.metric(row.label, row.formatted, help=row.explanation)

    st.markdown("**Todas as métricas calculadas no conjunto de teste**")
    st.dataframe(
        metrics_to_frame(result.metrics),
        **STRETCH,
        hide_index=True,
        column_config={
            "Métrica": st.column_config.TextColumn("Métrica", width="medium"),
            "Valor": st.column_config.TextColumn("Valor", width="small"),
            "Direção": st.column_config.TextColumn("Direção", width="small"),
            "O que significa": st.column_config.TextColumn("O que significa", width="large"),
        },
    )


def per_class_to_frame(rows: Sequence[ClassMetricRow]) -> pd.DataFrame:
    """Converte o desempenho por categoria em tabela exibível.

    Função pura, sem Streamlit, para poder ser verificada em teste.

    Args:
        rows: Desempenho por categoria.

    Returns:
        DataFrame com ``Categoria``, ``Precisão``, ``Revocação``, ``F1`` e
        ``Casos no teste``.
    """
    return pd.DataFrame(
        [
            {
                "Categoria": row.label,
                "Precisão": row.precision,
                "Revocação": row.recall,
                "F1": row.f1,
                "Casos no teste": row.support,
            }
            for row in rows
        ]
    )


def metrics_to_frame(rows: Sequence[MetricRow]) -> pd.DataFrame:
    """Converte as métricas em uma tabela pronta para exibição.

    Função pura, sem Streamlit, para poder ser verificada em teste.

    Args:
        rows: Métricas calculadas.

    Returns:
        DataFrame com colunas ``Métrica``, ``Valor``, ``Direção`` e
        ``O que significa``.
    """
    return pd.DataFrame(
        [
            {
                "Métrica": row.label,
                "Valor": row.formatted,
                "Direção": "maior é melhor" if row.higher_is_better else "menor é melhor",
                "O que significa": row.explanation,
            }
            for row in rows
        ]
    )


# --------------------------------------------------------------------------- #
# Formulário dinâmico de predição
# --------------------------------------------------------------------------- #


def render_prediction_inputs(
    column_specs: Sequence[ColumnSpec],
    previous: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Gera um campo de entrada para cada coluna usada no treino.

    O formulário é derivado dos dados: uma coluna numérica vira campo numérico
    com o intervalo observado no treino visível na ajuda; uma coluna categórica
    vira seletor com exatamente as categorias que o modelo conhece. O aprendiz
    percebe, pela própria interface, que o modelo só sabe responder dentro do
    universo que lhe foi mostrado.

    Args:
        column_specs: Especificações geradas no treino.
        previous: Valores digitados anteriormente, para preservar o formulário.

    Returns:
        Mapa ``{coluna: valor}`` com o que foi informado.
    """
    previous = previous or {}
    values: Dict[str, Any] = {}
    columns_per_row = 2

    for index in range(0, len(column_specs), columns_per_row):
        chunk = list(column_specs)[index : index + columns_per_row]
        slots = st.columns(len(chunk))
        for slot, spec in zip(slots, chunk):
            with slot:
                values[spec.name] = _render_single_input(spec, previous.get(spec.name))

    return values


def _render_single_input(spec: ColumnSpec, previous_value: Any) -> Any:
    """Desenha o widget adequado a uma coluna.

    Args:
        spec: Especificação da coluna.
        previous_value: Valor previamente informado, se houver.

    Returns:
        O valor informado pelo usuário.
    """
    if spec.kind == "numeric":
        low = spec.minimum if spec.minimum is not None else 0.0
        high = spec.maximum if spec.maximum is not None else 0.0
        help_text = f"No treino esta coluna variou de {low:g} a {high:g}."
        if spec.has_missing:
            help_text += " A coluna tinha valores ausentes, preenchidos pela mediana."

        default = float(previous_value) if isinstance(previous_value, (int, float)) else float(spec.default)
        # Passo proporcional à amplitude: evita incrementos absurdos em colunas
        # de escala grande e incrementos inúteis em colunas de escala pequena.
        span = abs(high - low)
        step = float(span / 100) if span > 0 else 1.0
        return st.number_input(
            spec.name,
            value=default,
            step=step,
            format="%.6g",
            help=help_text,
            key=f"entrada_{spec.name}",
        )

    options = list(spec.categories)
    help_text = f"O modelo conheceu {len(options)} categoria(s) desta coluna durante o treino."
    if not options:
        return st.text_input(spec.name, value=str(spec.default), help=help_text, key=f"entrada_{spec.name}")

    candidate = str(previous_value) if previous_value is not None else str(spec.default)
    index = options.index(candidate) if candidate in options else 0
    return st.selectbox(spec.name, options=options, index=index, help=help_text, key=f"entrada_{spec.name}")


# --------------------------------------------------------------------------- #
# Resultado da predição
# --------------------------------------------------------------------------- #


def render_prediction_result(result: PredictionResult, training: TrainingResult) -> None:
    """Apresenta a previsão junto com sua medida de confiança.

    Args:
        result: Resultado da predição.
        training: Resultado do treino que a originou.
    """
    left, right = st.columns([2, 1])

    with left:
        rotulo = "Categoria prevista" if training.task_type == "classification" else "Valor previsto"
        st.metric(f"{rotulo} para '{training.target}'", result.formatted_prediction)

    with right:
        if result.confidence is not None:
            st.metric("Nível de confiança", format_percentage(result.confidence, decimals=1))
        else:
            st.metric("Nível de confiança", "não disponível")

    if result.confidence is not None:
        st.progress(min(max(result.confidence, 0.0), 1.0))

    st.write(result.confidence_explanation)

    if result.class_probabilities:
        st.markdown("**Probabilidade atribuída a cada categoria**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Categoria": name,
                        "Probabilidade": value,
                    }
                    for name, value in result.class_probabilities.items()
                ]
            ),
            **STRETCH,
            hide_index=True,
            column_config={
                "Probabilidade": st.column_config.ProgressColumn(
                    "Probabilidade",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                )
            },
        )

    if result.interval is not None:
        low, high = result.interval
        st.markdown("**Faixa provável do valor real (95% de confiança)**")
        st.write(f"De {format_number(low)} até {format_number(high)}.")


def render_input_alerts(alerts: Sequence[str]) -> None:
    """Mostra avisos de extrapolação antes do resultado.

    Args:
        alerts: Mensagens geradas por ``predictor.detect_out_of_range``.
    """
    for alert in alerts:
        st.warning(alert)


def render_upload_help() -> None:
    """Explica as regras de upload em linguagem simples."""
    permitidas = ", ".join(sorted(extensao.lstrip(".") for extensao in config.ALLOWED_EXTENSIONS))
    st.caption(
        f"Formatos aceitos: {permitidas}. Tamanho máximo: {config.MAX_UPLOAD_MB} MB. "
        "Em arquivos de texto o separador é identificado automaticamente."
    )


def render_model_explanation(spec: ModelSpec) -> None:
    """Mostra a explicação do modelo em três perguntas objetivas.

    A spec 3 pediu explicações mais diretas. Em vez de um parágrafo, cada frase
    responde a uma pergunta que o aprendiz realmente faz antes de escolher.

    Args:
        spec: Entrada do catálogo referente ao modelo selecionado.
    """
    with st.container(border=True):
        st.markdown(f"**{spec.display_name}**")
        st.markdown(f"**O que faz:** {spec.summary}")
        st.markdown(f"**Como funciona:** {spec.how_it_works}")
        st.markdown(f"**Quando usar:** {spec.when_to_use}")


def format_number(value: float) -> str:
    """Formata um número no padrão pt-BR.

    Delega ao formatador do domínio para que métricas, previsões e textos da
    interface nunca divirjam entre si.

    Args:
        value: Valor a formatar.

    Returns:
        Texto pronto para exibição.
    """
    return core_format_number(value)
