"""Página 1 — Dados e treinamento (etapas 1 a 3).

Percurso didático desenhado aqui:

    Etapa 1  Carregar e **olhar** os dados. Antes de modelar, entender.
    Etapa 2  Decidir o problema, o alvo, o modelo e as colunas — nessa ordem.
    Etapa 3  Ler o resultado por classe, não só a média.

A ordem da etapa 2 mudou na spec 3 e a mudança é pedagógica, não cosmética:
começar por "o que você quer resolver" ancora todas as decisões seguintes. Quem
começa escolhendo a coluna alvo tende a modelar o que a planilha oferece, em vez
do que a pergunta exige.

Cada etapa permite voltar, porque aprender modelagem é justamente refazer a
escolha anterior e observar o efeito na métrica.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from scr.core import config, model_registry, trainer
from scr.core.data_loader import DataLoadError, load_tabular_file
from scr.core.formatting import format_number
from scr.core.schema import Dataset, TaskType, TrainingResult
from scr.core.trainer import TrainingError
from scr.ui import components, state
from scr.ui.compat import STRETCH

# --------------------------------------------------------------------------- #
# Camadas de cache
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, max_entries=5)
def _load_dataset(payload: bytes, filename: str, mime_type: Optional[str]) -> Dataset:
    """Lê e valida o arquivo enviado, guardando o resultado em cache.

    O cache é indexado pelo conteúdo em bytes: reenviar o mesmo arquivo não
    refaz o parsing, e trocar de arquivo invalida a entrada automaticamente.

    Args:
        payload: Conteúdo bruto do arquivo.
        filename: Nome original do arquivo.
        mime_type: Tipo MIME declarado pelo navegador.

    Returns:
        O :class:`Dataset` carregado.

    Raises:
        DataLoadError: Repassado da camada de ingestão.
    """
    return load_tabular_file(filename, payload, mime_type)


@st.cache_resource(show_spinner=False, max_entries=3)
def _train_cached(
    _frame: pd.DataFrame,
    signature: str,
    target: str,
    features: Tuple[str, ...],
    task_type: TaskType,
    model_key: str,
    test_size: float,
) -> TrainingResult:
    """Treina o modelo reaproveitando o resultado de configurações idênticas.

    ``_frame`` começa com sublinhado para que o Streamlit **não** tente calcular
    o hash da tabela inteira; a identidade dos dados é carregada por
    ``signature``, que já é o hash do arquivo de origem.

    Args:
        _frame: Tabela de treino (excluída do cálculo de hash do cache).
        signature: Impressão digital do arquivo carregado.
        target: Coluna alvo.
        features: Colunas de entrada.
        task_type: Tipo de tarefa.
        model_key: Modelo escolhido.
        test_size: Proporção reservada para teste.

    Returns:
        O :class:`TrainingResult` do ajuste.

    Raises:
        TrainingError: Repassado da camada de treino.
    """
    return trainer.train_model(
        _frame,
        target=target,
        features=list(features),
        task_type=task_type,
        model_key=model_key,
        test_size=test_size,
    )


@st.cache_data(show_spinner=False, max_entries=10)
def _suggest_features_cached(
    _frame: pd.DataFrame,
    signature: str,
    target: str,
    task_type: TaskType,
) -> Tuple[List[str], List[str]]:
    """Sugere colunas de entrada, com a checagem de vazamento em cache.

    A detecção de vazamento percorre todas as colunas e é cara demais para rodar
    a cada reexecução do script — o que, em Streamlit, significa a cada clique.

    Args:
        _frame: Tabela carregada (excluída do hash do cache).
        signature: Impressão digital do arquivo.
        target: Coluna alvo.
        task_type: Tipo de tarefa.

    Returns:
        Tupla ``(features_sugeridas, motivos_de_descarte)``.
    """
    return trainer.suggest_features(_frame, target, task_type)


# --------------------------------------------------------------------------- #
# Etapa 1 — Carregar o arquivo
# --------------------------------------------------------------------------- #


def _render_step_1() -> None:
    """Desenha a etapa de carregamento e visualização da tabela."""
    components.render_step_header(
        state.TRAIN_STEP_TITLES[1],
        "Envie a planilha com os dados históricos. Cada linha deve ser um caso já "
        "conhecido e cada coluna, uma informação sobre ele.",
    )

    uploaded = st.file_uploader(
        "Arquivo de dados",
        type=[extensao.lstrip(".") for extensao in sorted(config.ALLOWED_EXTENSIONS)],
        accept_multiple_files=False,
        key="uploader_arquivo",
    )
    components.render_upload_help()

    if uploaded is not None:
        payload = uploaded.getvalue()
        signature = state.compute_signature(payload, uploaded.name)

        if signature != st.session_state["dataset_signature"]:
            # Arquivo novo: todo o estado derivado do anterior perde a validade.
            state.reset_dataset()
            try:
                with st.spinner("Lendo o arquivo e identificando o separador..."):
                    dataset = _load_dataset(payload, uploaded.name, uploaded.type)
            except DataLoadError as error:
                st.error(str(error))
                return
            st.session_state["dataset"] = dataset
            st.session_state["dataset_signature"] = signature

    dataset = state.current_dataset()
    if dataset is None:
        st.info("Nenhum arquivo carregado ainda. Envie um arquivo para continuar.")
        return

    st.markdown(f"**Arquivo carregado:** {dataset.source_name}")
    components.render_dataset_summary(dataset)

    # A spec 3 removeu o bloco recolhível de observações desta etapa: o que
    # importa aqui é ver a tabela. Os avisos relevantes reaparecem na etapa 2,
    # já ligados às decisões que o aprendiz precisa tomar.

    tabela, perfil = st.tabs(["Tabela de dados", "Perfil das colunas"])
    with tabela:
        components.render_table(dataset)
    with perfil:
        components.render_column_profile(dataset)

    st.divider()
    _, direita = st.columns([3, 1])
    with direita:
        st.button(
            "Avançar",
            type="primary",
            **STRETCH,
            key="botao_avancar_1",
            on_click=state.go_to_train_step,
            args=(2,),
        )


# --------------------------------------------------------------------------- #
# Etapa 2 — Configurar o experimento
#
# Ordem definida pela spec 3: problema → alvo → modelo → colunas.
# --------------------------------------------------------------------------- #


def _resolve_task_type() -> TaskType:
    """Pergunta primeiro qual problema se quer resolver.

    Returns:
        O tipo de tarefa escolhido.
    """
    options: List[TaskType] = ["classification", "regression"]
    current = st.session_state.get("task_type") or "classification"
    index = options.index(current) if current in options else 0

    chosen = st.radio(
        "1. Que problema você quer resolver?",
        options=options,
        index=index,
        horizontal=True,
        format_func=lambda item: config.TASK_LABELS[item],
        key="seletor_tarefa",
        help=(
            "Classificação prevê uma categoria, como aprovado ou reprovado. "
            "Regressão prevê um número, como preço ou temperatura."
        ),
    )

    if chosen != st.session_state.get("task_type"):
        st.session_state["task_type"] = chosen
        st.session_state["model_key"] = None
        st.session_state["features"] = []
        state.reset_training()

    return chosen


def _resolve_target(dataset: Dataset, task_type: TaskType) -> str:
    """Desenha o seletor de coluna alvo e confere a coerência com o problema.

    Args:
        dataset: Tabela carregada.
        task_type: Tipo de problema já escolhido.

    Returns:
        Nome da coluna alvo escolhida.
    """
    columns = list(dataset.frame.columns)
    previous = st.session_state.get("target")
    index = columns.index(previous) if previous in columns else 0

    target = st.selectbox(
        "2. Qual coluna você quer prever?",
        options=columns,
        index=index,
        key="seletor_alvo",
        help="Esta é a resposta que o modelo vai aprender a reproduzir.",
    )

    if target != previous:
        # Mudar o alvo invalida colunas de entrada e métricas.
        st.session_state["target"] = target
        st.session_state["features"] = []
        state.reset_training()

    # A inferência agora serve para conferir a escolha, não para dirigi-la.
    suggested = trainer.infer_task_type(dataset.frame[target])
    if suggested != task_type:
        st.caption(
            f"Atenção: pelo conteúdo de '{target}', esta coluna se parece mais com um caso de "
            f"{config.TASK_LABELS[suggested].lower()}. Confirme se a escolha do passo 1 é mesmo "
            "a que você quer."
        )

    return target


def _resolve_model(task_type: TaskType) -> str:
    """Desenha o seletor de modelo e a explicação da opção escolhida.

    Args:
        task_type: Tipo de tarefa escolhido.

    Returns:
        A chave do modelo escolhido.
    """
    options = model_registry.model_options(task_type)
    keys = list(options.keys())
    previous = st.session_state.get("model_key")
    default = previous if previous in keys else model_registry.default_model_key(task_type)

    model_key = st.selectbox(
        "3. Qual modelo deve aprender com estes dados?",
        options=keys,
        index=keys.index(default),
        format_func=lambda key: options[key],
        key="seletor_modelo",
    )

    if model_key != previous:
        st.session_state["model_key"] = model_key
        state.reset_training()

    components.render_model_explanation(model_registry.get_model(task_type, model_key))
    return model_key


def _resolve_features(dataset: Dataset, target: str, task_type: TaskType) -> Sequence[str]:
    """Desenha a seleção de colunas de entrada, já com sugestão automática.

    Args:
        dataset: Tabela carregada.
        target: Coluna alvo.
        task_type: Tipo de tarefa.

    Returns:
        As colunas de entrada escolhidas.
    """
    suggested, reasons = _suggest_features_cached(
        dataset.frame, st.session_state["dataset_signature"], target, task_type
    )

    if not st.session_state.get("features"):
        st.session_state["features"] = list(suggested)

    available = [column for column in dataset.frame.columns if column != target]
    current = [column for column in st.session_state["features"] if column in available]

    features = st.multiselect(
        "4. Quais colunas o modelo pode usar como pista?",
        options=available,
        default=current,
        key="seletor_features",
        help="Remova colunas que não estariam disponíveis no momento real da decisão.",
    )

    if list(features) != list(st.session_state["features"]):
        st.session_state["features"] = list(features)
        state.reset_training()

    # Vazamento é grave o bastante para aparecer em destaque, não recolhido.
    for reason in reasons:
        if "vazamento" in reason:
            st.warning(reason)
    outros = [reason for reason in reasons if "vazamento" not in reason]
    components.render_warnings(outros, title="Colunas desmarcadas automaticamente")

    return features


def _render_step_2() -> None:
    """Desenha a etapa de configuração do experimento."""
    dataset = state.current_dataset()
    if dataset is None:
        st.info("Carregue um arquivo na etapa anterior para continuar.")
        st.button("Voltar", on_click=state.go_to_train_step, args=(1,), key="voltar_sem_dados")
        return

    components.render_step_header(
        state.TRAIN_STEP_TITLES[2],
        "Comece pelo problema que você quer resolver. As demais escolhas decorrem dele.",
    )

    task_type = _resolve_task_type()
    target = _resolve_target(dataset, task_type)
    model_key = _resolve_model(task_type)
    features = _resolve_features(dataset, target, task_type)

    with st.expander("Configuração avançada da avaliação", expanded=False):
        lower, upper = config.TEST_SIZE_BOUNDS
        test_size = st.slider(
            "Percentual dos dados reservado para teste",
            min_value=int(lower * 100),
            max_value=int(upper * 100),
            value=int(st.session_state["test_size"] * 100),
            step=5,
            key="slider_teste",
            help=(
                "Esses registros ficam de fora do treino e servem para medir o desempenho "
                "em dados nunca vistos."
            ),
        ) / 100
        if abs(test_size - st.session_state["test_size"]) > 1e-9:
            st.session_state["test_size"] = test_size
            state.reset_training()

    st.divider()
    esquerda, _, direita = st.columns([1, 2, 1])
    with esquerda:
        st.button(
            "Voltar",
            **STRETCH,
            key="botao_voltar_2",
            on_click=state.go_to_train_step,
            args=(1,),
        )
    with direita:
        treinar = st.button(
            "Treinar modelo",
            type="primary",
            **STRETCH,
            key="botao_treinar",
            disabled=not features,
        )

    if not features:
        st.warning("Selecione ao menos uma coluna de entrada para habilitar o treinamento.")
        return

    if treinar:
        _run_training(dataset, target, tuple(features), task_type, model_key)


def _run_training(
    dataset: Dataset,
    target: str,
    features: Tuple[str, ...],
    task_type: TaskType,
    model_key: str,
) -> None:
    """Executa o treino com indicador de carregamento e trata falhas.

    Args:
        dataset: Tabela carregada.
        target: Coluna alvo.
        features: Colunas de entrada.
        task_type: Tipo de tarefa.
        model_key: Modelo escolhido.
    """
    display_name = model_registry.get_model(task_type, model_key).display_name

    try:
        with st.spinner(f"Treinando {display_name}. Isso pode levar alguns instantes..."):
            result = _train_cached(
                dataset.frame,
                st.session_state["dataset_signature"],
                target,
                features,
                task_type,
                model_key,
                float(st.session_state["test_size"]),
            )
    except TrainingError as error:
        st.error(str(error))
        return
    except Exception as error:  # rede de segurança: a UI nunca deve quebrar
        st.error(f"Ocorreu um erro inesperado durante o treinamento: {error}")
        return

    st.session_state["training_result"] = result
    state.reset_prediction()
    state.go_to_train_step(3)
    st.rerun()


# --------------------------------------------------------------------------- #
# Etapa 3 — Resultado
# --------------------------------------------------------------------------- #


def _render_step_3() -> None:
    """Desenha a etapa de leitura das métricas."""
    result = state.current_training_result()
    if result is None:
        st.info("Nenhum modelo treinado nesta sessão. Volte e execute o treinamento.")
        st.button("Voltar", on_click=state.go_to_train_step, args=(2,), key="voltar_sem_modelo")
        return

    components.render_step_header(
        state.TRAIN_STEP_TITLES[3],
        "Veja o desempenho em cada categoria, e não apenas a média geral.",
    )

    st.markdown(
        f"Modelo **{result.model_display_name}** treinado para prever **{result.target}** "
        f"({config.TASK_LABELS[result.task_type].lower()}), usando {len(result.features)} coluna(s) "
        f"de entrada em {format_number(result.training_seconds, decimals=2)} segundo(s)."
    )

    components.render_metrics(result)
    components.render_warnings(result.warnings, title="Observações sobre o treinamento")

    st.divider()
    esquerda, meio, direita = st.columns([1, 1, 1])
    with esquerda:
        st.button(
            "Voltar",
            **STRETCH,
            key="botao_voltar_3",
            on_click=state.go_to_train_step,
            args=(2,),
        )
    with meio:
        st.button(
            "Refazer configuração",
            **STRETCH,
            key="botao_refazer",
            on_click=state.go_to_train_step,
            args=(2,),
        )
    with direita:
        if st.button(
            "Avaliar novos casos",
            type="primary",
            **STRETCH,
            key="botao_ir_avaliacao",
        ):
            state.switch_to_prediction_page()


# --------------------------------------------------------------------------- #
# Ponto de entrada da página
# --------------------------------------------------------------------------- #


def render() -> None:
    """Renderiza a página de dados e treinamento na etapa corrente."""
    state.init_state()

    components.render_page_title(
        "Dados e treinamento",
        "Carregue os dados históricos, escolha o que deve ser previsto e treine um "
        "modelo de aprendizado supervisionado.",
        logo="pandas",
    )

    step = int(st.session_state["train_step"])
    components.render_progress(step, state.LAST_TRAIN_STEP)

    if step == 1:
        _render_step_1()
    elif step == 2:
        _render_step_2()
    else:
        _render_step_3()
