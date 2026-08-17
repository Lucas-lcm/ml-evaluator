"""Página 2 — Avaliar novos casos (etapas 4 e 5).

Percurso didático desenhado aqui:

    Etapa 4  Descrever um caso que o modelo nunca viu, campo a campo.
    Etapa 5  Ler a previsão **e a incerteza que a acompanha**.

O ponto pedagógico da página é o segundo: mostrar que um modelo não devolve
verdade, devolve uma aposta com grau de confiança — e que essa confiança cai
quando o caso informado se afasta do que existia nos dados de treino.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from scr.core import config, predictor
from scr.core.predictor import PredictionError
from scr.ui import components, state
from scr.ui.compat import STRETCH


def _render_missing_model() -> None:
    """Orienta o usuário quando ainda não existe modelo treinado."""
    st.info(
        "Nenhum modelo foi treinado nesta sessão. Abra a página 'Dados e treinamento' "
        "no menu lateral, carregue um arquivo e conclua o treinamento para liberar esta etapa."
    )


def _render_step_4() -> None:
    """Desenha o formulário dinâmico de entrada do caso a avaliar."""
    result = state.current_training_result()
    if result is None:
        _render_missing_model()
        return

    components.render_step_header(
        state.PREDICT_STEP_TITLES[4],
        "Preencha os campos abaixo com as informações do caso. Eles foram gerados a partir "
        "das colunas usadas no treinamento.",
    )

    st.markdown(
        f"Modelo em uso: **{result.model_display_name}**, treinado para prever "
        f"**{result.target}** ({config.TASK_LABELS[result.task_type].lower()})."
    )

    with st.form("formulario_predicao", border=True):
        values: Dict[str, Any] = components.render_prediction_inputs(
            result.column_specs,
            st.session_state.get("prediction_values"),
        )
        enviado = st.form_submit_button("Calcular previsão", type="primary", **STRETCH)

    if enviado:
        _run_prediction(values)

    st.caption(
        "Os limites e as categorias oferecidas em cada campo vêm dos dados de treino: "
        "o modelo só aprendeu dentro desse universo."
    )


def _run_prediction(values: Dict[str, Any]) -> None:
    """Executa a predição e guarda o resultado em sessão.

    Args:
        values: Valores informados no formulário.
    """
    result = state.current_training_result()
    if result is None:  # pragma: no cover - protegido pelo fluxo da página
        _render_missing_model()
        return

    try:
        with st.spinner("Calculando a previsão..."):
            prediction = predictor.predict_one(result, values)
    except PredictionError as error:
        st.error(str(error))
        return
    except Exception as error:  # rede de segurança: a UI nunca deve quebrar
        st.error(f"Ocorreu um erro inesperado ao calcular a previsão: {error}")
        return

    st.session_state["prediction_values"] = dict(values)
    st.session_state["prediction_result"] = prediction
    st.session_state["prediction_alerts"] = predictor.detect_out_of_range(
        values, result.column_specs
    )
    state.go_to_predict_step(5)
    st.rerun()


def _render_step_5() -> None:
    """Desenha o resultado da avaliação com a medida de confiança."""
    training = state.current_training_result()
    prediction = state.current_prediction()

    if training is None:
        _render_missing_model()
        return

    if prediction is None:
        st.info("Nenhuma previsão calculada ainda.")
        st.button(
            "Voltar",
            on_click=state.go_to_predict_step,
            args=(4,),
            key="voltar_sem_predicao",
        )
        return

    components.render_step_header(
        state.PREDICT_STEP_TITLES[5],
        "Leia a previsão junto com a confiança: as duas informações só fazem sentido juntas.",
    )

    components.render_input_alerts(st.session_state.get("prediction_alerts", []))
    components.render_prediction_result(prediction, training)

    with st.expander("Dados informados neste caso", expanded=False):
        for name, value in st.session_state.get("prediction_values", {}).items():
            st.write(f"- **{name}**: {value}")

    with st.expander("Como o desempenho geral do modelo foi medido", expanded=False):
        st.write(
            f"O modelo aprendeu com {training.n_train} registro(s) e foi avaliado em "
            f"{training.n_test} registro(s) separados antes do treino."
        )
        st.dataframe(
            components.metrics_to_frame(training.metrics),
            **STRETCH,
            hide_index=True,
        )

    st.divider()
    esquerda, _, direita = st.columns([1, 2, 1])
    with esquerda:
        st.button(
            "Voltar",
            **STRETCH,
            key="botao_voltar_5",
            on_click=state.go_to_predict_step,
            args=(4,),
        )
    with direita:
        st.button(
            "Avaliar outro caso",
            type="primary",
            **STRETCH,
            key="botao_novo_caso",
            on_click=state.reset_prediction,
        )


def render() -> None:
    """Renderiza a página de avaliação de novos casos na etapa corrente."""
    state.init_state()

    components.render_page_title(
        "Avaliar novos casos",
        "Use o modelo treinado para estimar o resultado de uma situação nova e verificar "
        "o quanto ele confia nessa estimativa.",
        logo="sklearn",
    )

    step = int(st.session_state["predict_step"])
    components.render_progress(step - state.FIRST_PREDICT_STEP + 1, 2)

    if step == state.FIRST_PREDICT_STEP:
        _render_step_4()
    else:
        _render_step_5()
