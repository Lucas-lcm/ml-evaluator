"""Mapa de estado da sessão e operações de navegação entre etapas.

Streamlit reexecuta o script inteiro a cada interação. Sem um contrato explícito
de estado, isso vira uma fonte inesgotável de bugs: chaves criadas em uma página
e lidas em outra, valores obsoletos após trocar o arquivo, etapas que voltam
sozinhas. Este módulo concentra a inicialização, as transições e as invalidações
em um único lugar.

Mapa de estado
--------------
=========================  =========================================================
Chave                      Significado
=========================  =========================================================
``dataset``                :class:`Dataset` carregado, ou ``None``.
``dataset_signature``      Impressão digital do arquivo, usada como chave de cache.
``train_step``             Etapa atual da página de treinamento (1 a 3).
``predict_step``           Etapa atual da página de predição (4 ou 5).
``target``                 Coluna alvo escolhida.
``task_type``              ``"classification"`` ou ``"regression"``.
``model_key``              Modelo escolhido no catálogo.
``features``               Colunas de entrada selecionadas.
``test_size``              Proporção reservada para teste.
``training_result``        :class:`TrainingResult` do último treino concluído.
``prediction_values``      Últimos valores digitados no formulário de predição.
``prediction_result``      :class:`PredictionResult` da última predição.
``prediction_alerts``      Avisos de extrapolação da última predição.
``page_avaliacao``         Referência à página de avaliação, para navegação por código.
``page_treinamento``       Referência à página de treinamento, para navegação por código.
=========================  =========================================================
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Final, Optional

import streamlit as st

from scr.core import config
from scr.core.schema import Dataset, PredictionResult, TrainingResult

FIRST_TRAIN_STEP: Final[int] = 1
LAST_TRAIN_STEP: Final[int] = 3
FIRST_PREDICT_STEP: Final[int] = 4
LAST_PREDICT_STEP: Final[int] = 5

TRAIN_STEP_TITLES: Final[Dict[int, str]] = {
    1: "Etapa 1 de 3 — Carregar o arquivo e conferir a tabela",
    2: "Etapa 2 de 3 — Definir o problema, o alvo, o modelo e as colunas",
    3: "Etapa 3 de 3 — Resultado do treinamento",
}

PREDICT_STEP_TITLES: Final[Dict[int, str]] = {
    4: "Etapa 4 de 5 — Informar os dados do caso a avaliar",
    5: "Etapa 5 de 5 — Resultado da avaliação",
}

_DEFAULTS: Final[Dict[str, Any]] = {
    "dataset": None,
    "dataset_signature": "",
    "train_step": FIRST_TRAIN_STEP,
    "predict_step": FIRST_PREDICT_STEP,
    "target": None,
    "task_type": None,
    "model_key": None,
    "features": [],
    "test_size": config.DEFAULT_TEST_SIZE,
    "training_result": None,
    "prediction_values": {},
    "prediction_result": None,
    "prediction_alerts": [],
}


def init_state() -> None:
    """Garante que todas as chaves esperadas existam antes de qualquer leitura.

    Deve ser a primeira chamada de cada página. Nunca sobrescreve um valor já
    presente, portanto é seguro executar a cada reexecução do script.
    """
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            # Cópia rasa evita que listas e dicionários padrão sejam
            # compartilhados entre sessões diferentes.
            st.session_state[key] = list(value) if isinstance(value, list) else (
                dict(value) if isinstance(value, dict) else value
            )


def compute_signature(payload: bytes, filename: str) -> str:
    """Calcula uma impressão digital estável do arquivo carregado.

    Serve como chave de cache e como detector de troca de arquivo. Usamos SHA-256
    por ser determinístico e livre de colisões práticas — não é uso criptográfico
    de segurança, apenas identidade de conteúdo.

    Args:
        payload: Conteúdo bruto do arquivo.
        filename: Nome do arquivo.

    Returns:
        Hash hexadecimal do conteúdo combinado com o nome.
    """
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8", errors="replace"))
    digest.update(b"\x00")
    digest.update(payload)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Invalidações
# --------------------------------------------------------------------------- #


def reset_training() -> None:
    """Descarta o modelo treinado e tudo que dependia dele.

    Chamado quando o aprendiz muda alvo, tipo de problema, modelo ou colunas.
    Manter na tela uma métrica calculada com outra configuração seria a pior
    falha pedagógica possível desta aplicação.
    """
    st.session_state["training_result"] = None
    reset_prediction()


def reset_prediction() -> None:
    """Limpa o formulário e o resultado da última predição."""
    st.session_state["prediction_values"] = {}
    st.session_state["prediction_result"] = None
    st.session_state["prediction_alerts"] = []
    st.session_state["predict_step"] = FIRST_PREDICT_STEP


def reset_dataset() -> None:
    """Descarta a tabela carregada e todo o estado derivado dela."""
    st.session_state["dataset"] = None
    st.session_state["dataset_signature"] = ""
    st.session_state["target"] = None
    st.session_state["task_type"] = None
    st.session_state["model_key"] = None
    st.session_state["features"] = []
    st.session_state["train_step"] = FIRST_TRAIN_STEP
    reset_training()


# --------------------------------------------------------------------------- #
# Navegação entre etapas
# --------------------------------------------------------------------------- #


def go_to_train_step(step: int) -> None:
    """Move a página de treinamento para uma etapa, respeitando os limites.

    Args:
        step: Etapa desejada (1 a 3).
    """
    st.session_state["train_step"] = int(min(max(step, FIRST_TRAIN_STEP), LAST_TRAIN_STEP))


def switch_to_prediction_page() -> None:
    """Leva o usuário direto para a tela de avaliação (spec 3).

    O objeto de página é registrado por ``app.py`` no estado da sessão: páginas
    do ``st.navigation`` só existem durante a execução, e guardar a referência é
    a forma suportada de trocar de tela por código. Se por algum motivo a
    referência não existir, a função apenas orienta o usuário em texto, em vez
    de quebrar.
    """
    page = st.session_state.get("page_avaliacao")
    if page is None:
        st.info("Abra a página 'Avaliar novos casos' no menu lateral.")
        return
    st.switch_page(page)


def switch_to_training_page() -> None:
    """Leva o usuário da tela inicial para o fluxo de treinamento.

    Mesmo mecanismo de :func:`switch_to_prediction_page`: a referência da página
    é registrada por ``app.py`` no estado da sessão.
    """
    page = st.session_state.get("page_treinamento")
    if page is None:
        st.info("Abra a página 'Dados e treinamento' no menu lateral.")
        return
    st.switch_page(page)


def go_to_predict_step(step: int) -> None:
    """Move a página de predição para uma etapa, respeitando os limites.

    Args:
        step: Etapa desejada (4 ou 5).
    """
    st.session_state["predict_step"] = int(
        min(max(step, FIRST_PREDICT_STEP), LAST_PREDICT_STEP)
    )


# --------------------------------------------------------------------------- #
# Acesso tipado
# --------------------------------------------------------------------------- #


def current_dataset() -> Optional[Dataset]:
    """Devolve a tabela carregada, se houver.

    Returns:
        O :class:`Dataset` em sessão ou ``None``.
    """
    return st.session_state.get("dataset")


def current_training_result() -> Optional[TrainingResult]:
    """Devolve o resultado do treino concluído, se houver.

    Returns:
        O :class:`TrainingResult` em sessão ou ``None``.
    """
    return st.session_state.get("training_result")


def current_prediction() -> Optional[PredictionResult]:
    """Devolve o resultado da última predição, se houver.

    Returns:
        O :class:`PredictionResult` em sessão ou ``None``.
    """
    return st.session_state.get("prediction_result")
