"""Ponto de entrada do ML Evaluator.

Execute com::

    streamlit run project/app.py

A navegação usa ``st.navigation``, que lista apenas os nomes das páginas na
barra lateral (requisito da spec 2): trocar de tela é clicar no nome, sem
botões intermediários.

A tela inicial é **"Como funciona"**. Quem abre a ferramenta pela primeira vez
não sabe o que é alvo, treino ou métrica; jogá-lo direto no formulário de upload
é pedir que ele tome decisões antes de ter o vocabulário para tomá-las.

Responsabilidade deste arquivo: configuração global da página e roteamento.
Nenhuma regra de negócio mora aqui.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

# Garante que o pacote `scr` seja importável independentemente do diretório de
# onde o Streamlit foi acionado.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402  (import após ajuste do sys.path, de propósito)

from scr.ui import state  # noqa: E402
from scr.ui.pages import explicacao, predicao, treinamento  # noqa: E402


FAVICON_PATH = os.path.join(_PROJECT_ROOT, "assets", "favicon.png")
"""Ícone da aba: um diagrama de árvore de decisão.

O símbolo é o modelo mais reconhecível do aprendizado supervisionado e o que
melhor sobrevive à redução para 16x16 pixels — um gráfico de dispersão ou uma
rede neural viram borrão nesse tamanho.
"""


def _page_icon() -> Optional[Any]:
    """Devolve o ícone da aba, ou ``None`` se o arquivo não estiver presente.

    Um repositório clonado sem a pasta ``assets`` continua funcionando: a aba
    apenas usa o ícone padrão do Streamlit.

    Returns:
        Caminho do PNG do favicon, ou ``None``.
    """
    return FAVICON_PATH if os.path.exists(FAVICON_PATH) else None


def main() -> None:
    """Configura a página e despacha para a rota selecionada."""
    st.set_page_config(
        page_title="ML Evaluator — Aprendizado de Máquina",
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    state.init_state()

    # A ordem desta lista é a ordem da barra lateral, e a primeira com
    # `default=True` é a tela de abertura.
    pagina_explicacao = st.Page(
        explicacao.render, title="Como funciona", url_path="como-funciona", default=True
    )
    pagina_treinamento = st.Page(
        treinamento.render, title="Dados e treinamento", url_path="treinamento"
    )
    pagina_avaliacao = st.Page(predicao.render, title="Avaliar novos casos", url_path="avaliacao")

    # As referências ficam no estado para permitir navegação por código (o botão
    # "Avaliar novos casos" na etapa 3, o "Começar" na tela inicial). Objetos de
    # página só existem durante a execução, então guardá-los é o caminho suportado.
    st.session_state["page_avaliacao"] = pagina_avaliacao
    st.session_state["page_treinamento"] = pagina_treinamento

    _render_sidebar_context()

    navegacao = st.navigation([pagina_explicacao, pagina_treinamento, pagina_avaliacao])
    navegacao.run()


def _render_sidebar_context() -> None:
    """Mostra na barra lateral o estado atual da sessão, sem oferecer navegação.

    A navegação é feita apenas pelos nomes das páginas. Este bloco é informativo:
    lembra ao aprendiz qual arquivo está carregado e qual modelo está treinado,
    contexto que ele perde ao trocar de tela.
    """
    with st.sidebar:
        st.markdown("### Situação atual")

        dataset = state.current_dataset()
        if dataset is None:
            st.caption("Nenhum arquivo carregado.")
        else:
            st.caption(f"Arquivo: {dataset.source_name}")
            st.caption(f"{dataset.n_rows} linha(s) e {dataset.n_cols} coluna(s).")

        resultado = state.current_training_result()
        if resultado is None:
            st.caption("Nenhum modelo treinado.")
        else:
            st.caption(f"Modelo: {resultado.model_display_name}")
            st.caption(f"Alvo: {resultado.target}")

        st.divider()
        st.caption(
            "Ferramenta didática de aprendizado supervisionado. "
            "Os dados enviados permanecem apenas na memória desta sessão."
        )


if __name__ == "__main__":
    main()
