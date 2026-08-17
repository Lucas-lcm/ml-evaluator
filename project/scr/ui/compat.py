"""Compatibilidade com as mudanças da API de layout do Streamlit.

Contexto do problema: a partir da versão 1.49 o Streamlit substituiu
``use_container_width=True`` por ``width="stretch"`` e passou a **validar** o
parâmetro ``width``. Nessa nova API, ``width=None`` deixou de ser aceito em
``st.image`` e levanta ``StreamlitInvalidWidthError``.

Uma ferramenta didática é instalada em máquinas de alunos com versões
diferentes. Em vez de fixar uma versão e quebrar em todas as outras, este
módulo resolve a diferença em um único lugar, na importação, e o resto da UI
passa a usar apenas ``STRETCH`` e ``IMAGE_AUTO_WIDTH``.
"""

from __future__ import annotations

from typing import Any, Dict, Final, Tuple

import streamlit as st


def _parse_version(raw: str) -> Tuple[int, ...]:
    """Converte a versão do Streamlit em uma tupla comparável.

    Args:
        raw: Texto da versão, como ``"1.49.1"`` ou ``"1.50.0.dev20260101"``.

    Returns:
        Tupla de inteiros com as partes numéricas iniciais. Devolve ``(0,)``
        quando a versão não puder ser interpretada — o que faz o código cair no
        caminho da API antiga, mais conservador.
    """
    partes: list[int] = []
    for pedaco in str(raw).split("."):
        digitos = ""
        for caractere in pedaco:
            if not caractere.isdigit():
                break
            digitos += caractere
        if not digitos:
            break
        partes.append(int(digitos))
    return tuple(partes) if partes else (0,)


STREAMLIT_VERSION: Final[Tuple[int, ...]] = _parse_version(getattr(st, "__version__", "0"))

USA_API_DE_LARGURA: Final[bool] = STREAMLIT_VERSION >= (1, 49)
"""Se a versão instalada usa ``width="stretch"`` no lugar de ``use_container_width``."""

STRETCH: Final[Dict[str, Any]] = (
    {"width": "stretch"} if USA_API_DE_LARGURA else {"use_container_width": True}
)
"""Argumentos para ocupar toda a largura disponível do contêiner.

Use como ``st.dataframe(frame, **STRETCH)``. Concentrar isso aqui evita que a
migração de API precise ser repetida em cada chamada da interface.
"""

IMAGE_AUTO_WIDTH: Final[Any] = "content" if USA_API_DE_LARGURA else None
"""Largura natural de uma imagem.

Na API nova o valor é ``"content"``; na antiga, ``None``. Passar ``None`` para
uma versão nova levanta ``StreamlitInvalidWidthError``, que foi exatamente a
falha observada em execução.
"""
