"""Testes da camada de compatibilidade com a API de layout do Streamlit.

Motivação: em execução, `st.image(..., width=None)` levantou
``StreamlitInvalidWidthError`` a partir do Streamlit 1.49, que passou a validar
o parâmetro ``width``. Estes testes congelam a regra de conversão para que a
correção não se perca em uma refatoração futura.

Os testes que precisam do Streamlit instalado são pulados quando ele não está
disponível — a suíte do domínio continua rodando em ambientes sem a
dependência de interface.
"""

from __future__ import annotations

import pytest


def test_parse_version_le_formatos_reais() -> None:
    streamlit = pytest.importorskip("streamlit")
    from scr.ui.compat import _parse_version

    assert _parse_version("1.49.1") == (1, 49, 1)
    assert _parse_version("1.50.0") == (1, 50, 0)
    assert _parse_version("1.48.0") == (1, 48, 0)
    # Versões de desenvolvimento trazem sufixo não numérico.
    assert _parse_version("1.51.0.dev20260101") == (1, 51, 0)
    assert _parse_version(streamlit.__version__)[0] >= 1


def test_parse_version_com_entrada_invalida_cai_no_caminho_conservador() -> None:
    pytest.importorskip("streamlit")
    from scr.ui.compat import _parse_version

    # Sem versão legível, assume-se a API antiga: falhar para o lado seguro.
    assert _parse_version("desconhecida") == (0,)
    assert _parse_version("") == (0,)
    assert _parse_version("desconhecida") < (1, 49)


def test_stretch_e_coerente_com_a_versao_detectada() -> None:
    pytest.importorskip("streamlit")
    from scr.ui import compat

    if compat.USA_API_DE_LARGURA:
        assert compat.STRETCH == {"width": "stretch"}
    else:
        assert compat.STRETCH == {"use_container_width": True}


def test_largura_de_imagem_nunca_e_none_na_api_nova() -> None:
    pytest.importorskip("streamlit")
    from scr.ui import compat

    # Esta é exatamente a condição que quebrou a aplicação.
    if compat.USA_API_DE_LARGURA:
        assert compat.IMAGE_AUTO_WIDTH == "content"
        assert compat.IMAGE_AUTO_WIDTH is not None
    else:
        assert compat.IMAGE_AUTO_WIDTH is None


def test_valor_de_largura_e_aceito_pela_validacao_do_streamlit() -> None:
    """Confere o valor contra a própria validação do Streamlit, quando existir."""
    pytest.importorskip("streamlit")
    from scr.ui import compat

    try:
        from streamlit.elements.lib.layout_utils import validate_width
    except ImportError:
        pytest.skip("Versão do Streamlit sem validate_width.")

    # Não deve levantar: é o mesmo caminho percorrido por st.image.
    validate_width(compat.IMAGE_AUTO_WIDTH, allow_content=True)
    validate_width(compat.STRETCH["width"], allow_content=True)


def test_ilustracoes_declaradas_existem_no_repositorio() -> None:
    pytest.importorskip("streamlit")
    from scr.ui.components import ASSETS_DIR, _LOGOS

    for nome, arquivo in _LOGOS.items():
        caminho = ASSETS_DIR / arquivo
        assert caminho.exists(), f"ilustração '{nome}' ausente em {caminho}"
        assert caminho.stat().st_size > 0
