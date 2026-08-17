"""Testes da configuração de publicação.

O Streamlit lê ``.streamlit/config.toml`` a partir do **diretório de trabalho**,
não do diretório do script. Como o app roda tanto de dentro de ``project/``
(desenvolvimento) quanto da raiz do repositório (Streamlit Cloud), existem duas
cópias do arquivo. Duas cópias sem verificação viram duas cópias divergentes na
primeira alteração — daí estes testes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent

CONFIG_PROJECT = PROJECT_DIR / ".streamlit" / "config.toml"
CONFIG_REPO = REPO_DIR / ".streamlit" / "config.toml"
REQUIREMENTS_PROJECT = PROJECT_DIR / "requirements.txt"
REQUIREMENTS_REPO = REPO_DIR / "requirements.txt"


def _load(path: Path) -> Dict[str, Any]:
    """Lê um config.toml.

    Args:
        path: Caminho do arquivo.

    Returns:
        Conteúdo já convertido em dicionário.
    """
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_as_duas_copias_do_config_existem() -> None:
    assert CONFIG_PROJECT.exists(), "config de desenvolvimento ausente"
    assert CONFIG_REPO.exists(), "config da raiz (usada pelo Streamlit Cloud) ausente"


def test_as_duas_copias_do_config_sao_identicas() -> None:
    # Divergência aqui significa que o app publicado se comporta diferente do
    # app local — o tipo de bug que só aparece em produção.
    assert CONFIG_PROJECT.read_text(encoding="utf-8") == CONFIG_REPO.read_text(encoding="utf-8")


def test_menu_de_desenvolvedor_esta_oculto() -> None:
    config = _load(CONFIG_REPO)
    assert config["client"]["toolbarMode"] in {"viewer", "minimal"}


def test_limite_de_upload_do_servidor_bate_com_o_do_dominio() -> None:
    from scr.core import config as core_config

    config = _load(CONFIG_REPO)
    assert config["server"]["maxUploadSize"] == core_config.MAX_UPLOAD_MB


def test_protecoes_de_servidor_ligadas() -> None:
    config = _load(CONFIG_REPO)
    assert config["server"]["enableXsrfProtection"] is True
    assert config["server"]["enableCORS"] is False


def test_tema_escuro_configurado() -> None:
    config = _load(CONFIG_REPO)
    assert config["theme"]["base"] == "dark"


def test_requirements_existem_nos_dois_lugares_e_sao_identicos() -> None:
    # O Streamlit Cloud procura o arquivo de dependências na raiz do repositório.
    assert REQUIREMENTS_REPO.exists(), "requirements.txt da raiz ausente"
    assert REQUIREMENTS_PROJECT.read_text(encoding="utf-8") == REQUIREMENTS_REPO.read_text(
        encoding="utf-8"
    )


def test_requirements_cobre_as_dependencias_de_execucao() -> None:
    texto = REQUIREMENTS_REPO.read_text(encoding="utf-8").lower()
    for pacote in ("streamlit", "pandas", "numpy", "scikit-learn", "openpyxl"):
        assert pacote in texto, f"'{pacote}' não declarado em requirements.txt"


def test_segredos_nao_estao_versionados() -> None:
    gitignore = REPO_DIR / ".gitignore"
    if not gitignore.exists():
        pytest.skip("Sem .gitignore neste checkout.")
    conteudo = gitignore.read_text(encoding="utf-8")
    assert "secrets.toml" in conteudo

    for caminho in (PROJECT_DIR / ".streamlit" / "secrets.toml", REPO_DIR / ".streamlit" / "secrets.toml"):
        assert not caminho.exists(), f"arquivo de segredos presente no repositório: {caminho}"


def test_favicon_existe_e_e_quadrado() -> None:
    favicon = PROJECT_DIR / "assets" / "favicon.png"
    assert favicon.exists(), "favicon ausente"

    pillow = pytest.importorskip("PIL.Image")
    with pillow.open(favicon) as imagem:
        largura, altura = imagem.size
    assert largura == altura, "o ícone da aba precisa ser quadrado"
    assert largura >= 128, "resolução baixa demais para telas de alta densidade"
