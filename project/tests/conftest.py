"""Configuração compartilhada dos testes.

Coloca a raiz do projeto no ``sys.path`` para que ``import scr...`` funcione ao
rodar ``pytest`` de qualquer diretório.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture(scope="session")
def classification_frame() -> pd.DataFrame:
    """Base sintética separável para testes de classificação.

    Returns:
        Tabela com duas colunas numéricas, uma categórica e um alvo binário.
    """
    rng = np.random.default_rng(7)
    n = 240
    idade = rng.integers(18, 70, size=n)
    renda = rng.normal(5000, 1500, size=n)
    regiao = rng.choice(["norte", "sul", "sudeste"], size=n)
    score = 0.05 * idade + 0.0006 * renda + rng.normal(0, 0.4, size=n)
    aprovado = np.where(score > np.median(score), "sim", "nao")
    return pd.DataFrame(
        {"idade": idade, "renda": renda, "regiao": regiao, "aprovado": aprovado}
    )


@pytest.fixture(scope="session")
def regression_frame() -> pd.DataFrame:
    """Base sintética linear para testes de regressão.

    Returns:
        Tabela com duas colunas numéricas, uma categórica e um alvo contínuo.
    """
    rng = np.random.default_rng(11)
    n = 240
    area = rng.uniform(40, 300, size=n)
    quartos = rng.integers(1, 6, size=n)
    bairro = rng.choice(["centro", "praia", "serra"], size=n)
    preco = 1200 * area + 15000 * quartos + rng.normal(0, 8000, size=n)
    return pd.DataFrame({"area": area, "quartos": quartos, "bairro": bairro, "preco": preco})


@pytest.fixture(scope="session")
def csv_bytes(classification_frame: pd.DataFrame) -> Tuple[bytes, bytes]:
    """Mesma tabela serializada com vírgula e com ponto e vírgula.

    Args:
        classification_frame: Base de classificação.

    Returns:
        Tupla ``(csv_com_virgula, csv_com_ponto_e_virgula)`` em bytes.
    """
    head = classification_frame.head(30)
    return (
        head.to_csv(index=False).encode("utf-8"),
        head.to_csv(index=False, sep=";").encode("utf-8"),
    )
