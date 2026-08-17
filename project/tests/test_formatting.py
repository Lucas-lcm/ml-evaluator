"""Testes da formatação numérica em pt-BR."""

from __future__ import annotations

import math

import pytest

from scr.core.formatting import INDISPONIVEL, format_integer, format_number, format_percentage


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [
        (210482.3712, "210.482,37"),
        (1234.5, "1.234,50"),
        (-1234.5, "-1.234,50"),
        (0.8231, "0,8231"),
        (0.0, "0,0000"),
    ],
)
def test_format_number_usa_padrao_brasileiro(valor: float, esperado: str) -> None:
    assert format_number(valor) == esperado


def test_format_number_nunca_usa_notacao_cientifica() -> None:
    # Notação científica é ilegível para o público não técnico da ferramenta.
    assert "e+" not in format_number(1_500_000.123)
    assert "e-" not in format_number(0.0000001)


@pytest.mark.parametrize("valor", [None, float("nan"), float("inf")])
def test_valores_indefinidos_viram_texto(valor) -> None:
    assert format_number(valor) == INDISPONIVEL
    assert format_percentage(valor) == INDISPONIVEL


def test_format_percentage() -> None:
    assert format_percentage(0.9231) == "92,31%"
    assert format_percentage(1.0) == "100,00%"
    assert format_percentage(0.9231, decimals=1) == "92,3%"


def test_format_integer_usa_ponto_de_milhar() -> None:
    assert format_integer(12500) == "12.500"
    assert format_integer(7) == "7"
    assert format_integer(None) == INDISPONIVEL


def test_format_number_com_entrada_nao_numerica_nao_quebra() -> None:
    assert format_number("abc") == INDISPONIVEL  # type: ignore[arg-type]
    assert not math.isnan(0)  # sanidade do próprio teste
