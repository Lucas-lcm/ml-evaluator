"""Formatação numérica no padrão brasileiro.

Um público não técnico lê ``210.482,37``; não lê ``2,105e+05``. Como métricas,
previsões e avisos precisam concordar entre si, a formatação vive em um único
módulo, sem dependência de Streamlit ou de locale do sistema operacional —
depender de locale tornaria a saída diferente conforme a máquina do aluno.
"""

from __future__ import annotations

import math
from typing import Optional

INDISPONIVEL = "não disponível"


def format_number(value: Optional[float], *, decimals: int = 4) -> str:
    """Formata um número com ponto de milhar e vírgula decimal.

    Args:
        value: Valor a formatar. ``None``, ``NaN`` e infinitos viram texto.
        decimals: Casas decimais para valores de magnitude pequena.

    Returns:
        Texto pronto para exibição.

    Examples:
        >>> format_number(210482.3712)
        '210.482,37'
        >>> format_number(0.8231)
        '0,8231'
    """
    if value is None:
        return INDISPONIVEL
    try:
        number = float(value)
    except (TypeError, ValueError):
        return INDISPONIVEL
    if math.isnan(number) or math.isinf(number):
        return INDISPONIVEL

    magnitude = abs(number)
    if magnitude >= 1000:
        places = 2
    elif magnitude != 0 and magnitude < 0.0001:
        places = 8
    else:
        places = decimals

    texto = f"{number:,.{places}f}"
    # Troca o padrão en-US (1,234.56) pelo pt-BR (1.234,56) sem usar locale.
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def format_percentage(value: Optional[float], *, decimals: int = 2) -> str:
    """Formata uma fração de 0 a 1 como percentual em pt-BR.

    Args:
        value: Fração a converter.
        decimals: Casas decimais do percentual.

    Returns:
        Texto no formato ``"92,50%"``, ou ``"não disponível"``.
    """
    if value is None:
        return INDISPONIVEL
    try:
        number = float(value)
    except (TypeError, ValueError):
        return INDISPONIVEL
    if math.isnan(number) or math.isinf(number):
        return INDISPONIVEL
    return f"{number * 100:.{decimals}f}".replace(".", ",") + "%"


def format_integer(value: Optional[float]) -> str:
    """Formata um inteiro com ponto de milhar.

    Args:
        value: Valor a formatar.

    Returns:
        Texto no formato ``"12.500"``, ou ``"não disponível"``.
    """
    if value is None:
        return INDISPONIVEL
    try:
        return f"{int(round(float(value))):,}".replace(",", ".")
    except (TypeError, ValueError, OverflowError):
        return INDISPONIVEL
