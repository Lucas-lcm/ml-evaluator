"""Camada de apresentação (Streamlit).

Regra de fronteira: módulos deste pacote podem importar `scr.core`, mas nunca o
contrário. Toda regra de negócio vive em `scr.core` e permanece testável sem
Streamlit.
"""

__all__ = ["state", "components", "pages"]
