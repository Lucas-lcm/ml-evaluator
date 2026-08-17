"""Pacote raiz do ML Evaluator.

Camadas:
    - ``scr.core``: lógica pura (dados, modelos, treino, predição). Sem Streamlit.
    - ``scr.ui``: camada de apresentação (Streamlit). Sem regra de negócio.
"""

__all__ = ["core", "ui"]
__version__ = "1.0.0"
