"""Camada de domínio: ingestão de dados, catálogo de modelos, treino e predição.

Nenhum módulo deste pacote importa Streamlit, o que o torna testável como
código Python comum.
"""

from scr.core import config, metrics, model_registry, predictor, schema, trainer
from scr.core.data_loader import DataLoadError, load_tabular_file

__all__ = [
    "config",
    "metrics",
    "model_registry",
    "predictor",
    "schema",
    "trainer",
    "DataLoadError",
    "load_tabular_file",
]
