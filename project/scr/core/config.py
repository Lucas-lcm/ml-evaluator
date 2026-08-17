"""Constantes e limites operacionais da aplicação.

Centralizar limites aqui é uma medida de *defense-in-depth*: qualquer guarda
contra exaustão de memória ou arquivo malicioso tem um único ponto de verdade,
auditável e ajustável sem tocar na lógica.
"""

from __future__ import annotations

from typing import Final, FrozenSet, Mapping, Tuple

# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #

RANDOM_STATE: Final[int] = 42
"""Semente única usada em splits e modelos estocásticos.

Reprodutibilidade é requisito pedagógico: o aprendiz precisa obter a mesma
métrica ao repetir o experimento para conseguir atribuir a variação observada
à mudança que ele fez, e não ao acaso.
"""

# --------------------------------------------------------------------------- #
# Guardas de upload
# --------------------------------------------------------------------------- #

MAX_UPLOAD_MB: Final[int] = 25
MAX_UPLOAD_BYTES: Final[int] = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".csv", ".txt", ".xlsx", ".json"})
"""Extensões aceitas. Formatos executáveis (``.pkl``, ``.joblib``) são
deliberadamente excluídos: desserializar objetos arbitrários equivale a
execução remota de código."""

ALLOWED_MIME_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        "text/csv",
        "text/plain",
        "text/tab-separated-values",
        "application/csv",
        "application/json",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",  # navegadores frequentemente enviam este genérico
    }
)

MAX_ROWS: Final[int] = 200_000
"""Teto de linhas lidas. Protege a sessão contra exaustão de memória."""

MAX_COLUMNS: Final[int] = 512

SNIFF_SAMPLE_BYTES: Final[int] = 64 * 1024
"""Quantidade de bytes lidos para detectar separador e codificação."""

CANDIDATE_SEPARATORS: Final[Tuple[str, ...]] = (",", ";", "\t", "|", ":")

CANDIDATE_ENCODINGS: Final[Tuple[str, ...]] = ("utf-8-sig", "utf-8", "latin-1")

# --------------------------------------------------------------------------- #
# Regras de modelagem
# --------------------------------------------------------------------------- #

MIN_ROWS_FOR_TRAINING: Final[int] = 10
"""Abaixo disso a divisão treino/teste deixa de ter qualquer significado
estatístico e a métrica reportada seria ruído."""

MAX_CATEGORY_CARDINALITY: Final[int] = 50
"""Colunas categóricas com mais categorias que isso viram um one-hot enorme.
São descartadas das features com aviso explícito ao aprendiz."""

CLASSIFICATION_UNIQUE_RATIO: Final[float] = 0.05
"""Heurística de inferência de tarefa: uma coluna numérica com poucos valores
distintos em relação ao total de linhas provavelmente é um rótulo."""

MAX_CLASSES_FOR_INFERENCE: Final[int] = 20

DEFAULT_TEST_SIZE: Final[float] = 0.2
TEST_SIZE_BOUNDS: Final[Tuple[float, float]] = (0.1, 0.4)

PREVIEW_ROWS: Final[int] = 50

TASK_LABELS: Final[Mapping[str, str]] = {
    "classification": "Classificação",
    "regression": "Regressão",
}
