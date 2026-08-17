"""Catálogo de modelos supervisionados disponíveis na aplicação.

Conceito de ML envolvido: não existe um "melhor modelo" universal (*no free
lunch*). Cada família carrega um viés indutivo diferente — linear, baseado em
partições, baseado em vizinhança, baseado em margem. Expor essas famílias lado a
lado, com uma explicação curta de cada uma, é o que transforma a escolha do
modelo em um exercício de raciocínio em vez de um sorteio.

O catálogo é declarativo: cada entrada guarda uma *fábrica* (callable que
constrói uma instância nova), nunca uma instância compartilhada. Reutilizar um
estimador já ajustado entre sessões causaria vazamento de estado entre usuários.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Tuple

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from scr.core.config import RANDOM_STATE
from scr.core.schema import TaskType


@dataclass(frozen=True)
class ModelSpec:
    """Entrada do catálogo de modelos.

    A explicação é dividida em três campos curtos e objetivos em vez de um
    parágrafo corrido (spec 3). Cada um responde a uma pergunta que o aprendiz
    faz de fato — *o que é*, *como chega no resultado*, *quando escolher* — e a
    interface pode renderizá-los como rótulos distintos.

    Attributes:
        key: Identificador estável, usado em ``st.session_state`` e nos testes.
        display_name: Nome completo do modelo, exibido no seletor.
        summary: Uma frase dizendo o que o modelo faz.
        how_it_works: Uma frase dizendo como ele chega ao resultado.
        when_to_use: Uma frase dizendo em que situação escolhê-lo.
        factory: Função sem argumentos que devolve um estimador novo.
        supports_probability: Se o modelo fornece ``predict_proba``.
        is_ensemble: Se o modelo é um comitê de estimadores (permite estimar a
            dispersão da predição em regressão).
    """

    key: str
    display_name: str
    summary: str
    how_it_works: str
    when_to_use: str
    factory: Callable[[], BaseEstimator]
    supports_probability: bool = False
    is_ensemble: bool = False

    @property
    def description(self) -> str:
        """Explicação completa em texto corrido.

        Returns:
            As três frases concatenadas, para uso em contextos que precisam de
            um único bloco de texto (tooltips, notebook, testes).
        """
        return f"{self.summary} {self.how_it_works} {self.when_to_use}"

    def build(self) -> BaseEstimator:
        """Cria uma instância nova e não ajustada do estimador.

        Returns:
            Estimador scikit-learn pronto para ser inserido em um pipeline.
        """
        return self.factory()


# --------------------------------------------------------------------------- #
# Classificação
#
# O conjunto foi reduzido (spec 3) a quatro famílias com vieses indutivos
# claramente distintos: linear (Logística), partição (Árvore), comitê de
# partições (Floresta) e vizinhança (KNN). Mais opções que isso, para um
# público iniciante, viram ruído em vez de escolha informada.
# --------------------------------------------------------------------------- #

_CLASSIFICATION_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        key="logistic_regression",
        display_name="Regressão Logística",
        summary="Estima a probabilidade de o caso pertencer a cada categoria.",
        how_it_works=(
            "Usa uma curva em forma de S que vai de 0 a 1 para transformar as informações "
            "de entrada em probabilidade, e classifica pelo limiar de 50%."
        ),
        when_to_use=(
            "Classificação binária em que interpretar o resultado importa, como detectar "
            "spam ou prever a presença de uma condição."
        ),
        factory=lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        supports_probability=True,
    ),
    ModelSpec(
        key="random_forest_classifier",
        display_name="Floresta Aleatória (Random Forest)",
        summary="Combina muitas árvores de decisão em uma única resposta, por votação.",
        how_it_works=(
            "Treina centenas de árvores em amostras aleatórias dos dados e decide pela "
            "categoria mais votada entre elas."
        ),
        when_to_use=(
            "Quando se quer generalização melhor que a de uma árvore isolada e menor "
            "sensibilidade a variações nos dados. É o ponto de partida mais seguro."
        ),
        factory=lambda: RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        supports_probability=True,
        is_ensemble=True,
    ),
    ModelSpec(
        key="decision_tree_classifier",
        display_name="Árvore de Decisão",
        summary="Cria regras encadeadas do tipo 'se isto, então aquilo' até chegar à categoria.",
        how_it_works=(
            "Divide os dados sucessivamente pela informação que melhor separa as categorias, "
            "até que cada grupo tenha uma decisão final."
        ),
        when_to_use=(
            "Quando explicar o raciocínio é essencial. Atenção: é altamente propensa a "
            "decorar os dados de treino."
        ),
        factory=lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        supports_probability=True,
    ),
    ModelSpec(
        key="knn_classifier",
        display_name="K Vizinhos Mais Próximos (KNN)",
        summary="Classifica o caso novo pelo grupo dos registros históricos mais parecidos.",
        how_it_works=(
            "Mede a distância do caso novo até todos os registros conhecidos e adota a "
            "categoria majoritária entre os cinco mais próximos."
        ),
        when_to_use=(
            "Quando se quer uma classificação simples e há dados suficientes. Perde "
            "qualidade quando o número de colunas é grande."
        ),
        factory=lambda: KNeighborsClassifier(n_neighbors=5),
        supports_probability=True,
    ),
)

# --------------------------------------------------------------------------- #
# Regressão
# --------------------------------------------------------------------------- #

_REGRESSION_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        key="linear_regression",
        display_name="Regressão Linear",
        summary="Prevê um valor numérico contínuo a partir das colunas de entrada.",
        how_it_works=(
            "Encontra a reta que melhor se ajusta aos dados, minimizando a distância entre "
            "o valor real e o valor previsto."
        ),
        when_to_use=(
            "Quando o alvo é um número e a relação com as entradas é aproximadamente "
            "proporcional, como preço por metro quadrado."
        ),
        factory=lambda: LinearRegression(),
    ),
    ModelSpec(
        key="random_forest_regressor",
        display_name="Floresta Aleatória (Random Forest)",
        summary="Combina muitas árvores e prevê a média dos valores que elas indicam.",
        how_it_works=(
            "Treina centenas de árvores em amostras aleatórias dos dados e devolve a média "
            "das previsões; a discordância entre elas indica a incerteza."
        ),
        when_to_use=(
            "Quando a relação entre entradas e alvo não é uma linha reta e se quer boa "
            "generalização sem ajuste fino."
        ),
        factory=lambda: RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        is_ensemble=True,
    ),
    ModelSpec(
        key="decision_tree_regressor",
        display_name="Árvore de Decisão",
        summary="Agrupa casos parecidos por regras encadeadas e prevê a média do grupo.",
        how_it_works=(
            "Divide os dados sucessivamente pela informação que melhor separa os valores do "
            "alvo, e prevê a média do grupo em que o caso caiu."
        ),
        when_to_use=(
            "Quando explicar o raciocínio é essencial. Produz previsões em degraus e decora "
            "os dados com facilidade."
        ),
        factory=lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    ),
)

CATALOG: Mapping[TaskType, Tuple[ModelSpec, ...]] = {
    "classification": _CLASSIFICATION_MODELS,
    "regression": _REGRESSION_MODELS,
}


def list_models(task_type: TaskType) -> List[ModelSpec]:
    """Lista os modelos disponíveis para um tipo de tarefa.

    Args:
        task_type: ``"classification"`` ou ``"regression"``.

    Returns:
        Lista de :class:`ModelSpec` na ordem didática do catálogo (do mais
        simples e interpretável para o mais complexo).

    Raises:
        KeyError: Se o tipo de tarefa for desconhecido.
    """
    if task_type not in CATALOG:
        raise KeyError(f"Tipo de tarefa desconhecido: {task_type!r}.")
    return list(CATALOG[task_type])


def get_model(task_type: TaskType, key: str) -> ModelSpec:
    """Recupera uma especificação de modelo pelo identificador.

    Args:
        task_type: Tipo de tarefa a que o modelo pertence.
        key: Identificador do modelo no catálogo.

    Returns:
        A :class:`ModelSpec` correspondente.

    Raises:
        KeyError: Se não houver modelo com essa chave para a tarefa informada.
    """
    for spec in list_models(task_type):
        if spec.key == key:
            return spec
    raise KeyError(f"Modelo {key!r} não existe no catálogo de {task_type!r}.")


def default_model_key(task_type: TaskType) -> str:
    """Devolve o modelo sugerido como ponto de partida.

    A floresta aleatória é o padrão em ambas as tarefas por ser a opção que
    entrega um resultado razoável sem ajuste de hiperparâmetros, evitando que o
    aprendiz conclua cedo demais que "modelagem não funciona".

    Args:
        task_type: Tipo de tarefa.

    Returns:
        A chave do modelo padrão.
    """
    preferred = {
        "classification": "random_forest_classifier",
        "regression": "random_forest_regressor",
    }
    return preferred[task_type]


def model_options(task_type: TaskType) -> Dict[str, str]:
    """Mapa ``{chave: nome completo}`` para alimentar seletores da interface.

    Args:
        task_type: Tipo de tarefa.

    Returns:
        Dicionário ordenado conforme o catálogo.
    """
    return {spec.key: spec.display_name for spec in list_models(task_type)}
