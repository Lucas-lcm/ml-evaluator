"""Pré-processamento e treino supervisionado.

Conceito de ML envolvido: o modelo é só a última peça do pipeline. Antes dele
existem decisões que determinam o resultado — imputação de ausentes,
padronização de escala, codificação de categorias — e todas precisam ser
aprendidas **apenas no conjunto de treino**. Encapsular tudo em um
``sklearn.pipeline.Pipeline`` é o que impede o vazamento de informação do teste
para o treino (*data leakage*), o erro mais comum e mais difícil de perceber em
projetos iniciantes.
"""

from __future__ import annotations

import time
import warnings as _warnings
from typing import Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scr.core import config, metrics as metrics_module, model_registry
from scr.core.data_loader import build_column_specs, is_numeric
from scr.core.schema import TaskType, TrainingResult


class TrainingError(Exception):
    """Erro de treino com mensagem em pt-BR pronta para exibição."""


# --------------------------------------------------------------------------- #
# Inferência de tarefa
# --------------------------------------------------------------------------- #


def infer_task_type(series: pd.Series) -> TaskType:
    """Sugere se o alvo caracteriza classificação ou regressão.

    A sugestão é uma heurística, não uma imposição: a interface sempre permite
    que o aprendiz mude a escolha. Discutir *por que* a máquina sugeriu um tipo
    faz parte do exercício.

    Regras aplicadas, nesta ordem:
        1. Coluna não numérica ou booleana → classificação.
        2. Coluna de inteiros com poucos valores distintos → classificação.
        3. Caso contrário → regressão.

    Args:
        series: Coluna escolhida como alvo.

    Returns:
        ``"classification"`` ou ``"regression"``.
    """
    clean = series.dropna()
    if clean.empty:
        return "classification"

    if not is_numeric(clean):
        return "classification"

    distinct = int(clean.nunique())
    if distinct <= 2:
        return "classification"

    is_integer_like = bool(pd.api.types.is_integer_dtype(clean)) or bool(
        np.allclose(clean.to_numpy(dtype=float) % 1, 0)
    )
    ratio = distinct / max(len(clean), 1)
    if (
        is_integer_like
        and distinct <= config.MAX_CLASSES_FOR_INFERENCE
        and ratio <= config.CLASSIFICATION_UNIQUE_RATIO
    ):
        return "classification"

    return "regression"


LEAKAGE_MUTUAL_INFO_THRESHOLD: Final[float] = 0.95
"""Acima disso, uma coluna praticamente determina o alvo."""

LEAKAGE_CORRELATION_THRESHOLD: Final[float] = 0.999
"""Correlação quase perfeita com o alvo numérico indica cópia disfarçada."""


def _association_with_target(feature: pd.Series, target: pd.Series, task_type: TaskType) -> float:
    """Mede o quanto uma coluna sozinha já determina o alvo.

    Em classificação usa informação mútua normalizada, discretizando colunas
    numéricas em quantis — a medida é sensível a qualquer relação, não só a
    linear. Em regressão usa o valor absoluto da correlação de Pearson.

    Args:
        feature: Coluna candidata a entrada.
        target: Coluna alvo.
        task_type: Tipo de tarefa.

    Returns:
        Valor entre 0 e 1; quanto mais perto de 1, mais a coluna sozinha explica
        o alvo. Devolve 0 quando a medida não é aplicável.
    """
    paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(paired) < config.MIN_ROWS_FOR_TRAINING:
        return 0.0

    values = paired["feature"]

    if task_type == "classification":
        if is_numeric(values):
            try:
                # 20 faixas é granularidade suficiente para flagrar cópia sem
                # transformar ruído numérico em falso positivo.
                values = pd.qcut(values, q=20, duplicates="drop").astype(str)
            except (ValueError, TypeError):
                values = values.astype(str)
        else:
            values = values.astype(str)
        try:
            return float(
                normalized_mutual_info_score(paired["target"].astype(str), values)
            )
        except ValueError:
            return 0.0

    if not is_numeric(values):
        return 0.0
    try:
        correlation = float(np.corrcoef(values.astype(float), paired["target"].astype(float))[0, 1])
    except (ValueError, TypeError, FloatingPointError):
        return 0.0
    return 0.0 if np.isnan(correlation) else abs(correlation)


def detect_leakage(
    frame: pd.DataFrame,
    target: str,
    candidates: Sequence[str],
    task_type: TaskType,
) -> Dict[str, float]:
    """Identifica colunas que praticamente já contêm a resposta.

    Conceito de ML envolvido: *vazamento de alvo* (target leakage). Uma coluna
    derivada da resposta — o parecer final, a data de aprovação, um identificador
    atribuído depois da decisão — faz a métrica saltar para perto de 100% no
    teste e desabar na vida real. É o defeito mais perigoso desta ferramenta,
    porque se disfarça de sucesso.

    A detecção é uma heurística deliberadamente conservadora: só acusa relações
    quase perfeitas, que uma variável legítima raramente apresenta.

    Args:
        frame: Tabela carregada.
        target: Coluna alvo.
        candidates: Colunas candidatas a entrada.
        task_type: Tipo de tarefa.

    Returns:
        Mapa ``{coluna: força_da_associação}`` apenas para as colunas suspeitas.
    """
    limit = (
        LEAKAGE_MUTUAL_INFO_THRESHOLD
        if task_type == "classification"
        else LEAKAGE_CORRELATION_THRESHOLD
    )
    usable = frame.dropna(subset=[target])
    suspects: Dict[str, float] = {}

    for column in candidates:
        if column == target or column not in usable.columns:
            continue
        score = _association_with_target(usable[column], usable[target], task_type)
        if score >= limit:
            suspects[column] = score

    return suspects


def suggest_features(
    frame: pd.DataFrame,
    target: str,
    task_type: Optional[TaskType] = None,
) -> Tuple[List[str], List[str]]:
    """Separa colunas utilizáveis como entrada das que devem ficar de fora.

    Colunas descartadas automaticamente:
        * a própria coluna alvo;
        * colunas constantes (não carregam informação);
        * colunas totalmente vazias;
        * colunas categóricas com cardinalidade altíssima, que normalmente são
          identificadores (CPF, código do pedido) e só ensinariam o modelo a
          decorar linhas;
        * colunas que praticamente reproduzem o alvo (vazamento), quando o tipo
          de tarefa é informado.

    O descarte nunca é definitivo: a interface mantém todas as colunas
    selecionáveis e apenas comunica o motivo da exclusão inicial.

    Args:
        frame: Tabela carregada.
        target: Nome da coluna alvo.
        task_type: Tipo de tarefa, necessário para a checagem de vazamento.

    Returns:
        Tupla ``(features_sugeridas, motivos_de_descarte)``.
    """
    suggested: List[str] = []
    reasons: List[str] = []

    for column in frame.columns:
        if column == target:
            continue

        series = frame[column]
        if series.isna().all():
            reasons.append(f"'{column}' foi ignorada: está completamente vazia.")
            continue
        if series.nunique(dropna=True) <= 1:
            reasons.append(f"'{column}' foi ignorada: tem sempre o mesmo valor.")
            continue
        if not is_numeric(series) and series.nunique(dropna=True) > config.MAX_CATEGORY_CARDINALITY:
            reasons.append(
                f"'{column}' foi ignorada: tem {series.nunique(dropna=True)} categorias distintas "
                "e provavelmente é um identificador."
            )
            continue

        suggested.append(column)

    if task_type is not None and suggested:
        leaking = detect_leakage(frame, target, suggested, task_type)
        for column, score in leaking.items():
            suggested.remove(column)
            reasons.append(
                f"'{column}' foi desmarcada por suspeita de vazamento: sozinha ela já "
                f"determina {score * 100:.0f}% do alvo '{target}'. Colunas assim inflam a "
                "métrica para perto de 100% e o modelo falha na vida real. Marque-a "
                "manualmente se tiver certeza de que estará disponível antes da decisão."
            )

    return suggested, reasons


# --------------------------------------------------------------------------- #
# Pré-processamento
# --------------------------------------------------------------------------- #


def build_preprocessor(frame: pd.DataFrame, features: Sequence[str]) -> ColumnTransformer:
    """Monta o pré-processador adequado aos tipos das colunas de entrada.

    Colunas numéricas recebem imputação pela mediana (robusta a valores
    atípicos) seguida de padronização — indispensável para modelos sensíveis a
    escala, como KNN, SVM e regressões regularizadas. Colunas categóricas
    recebem imputação pela moda e codificação one-hot com
    ``handle_unknown="ignore"``, para que uma categoria inédita na predição não
    derrube a aplicação.

    Args:
        frame: Tabela de referência para inferir tipos.
        features: Colunas de entrada.

    Returns:
        O :class:`ColumnTransformer` ainda não ajustado.

    Raises:
        TrainingError: Se nenhuma coluna de entrada for informada.
    """
    numeric = [column for column in features if is_numeric(frame[column])]
    categorical = [column for column in features if column not in numeric]

    if not numeric and not categorical:
        raise TrainingError("Selecione ao menos uma coluna de entrada para treinar o modelo.")

    transformers = []
    if numeric:
        transformers.append(
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("imputacao", SimpleImputer(strategy="median")),
                        ("padronizacao", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputacao", SimpleImputer(strategy="most_frequent")),
                        ("codificacao", _build_one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_one_hot_encoder() -> OneHotEncoder:
    """Cria o codificador one-hot compatível com a versão instalada do sklearn.

    O parâmetro mudou de ``sparse`` para ``sparse_output`` na versão 1.2. Tratar
    isso aqui evita que a aplicação quebre em ambientes de aluno com versões
    diferentes.

    Returns:
        O codificador configurado para ignorar categorias desconhecidas.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# --------------------------------------------------------------------------- #
# Validação prévia
# --------------------------------------------------------------------------- #


def validate_training_request(
    frame: pd.DataFrame,
    target: str,
    features: Sequence[str],
    task_type: TaskType,
) -> None:
    """Valida a configuração antes de gastar tempo treinando.

    Args:
        frame: Tabela carregada.
        target: Coluna alvo.
        features: Colunas de entrada.
        task_type: Tipo de tarefa escolhido.

    Raises:
        TrainingError: Se a configuração for inviável, sempre com uma mensagem
            que explica ao aprendiz o que precisa mudar.
    """
    if target not in frame.columns:
        raise TrainingError(f"A coluna alvo '{target}' não existe na tabela.")

    if not features:
        raise TrainingError("Selecione ao menos uma coluna de entrada.")

    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise TrainingError(f"Colunas de entrada inexistentes: {', '.join(missing)}.")

    if target in features:
        raise TrainingError(
            f"A coluna '{target}' foi escolhida como alvo e não pode ser também uma entrada. "
            "Isso faria o modelo simplesmente copiar a resposta."
        )

    usable = frame.dropna(subset=[target])
    if len(usable) < config.MIN_ROWS_FOR_TRAINING:
        raise TrainingError(
            f"São necessárias ao menos {config.MIN_ROWS_FOR_TRAINING} linhas com o alvo preenchido; "
            f"a tabela tem {len(usable)}."
        )

    if task_type == "classification":
        classes = usable[target].astype(str).nunique()
        if classes < 2:
            raise TrainingError(
                "Para classificação o alvo precisa ter pelo menos duas categorias diferentes."
            )
        if classes > 100:
            raise TrainingError(
                f"O alvo tem {classes} categorias distintas. Esse volume caracteriza um "
                "identificador, não uma classificação."
            )
    else:
        if not is_numeric(usable[target]):
            raise TrainingError(
                f"Para regressão o alvo precisa ser numérico, mas '{target}' contém texto. "
                "Escolha classificação ou selecione outra coluna."
            )
        if usable[target].nunique() < 2:
            raise TrainingError("O alvo tem sempre o mesmo valor; não há o que prever.")


# --------------------------------------------------------------------------- #
# Treino
# --------------------------------------------------------------------------- #


def _split(
    features_frame: pd.DataFrame,
    target_series: pd.Series,
    task_type: TaskType,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Divide os dados em treino e teste, estratificando quando for seguro.

    A estratificação preserva a proporção das classes nos dois conjuntos, mas
    exige ao menos duas amostras por classe. Quando isso não vale, o split
    prossegue sem estratificar e o fato é registrado como aviso — silenciar
    seria esconder do aprendiz uma fonte real de instabilidade da métrica.

    Args:
        features_frame: Colunas de entrada.
        target_series: Coluna alvo.
        task_type: Tipo de tarefa.
        test_size: Proporção reservada para teste.

    Returns:
        Tupla ``(X_treino, X_teste, y_treino, y_teste, avisos)``.
    """
    notes: List[str] = []
    stratify: Optional[pd.Series] = None

    if task_type == "classification":
        counts = target_series.value_counts()
        n_classes = len(counts)
        enough_per_class = bool((counts >= 2).all())
        enough_for_test = len(target_series) * test_size >= n_classes
        if enough_per_class and enough_for_test:
            stratify = target_series
        else:
            notes.append(
                "A divisão treino/teste não pôde ser estratificada porque alguma categoria "
                "aparece pouquíssimas vezes. As métricas ficam mais instáveis nesse cenário."
            )

    x_train, x_test, y_train, y_test = train_test_split(
        features_frame,
        target_series,
        test_size=test_size,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )
    return x_train, x_test, y_train, y_test, notes


def train_model(
    frame: pd.DataFrame,
    target: str,
    features: Sequence[str],
    task_type: TaskType,
    model_key: str,
    *,
    test_size: float = config.DEFAULT_TEST_SIZE,
) -> TrainingResult:
    """Treina um modelo supervisionado e avalia em dados nunca vistos.

    Fluxo executado:
        1. Valida a configuração.
        2. Remove linhas sem alvo (não se aprende com resposta ausente).
        3. Separa treino e teste.
        4. Ajusta o pipeline (pré-processamento + modelo) só no treino.
        5. Calcula o painel de métricas no teste.

    Args:
        frame: Tabela carregada.
        target: Nome da coluna alvo.
        features: Colunas de entrada.
        task_type: ``"classification"`` ou ``"regression"``.
        model_key: Chave do modelo no catálogo.
        test_size: Proporção dos dados reservada para avaliação.

    Returns:
        O :class:`TrainingResult` com pipeline ajustado, métricas e tudo que a
        etapa de predição precisa.

    Raises:
        TrainingError: Para qualquer configuração inviável ou falha de ajuste,
            com mensagem em pt-BR.
    """
    features = list(features)
    validate_training_request(frame, target, features, task_type)

    lower, upper = config.TEST_SIZE_BOUNDS
    test_size = float(min(max(test_size, lower), upper))

    spec = model_registry.get_model(task_type, model_key)

    working = frame.loc[:, features + [target]].copy()
    before = len(working)
    working = working.dropna(subset=[target])
    notes: List[str] = []
    if len(working) < before:
        notes.append(
            f"{before - len(working)} linha(s) foram descartadas por não terem valor na coluna alvo."
        )

    target_series = (
        working[target].astype(str) if task_type == "classification" else working[target].astype(float)
    )
    features_frame = working.loc[:, features]

    x_train, x_test, y_train, y_test, split_notes = _split(
        features_frame, target_series, task_type, test_size
    )
    notes.extend(split_notes)

    pipeline = Pipeline(
        steps=[
            ("preprocessamento", build_preprocessor(features_frame, features)),
            ("modelo", spec.build()),
        ]
    )

    started = time.perf_counter()
    with _warnings.catch_warnings(record=True) as captured:
        _warnings.simplefilter("always")
        try:
            pipeline.fit(x_train, y_train)
        except Exception as exc:
            raise TrainingError(
                f"O treino do modelo '{spec.display_name}' falhou: {exc}. "
                "Tente outro modelo ou revise as colunas selecionadas."
            ) from exc

        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            notes.append(
                "O modelo não convergiu totalmente dentro do número de iterações previsto. "
                "O resultado ainda é utilizável, mas pode melhorar com mais dados ou outro modelo."
            )
    elapsed = time.perf_counter() - started

    y_pred = pipeline.predict(x_test)

    if task_type == "classification":
        class_labels = [str(label) for label in pipeline.named_steps["modelo"].classes_]
        probabilities = None
        if hasattr(pipeline.named_steps["modelo"], "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(x_test)
            except (AttributeError, ValueError):
                probabilities = None

        computed = metrics_module.classification_metrics(
            y_test,
            y_pred,
            y_proba=probabilities,
            class_labels=class_labels,
            n_train=len(x_train),
            n_test=len(x_test),
        )
        per_class = metrics_module.per_class_metrics(
            y_test, y_pred, class_labels=class_labels
        )
        residual_std = None
        target_range = None
    else:
        class_labels = None
        per_class = []
        residuals = np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float)
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.abs(residuals).mean())
        target_range = float(np.ptp(np.asarray(target_series, dtype=float)))
        computed = metrics_module.regression_metrics(
            y_test,
            y_pred,
            n_features=len(features),
            n_train=len(x_train),
            n_test=len(x_test),
        )

    return TrainingResult(
        pipeline=pipeline,
        task_type=task_type,
        model_key=model_key,
        model_display_name=spec.display_name,
        target=target,
        features=features,
        column_specs=build_column_specs(features_frame, features),
        metrics=computed,
        per_class=per_class,
        class_labels=class_labels,
        residual_std=residual_std,
        target_range=target_range,
        n_train=len(x_train),
        n_test=len(x_test),
        training_seconds=elapsed,
        warnings=notes,
    )
