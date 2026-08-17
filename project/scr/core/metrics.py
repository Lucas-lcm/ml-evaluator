"""Cálculo e explicação das métricas de avaliação.

Conceito de ML envolvido: uma única métrica nunca descreve um modelo por
inteiro. Acurácia esconde desequilíbrio de classes; R² esconde erros grandes em
poucos casos. Por isso este módulo devolve sempre um **conjunto** de métricas,
cada uma acompanhada de uma explicação de uma linha em pt-BR — a explicação faz
parte do contrato, não é enfeite de interface.

O módulo não importa Streamlit: ele produz dados (`MetricRow`), e a camada de
apresentação decide como desenhá-los.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from scr.core.formatting import format_integer, format_number, format_percentage
from scr.core.schema import ClassMetricRow, MetricRow

METRIC_GLOSSARY: Dict[str, str] = {
    # Classificação
    "accuracy": "Percentual de previsões corretas sobre o total de casos avaliados.",
    "balanced_accuracy": "Acurácia calculada por classe e depois promediada, o que evita que a classe majoritária domine o resultado.",
    "precision": "Entre os casos que o modelo apontou como de uma classe, quantos realmente eram dela.",
    "recall": "Entre os casos que de fato pertenciam a uma classe, quantos o modelo conseguiu encontrar.",
    "f1": "Média harmônica entre precisão e revocação, útil quando errar para mais e para menos custa caro.",
    "roc_auc": "Probabilidade de o modelo dar uma nota mais alta a um caso positivo do que a um negativo escolhidos ao acaso.",
    "cohen_kappa": "Mede o quanto o modelo supera o acerto que se obteria apenas por sorte.",
    "matthews_corrcoef": "Correlação entre previsto e real que só fica alta quando o modelo acerta em todas as classes.",
    "n_classes": "Quantidade de categorias distintas que o modelo aprendeu a distinguir.",
    # Regressão
    "r2": "Fração da variação do alvo que o modelo consegue explicar; 1 é perfeito e 0 equivale a chutar sempre a média.",
    "adjusted_r2": "O R² penalizado pelo número de colunas usadas, para não recompensar o modelo por acrescentar variáveis inúteis.",
    "mae": "Erro médio em módulo, na mesma unidade do alvo: quanto o modelo erra em um caso típico.",
    "rmse": "Raiz do erro quadrático médio; pune erros grandes com mais peso do que o erro médio.",
    "mse": "Média dos erros elevados ao quadrado, base de cálculo do RMSE.",
    "median_absolute_error": "Erro em módulo do caso mediano, pouco afetado por alguns poucos erros muito grandes.",
    "mape": "Erro médio expresso em percentual do valor real, útil para comparar alvos de escalas diferentes.",
    "explained_variance": "Quanto da dispersão dos valores reais o modelo reproduz, ignorando um eventual viés constante.",
    # Contexto
    "n_train": "Quantidade de registros usados para o modelo aprender.",
    "n_test": "Quantidade de registros separados e nunca vistos no treino, usados para medir o desempenho.",
}


def _format_number(value: Optional[float], *, percent: bool = False, digits: int = 4) -> str:
    """Formata um número para leitura humana em pt-BR.

    Args:
        value: Valor a formatar; ``None`` e ``NaN`` viram texto informativo.
        percent: Se ``True``, multiplica por 100 e acrescenta o símbolo ``%``.
        digits: Casas decimais para valores absolutos; ``0`` formata como inteiro.

    Returns:
        Texto pronto para exibição, com vírgula decimal e ponto de milhar.
    """
    if percent:
        return format_percentage(value)
    if digits == 0:
        return format_integer(value)
    return format_number(value, decimals=digits)


def _row(
    key: str,
    label: str,
    value: Optional[float],
    *,
    percent: bool = False,
    higher_is_better: bool = True,
    digits: int = 4,
) -> MetricRow:
    """Monta uma :class:`MetricRow` já formatada e explicada.

    Args:
        key: Identificador da métrica no glossário.
        label: Nome em pt-BR.
        value: Valor bruto.
        percent: Se deve ser exibida como percentual.
        higher_is_better: Se valores maiores são melhores.
        digits: Casas decimais.

    Returns:
        A linha de métrica pronta para a tabela.
    """
    return MetricRow(
        key=key,
        label=label,
        value=None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value),
        formatted=_format_number(value, percent=percent, digits=digits),
        explanation=METRIC_GLOSSARY.get(key, ""),
        higher_is_better=higher_is_better,
    )


def _safe(function, *args, **kwargs) -> Optional[float]:
    """Executa um cálculo de métrica devolvendo ``None`` em caso de falha.

    Algumas métricas são indefinidas em certos cenários (uma única classe no
    conjunto de teste, alvo com valor zero no MAPE). Falhar a tela inteira por
    causa de uma métrica seria desproporcional; devolver ``None`` mantém as
    demais visíveis e sinaliza a lacuna ao aprendiz.

    Args:
        function: Função de métrica do scikit-learn.
        *args: Argumentos posicionais repassados.
        **kwargs: Argumentos nomeados repassados.

    Returns:
        O valor calculado, ou ``None`` se o cálculo não for aplicável.
    """
    try:
        result = float(function(*args, **kwargs))
    except (ValueError, ZeroDivisionError, TypeError, IndexError):
        return None
    return None if math.isnan(result) or math.isinf(result) else result


def classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    y_proba: Optional[np.ndarray] = None,
    class_labels: Optional[Sequence] = None,
    n_train: int = 0,
    n_test: int = 0,
) -> List[MetricRow]:
    """Calcula o painel completo de métricas de classificação.

    Args:
        y_true: Rótulos verdadeiros do conjunto de teste.
        y_pred: Rótulos previstos pelo modelo.
        y_proba: Matriz de probabilidades por classe, quando disponível.
        class_labels: Rótulos de classe na ordem das colunas de ``y_proba``.
        n_train: Tamanho do conjunto de treino.
        n_test: Tamanho do conjunto de teste.

    Returns:
        Lista de :class:`MetricRow` ordenada da métrica mais geral para a mais
        específica.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    labels = list(class_labels) if class_labels is not None else sorted(set(y_true_array.tolist()))

    rows: List[MetricRow] = [
        _row("accuracy", "Acurácia", _safe(accuracy_score, y_true_array, y_pred_array), percent=True),
        _row(
            "balanced_accuracy",
            "Acurácia balanceada",
            _safe(balanced_accuracy_score, y_true_array, y_pred_array),
            percent=True,
        ),
        _row(
            "precision",
            "Precisão (média ponderada)",
            _safe(precision_score, y_true_array, y_pred_array, average="weighted", zero_division=0),
            percent=True,
        ),
        _row(
            "recall",
            "Revocação (média ponderada)",
            _safe(recall_score, y_true_array, y_pred_array, average="weighted", zero_division=0),
            percent=True,
        ),
        _row(
            "f1",
            "F1 (média ponderada)",
            _safe(f1_score, y_true_array, y_pred_array, average="weighted", zero_division=0),
            percent=True,
        ),
    ]

    # ROC AUC exige probabilidades e ao menos duas classes presentes no teste.
    auc: Optional[float] = None
    if y_proba is not None and len(labels) >= 2 and len(set(y_true_array.tolist())) >= 2:
        if len(labels) == 2 and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            auc = _safe(roc_auc_score, y_true_array, y_proba[:, 1], labels=labels)
        else:
            auc = _safe(
                roc_auc_score,
                y_true_array,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=labels,
            )
    rows.append(_row("roc_auc", "Área sob a curva ROC", auc))

    rows.extend(
        [
            _row("cohen_kappa", "Kappa de Cohen", _safe(cohen_kappa_score, y_true_array, y_pred_array)),
            _row(
                "matthews_corrcoef",
                "Coeficiente de Matthews",
                _safe(matthews_corrcoef, y_true_array, y_pred_array),
            ),
            _row("n_classes", "Classes distintas", float(len(labels)), digits=0),
            _row("n_train", "Registros de treino", float(n_train), digits=0),
            _row("n_test", "Registros de teste", float(n_test), digits=0),
        ]
    )
    return rows


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    n_features: int = 0,
    n_train: int = 0,
    n_test: int = 0,
) -> List[MetricRow]:
    """Calcula o painel completo de métricas de regressão.

    Args:
        y_true: Valores verdadeiros do conjunto de teste.
        y_pred: Valores previstos pelo modelo.
        n_features: Número de colunas de entrada, necessário para o R² ajustado.
        n_train: Tamanho do conjunto de treino.
        n_test: Tamanho do conjunto de teste.

    Returns:
        Lista de :class:`MetricRow` combinando métricas relativas (R²) e
        absolutas (MAE, RMSE), porque cada grupo responde a uma pergunta
        diferente sobre o mesmo modelo.
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    r2 = _safe(r2_score, y_true_array, y_pred_array)
    mse = _safe(mean_squared_error, y_true_array, y_pred_array)
    rmse = math.sqrt(mse) if mse is not None else None

    # R² ajustado só é definido quando há mais observações do que colunas + 1.
    adjusted: Optional[float] = None
    n_samples = len(y_true_array)
    if r2 is not None and n_samples > n_features + 1 and n_features > 0:
        adjusted = 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

    return [
        _row("r2", "R² (coeficiente de determinação)", r2),
        _row("adjusted_r2", "R² ajustado", adjusted),
        _row("mae", "Erro absoluto médio (MAE)", _safe(mean_absolute_error, y_true_array, y_pred_array), higher_is_better=False),
        _row("rmse", "Raiz do erro quadrático médio (RMSE)", rmse, higher_is_better=False),
        _row("mse", "Erro quadrático médio (MSE)", mse, higher_is_better=False),
        _row(
            "median_absolute_error",
            "Erro absoluto mediano",
            _safe(median_absolute_error, y_true_array, y_pred_array),
            higher_is_better=False,
        ),
        _row(
            "mape",
            "Erro percentual absoluto médio (MAPE)",
            _safe(mean_absolute_percentage_error, y_true_array, y_pred_array),
            percent=True,
            higher_is_better=False,
        ),
        _row(
            "explained_variance",
            "Variância explicada",
            _safe(explained_variance_score, y_true_array, y_pred_array),
        ),
        _row("n_train", "Registros de treino", float(n_train), digits=0),
        _row("n_test", "Registros de teste", float(n_test), digits=0),
    ]


def per_class_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    class_labels: Optional[Sequence] = None,
) -> List[ClassMetricRow]:
    """Calcula precisão, revocação e F1 para cada categoria separadamente.

    Conceito de ML envolvido: a acurácia é uma média, e média esconde
    desequilíbrio. Um modelo que acerta 95% dos casos da classe majoritária e
    nenhum da minoritária ainda exibe acurácia alta. A tabela por classe é o
    que revela esse fracasso — daí ela ser a visão principal da etapa de
    resultado, e não um detalhe opcional.

    Args:
        y_true: Rótulos verdadeiros do conjunto de teste.
        y_pred: Rótulos previstos pelo modelo.
        class_labels: Ordem das categorias; se omitida, usa a ordem alfabética.

    Returns:
        Uma :class:`ClassMetricRow` por categoria, na ordem informada.
    """
    y_true_array = np.asarray([str(item) for item in y_true])
    y_pred_array = np.asarray([str(item) for item in y_pred])
    labels = (
        [str(item) for item in class_labels]
        if class_labels is not None
        else sorted(set(y_true_array.tolist()) | set(y_pred_array.tolist()))
    )

    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_array,
            y_pred_array,
            labels=labels,
            average=None,
            zero_division=0,
        )
    except ValueError:
        return []

    return [
        ClassMetricRow(
            label=label,
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(labels)
    ]
