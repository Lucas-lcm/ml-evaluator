"""Predição de casos novos com estimativa de confiança.

Conceito de ML envolvido: um modelo que devolve apenas "a resposta" ensina o
aprendiz a confiar cegamente nele. Um modelo que devolve "a resposta **e** o
quanto ela é incerta" ensina julgamento. Este módulo trata a confiança como
parte obrigatória da saída, com origens diferentes conforme a tarefa:

    * **Classificação** — a probabilidade da classe vencedora, quando o modelo
      a fornece. É a confiança *calibrada pelo próprio modelo*.
    * **Regressão** — não existe probabilidade; a incerteza vem da dispersão do
      erro. Em comitês de árvores usamos a discordância entre as árvores; nos
      demais modelos, o desvio-padrão dos resíduos medidos no conjunto de teste.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from scr.core.formatting import format_number, format_percentage
from scr.core.schema import ColumnSpec, PredictionResult, TrainingResult

_Z_95 = 1.959963985  # quantil normal para um intervalo de 95%

_ENSEMBLE_WITH_INDEPENDENT_MEMBERS = {
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "BaggingRegressor",
}
"""Comitês cujos membros preveem o alvo de forma independente.

Boosting fica de fora de propósito: ali cada árvore prevê o *resíduo* da
anterior, então a dispersão entre elas não representa incerteza sobre o valor.
"""


class PredictionError(Exception):
    """Erro de predição com mensagem em pt-BR pronta para exibição."""


def build_input_frame(
    values: Mapping[str, Any],
    column_specs: List[ColumnSpec],
) -> pd.DataFrame:
    """Converte os valores digitados no formulário em uma linha de DataFrame.

    A ordem e os tipos das colunas precisam reproduzir exatamente o que o
    pipeline viu no treino; qualquer divergência aqui gera um erro obscuro lá
    dentro. Fazer a conversão explicitamente mantém o problema visível.

    Args:
        values: Mapa ``{coluna: valor}`` vindo da interface.
        column_specs: Especificações geradas no treino.

    Returns:
        DataFrame de uma única linha, com as colunas na ordem do treino.

    Raises:
        PredictionError: Se faltar alguma coluna ou um valor numérico for
            inválido.
    """
    row: Dict[str, Any] = {}

    for spec in column_specs:
        if spec.name not in values:
            raise PredictionError(f"O campo '{spec.name}' não foi preenchido.")
        raw = values[spec.name]

        if spec.kind == "numeric":
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                row[spec.name] = np.nan  # o imputador do pipeline resolve
                continue
            try:
                row[spec.name] = float(raw)
            except (TypeError, ValueError) as exc:
                raise PredictionError(
                    f"O campo '{spec.name}' espera um número, mas recebeu '{raw}'."
                ) from exc
        else:
            row[spec.name] = None if raw is None else str(raw)

    return pd.DataFrame([row], columns=[spec.name for spec in column_specs])


def detect_out_of_range(
    values: Mapping[str, Any],
    column_specs: List[ColumnSpec],
) -> List[str]:
    """Aponta valores fora da faixa que o modelo observou no treino.

    Prever fora do intervalo conhecido é *extrapolação*, e nenhuma métrica de
    teste cobre esse cenário. Avisar é mais honesto do que devolver um número
    com aparência de certeza.

    Args:
        values: Valores informados para a predição.
        column_specs: Especificações geradas no treino.

    Returns:
        Lista de avisos em pt-BR; vazia quando tudo está dentro da faixa.
    """
    alerts: List[str] = []

    for spec in column_specs:
        raw = values.get(spec.name)
        if raw is None:
            continue

        if spec.kind == "numeric":
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if spec.minimum is not None and value < spec.minimum:
                alerts.append(
                    f"O valor de '{spec.name}' ({format_number(value)}) está abaixo do mínimo "
                    f"visto no treino ({format_number(spec.minimum)}). A previsão é uma extrapolação."
                )
            elif spec.maximum is not None and value > spec.maximum:
                alerts.append(
                    f"O valor de '{spec.name}' ({format_number(value)}) está acima do máximo "
                    f"visto no treino ({format_number(spec.maximum)}). A previsão é uma extrapolação."
                )
        else:
            if spec.categories and str(raw) not in spec.categories:
                alerts.append(
                    f"A categoria '{raw}' em '{spec.name}' não apareceu no treino e será "
                    "tratada como desconhecida."
                )

    return alerts


def _format_value(value: float) -> str:
    """Formata um número previsto no padrão pt-BR.

    Args:
        value: Valor previsto.

    Returns:
        Texto formatado para exibição.
    """
    return format_number(value)


def _regression_sigma(result: TrainingResult, input_frame: pd.DataFrame) -> tuple[float, str]:
    """Estima o desvio-padrão da previsão para um caso individual.

    Args:
        result: Resultado do treino, com pipeline e resíduos.
        input_frame: Linha a prever.

    Returns:
        Tupla ``(sigma, origem)``, onde ``origem`` é ``"comitê"`` quando a
        incerteza vem da discordância entre os membros do comitê e
        ``"resíduos"`` quando vem do erro medido no conjunto de teste.
    """
    estimator = result.pipeline.named_steps["modelo"]
    residual_sigma = float(result.residual_std or 0.0)

    if estimator.__class__.__name__ in _ENSEMBLE_WITH_INDEPENDENT_MEMBERS:
        try:
            transformed = result.pipeline.named_steps["preprocessamento"].transform(input_frame)
            member_predictions = np.array(
                [float(member.predict(transformed)[0]) for member in estimator.estimators_]
            )
            spread = float(np.std(member_predictions, ddof=1))
            # A incerteza total combina o desacordo do comitê com o erro
            # irredutível observado no teste.
            return float(np.hypot(spread, residual_sigma)), "comitê"
        except Exception:
            pass  # qualquer falha cai no caminho robusto abaixo

    return residual_sigma, "resíduos"


def predict_one(
    result: TrainingResult,
    values: Mapping[str, Any],
) -> PredictionResult:
    """Executa uma predição individual e estima sua confiabilidade.

    Args:
        result: Resultado de um treino concluído.
        values: Valores informados pelo usuário, um por coluna de entrada.

    Returns:
        O :class:`PredictionResult` com o valor previsto, a confiança e a
        explicação de como interpretá-la.

    Raises:
        PredictionError: Se a entrada for inválida ou o pipeline falhar.
    """
    input_frame = build_input_frame(values, result.column_specs)

    try:
        raw_prediction = result.pipeline.predict(input_frame)[0]
    except Exception as exc:
        raise PredictionError(f"Não foi possível calcular a previsão: {exc}") from exc

    if result.task_type == "classification":
        return _classification_result(result, input_frame, raw_prediction)
    return _regression_result(result, input_frame, float(raw_prediction))


def _classification_result(
    result: TrainingResult,
    input_frame: pd.DataFrame,
    raw_prediction: Any,
) -> PredictionResult:
    """Monta o resultado de uma predição de classificação.

    Args:
        result: Resultado do treino.
        input_frame: Linha a prever.
        raw_prediction: Rótulo devolvido pelo pipeline.

    Returns:
        O :class:`PredictionResult` correspondente.
    """
    label = str(raw_prediction)
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None

    estimator = result.pipeline.named_steps["modelo"]
    if hasattr(estimator, "predict_proba"):
        try:
            row = result.pipeline.predict_proba(input_frame)[0]
            classes = [str(item) for item in estimator.classes_]
            probabilities = {
                name: float(value)
                for name, value in sorted(zip(classes, row), key=lambda pair: -pair[1])
            }
            confidence = float(max(row))
        except (AttributeError, ValueError):
            probabilities = None

    if confidence is not None:
        explanation = (
            f"O modelo atribuiu {format_percentage(confidence, decimals=1)} de probabilidade à "
            "categoria prevista. Quanto mais essa probabilidade se aproxima de 100%, mais o caso "
            "se parece com os exemplos que o modelo viu daquela categoria."
        )
        kind = "probability"
    else:
        fallback = result.metric("accuracy")
        confidence = float(fallback) if fallback is not None else None
        explanation = (
            "Este modelo não fornece probabilidade por classe. Como referência, mostramos a "
            "acurácia obtida no conjunto de teste, que indica o acerto médio esperado."
        )
        kind = "probability" if confidence is not None else "unavailable"

    return PredictionResult(
        prediction=label,
        formatted_prediction=label,
        confidence=confidence,
        confidence_kind=kind,
        confidence_explanation=explanation,
        class_probabilities=probabilities,
    )


def _regression_result(
    result: TrainingResult,
    input_frame: pd.DataFrame,
    prediction: float,
) -> PredictionResult:
    """Monta o resultado de uma predição de regressão com intervalo.

    Args:
        result: Resultado do treino.
        input_frame: Linha a prever.
        prediction: Valor previsto pelo pipeline.

    Returns:
        O :class:`PredictionResult` com intervalo de 95% e confiança relativa.
    """
    sigma, origin = _regression_sigma(result, input_frame)

    if sigma <= 0:
        return PredictionResult(
            prediction=prediction,
            formatted_prediction=_format_value(prediction),
            confidence=None,
            confidence_kind="unavailable",
            confidence_explanation=(
                "Não há erro residual suficiente no conjunto de teste para estimar a incerteza "
                "desta previsão."
            ),
        )

    margin = _Z_95 * sigma
    interval = (prediction - margin, prediction + margin)

    # Confiança relativa: quanto da amplitude do alvo a faixa de erro consome.
    # Uma faixa estreita diante de um alvo muito variável indica previsão útil.
    amplitude = float(result.target_range or 0.0)
    if amplitude > 0:
        confidence = float(np.clip(1.0 - (2 * margin) / amplitude, 0.0, 1.0))
    else:
        confidence = None

    source = (
        "a discordância entre as árvores do comitê" if origin == "comitê" else "o erro medido no conjunto de teste"
    )
    explanation = (
        f"Com 95% de chance o valor real está entre {_format_value(interval[0])} e "
        f"{_format_value(interval[1])}. Essa faixa foi estimada a partir de {source}; "
        "quanto mais estreita em relação à variação do alvo, mais confiável é a previsão."
    )

    return PredictionResult(
        prediction=prediction,
        formatted_prediction=_format_value(prediction),
        confidence=confidence,
        confidence_kind="interval",
        confidence_explanation=explanation,
        interval=interval,
    )
