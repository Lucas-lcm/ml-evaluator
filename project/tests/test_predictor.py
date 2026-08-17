"""Testes de predição individual e estimativa de confiança."""

from __future__ import annotations

import pandas as pd
import pytest

from scr.core import predictor, trainer
from scr.core.predictor import PredictionError


@pytest.fixture(scope="module")
def classificador(classification_frame: pd.DataFrame):
    """Modelo de classificação treinado uma única vez para o módulo."""
    return trainer.train_model(
        classification_frame,
        target="aprovado",
        features=["idade", "renda", "regiao"],
        task_type="classification",
        model_key="random_forest_classifier",
    )


@pytest.fixture(scope="module")
def regressor(regression_frame: pd.DataFrame):
    """Modelo de regressão em comitê, que permite estimar dispersão."""
    return trainer.train_model(
        regression_frame,
        target="preco",
        features=["area", "quartos", "bairro"],
        task_type="regression",
        model_key="random_forest_regressor",
    )


@pytest.fixture(scope="module")
def regressor_linear(regression_frame: pd.DataFrame):
    """Modelo de regressão sem comitê: a incerteza vem dos resíduos."""
    return trainer.train_model(
        regression_frame,
        target="preco",
        features=["area", "quartos", "bairro"],
        task_type="regression",
        model_key="linear_regression",
    )


# --------------------------------------------------------------------------- #
# Montagem da entrada
# --------------------------------------------------------------------------- #


def test_build_input_frame_respeita_ordem_das_colunas(classificador) -> None:
    frame = predictor.build_input_frame(
        {"idade": 30, "renda": 4000, "regiao": "sul"}, classificador.column_specs
    )
    assert list(frame.columns) == ["idade", "renda", "regiao"]
    assert len(frame) == 1


def test_build_input_frame_falha_com_campo_faltando(classificador) -> None:
    with pytest.raises(PredictionError, match="não foi preenchido"):
        predictor.build_input_frame({"idade": 30, "renda": 4000}, classificador.column_specs)


def test_build_input_frame_falha_com_numero_invalido(classificador) -> None:
    with pytest.raises(PredictionError, match="espera um número"):
        predictor.build_input_frame(
            {"idade": "trinta", "renda": 4000, "regiao": "sul"}, classificador.column_specs
        )


def test_build_input_frame_aceita_numerico_vazio(classificador) -> None:
    frame = predictor.build_input_frame(
        {"idade": None, "renda": 4000, "regiao": "sul"}, classificador.column_specs
    )
    # O valor ausente é resolvido pelo imputador do pipeline, não aqui.
    assert frame["idade"].isna().all()


# --------------------------------------------------------------------------- #
# Classificação
# --------------------------------------------------------------------------- #


def test_predicao_de_classificacao_retorna_confianca(classificador) -> None:
    resultado = predictor.predict_one(
        classificador, {"idade": 45, "renda": 6000, "regiao": "sul"}
    )

    assert resultado.confidence_kind == "probability"
    assert resultado.confidence is not None and 0.0 <= resultado.confidence <= 1.0
    assert resultado.prediction in (classificador.class_labels or [])
    assert resultado.class_probabilities is not None
    assert pytest.approx(sum(resultado.class_probabilities.values()), abs=1e-6) == 1.0
    # A maior probabilidade deve corresponder à classe prevista.
    vencedora = next(iter(resultado.class_probabilities))
    assert vencedora == resultado.prediction


def test_probabilidades_vem_ordenadas_da_maior_para_a_menor(classificador) -> None:
    resultado = predictor.predict_one(
        classificador, {"idade": 22, "renda": 2000, "regiao": "norte"}
    )
    valores = list((resultado.class_probabilities or {}).values())
    assert valores == sorted(valores, reverse=True)


# --------------------------------------------------------------------------- #
# Regressão
# --------------------------------------------------------------------------- #


def test_predicao_de_regressao_retorna_intervalo(regressor) -> None:
    resultado = predictor.predict_one(
        regressor, {"area": 120.0, "quartos": 3, "bairro": "centro"}
    )

    assert resultado.confidence_kind == "interval"
    assert resultado.interval is not None
    inferior, superior = resultado.interval
    assert inferior < resultado.prediction < superior
    assert resultado.confidence is not None and 0.0 <= resultado.confidence <= 1.0


def test_regressao_sem_comite_usa_residuos(regressor_linear) -> None:
    resultado = predictor.predict_one(
        regressor_linear, {"area": 120.0, "quartos": 3, "bairro": "centro"}
    )
    assert resultado.interval is not None
    assert "conjunto de teste" in resultado.confidence_explanation


def test_predicao_de_regressao_e_plausivel(regressor_linear, regression_frame: pd.DataFrame) -> None:
    resultado = predictor.predict_one(
        regressor_linear, {"area": 150.0, "quartos": 3, "bairro": "praia"}
    )
    esperado = 1200 * 150 + 15000 * 3
    # Tolerância larga: o objetivo é detectar erro de pipeline, não medir ajuste.
    assert abs(float(resultado.prediction) - esperado) < 0.5 * esperado


# --------------------------------------------------------------------------- #
# Extrapolação
# --------------------------------------------------------------------------- #


def test_detecta_valor_numerico_fora_da_faixa_de_treino(classificador) -> None:
    alertas = predictor.detect_out_of_range(
        {"idade": 999, "renda": 5000, "regiao": "sul"}, classificador.column_specs
    )
    assert any("idade" in alerta and "acima do máximo" in alerta for alerta in alertas)


def test_detecta_categoria_desconhecida(classificador) -> None:
    alertas = predictor.detect_out_of_range(
        {"idade": 40, "renda": 5000, "regiao": "marte"}, classificador.column_specs
    )
    assert any("marte" in alerta for alerta in alertas)


def test_sem_alertas_quando_tudo_esta_dentro_da_faixa(classificador) -> None:
    valores = {spec.name: spec.default for spec in classificador.column_specs}
    assert predictor.detect_out_of_range(valores, classificador.column_specs) == []


def test_categoria_desconhecida_nao_derruba_a_predicao(classificador) -> None:
    # handle_unknown="ignore" no one-hot precisa manter a aplicação de pé.
    resultado = predictor.predict_one(
        classificador, {"idade": 40, "renda": 5000, "regiao": "marte"}
    )
    assert resultado.formatted_prediction
