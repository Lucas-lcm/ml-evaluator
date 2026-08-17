"""Testes do catálogo de modelos."""

from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator

from scr.core import model_registry


@pytest.mark.parametrize(("task_type", "minimo"), [("classification", 4), ("regression", 3)])
def test_catalogo_tem_o_tamanho_definido_pela_spec_3(task_type: str, minimo: int) -> None:
    # A spec 1 pedia 4 de cada tipo; a spec 3 reduziu a lista deliberadamente
    # para 4 classificadores e 3 regressores.
    assert len(model_registry.list_models(task_type)) == minimo


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_chaves_sao_unicas(task_type: str) -> None:
    chaves = [spec.key for spec in model_registry.list_models(task_type)]
    assert len(chaves) == len(set(chaves))


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_toda_explicacao_responde_as_tres_perguntas(task_type: str) -> None:
    # Spec 3: explicações objetivas divididas em o que faz / como funciona / quando usar.
    for spec in model_registry.list_models(task_type):
        assert spec.display_name.strip()
        for campo in (spec.summary, spec.how_it_works, spec.when_to_use):
            assert campo.strip(), f"{spec.key} tem campo de explicação vazio"
            assert campo.strip().endswith("."), f"{spec.key}: explicação deve ser uma frase"
        assert spec.summary in spec.description


CLASSIFICADORES_ESPERADOS = {
    "logistic_regression",
    "random_forest_classifier",
    "decision_tree_classifier",
    "knn_classifier",
}
REGRESSORES_ESPERADOS = {
    "linear_regression",
    "random_forest_regressor",
    "decision_tree_regressor",
}


def test_catalogo_reduzido_conforme_spec_3() -> None:
    assert {s.key for s in model_registry.list_models("classification")} == CLASSIFICADORES_ESPERADOS
    assert {s.key for s in model_registry.list_models("regression")} == REGRESSORES_ESPERADOS


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_factory_devolve_instancia_nova(task_type: str) -> None:
    for spec in model_registry.list_models(task_type):
        primeiro = spec.build()
        segundo = spec.build()
        assert isinstance(primeiro, BaseEstimator)
        # Instâncias distintas evitam vazamento de estado entre sessões.
        assert primeiro is not segundo


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_modelo_padrao_existe_no_catalogo(task_type: str) -> None:
    chave = model_registry.default_model_key(task_type)
    assert model_registry.get_model(task_type, chave).key == chave


def test_get_model_com_chave_inexistente_falha() -> None:
    with pytest.raises(KeyError):
        model_registry.get_model("classification", "modelo_que_nao_existe")


def test_list_models_com_tarefa_invalida_falha() -> None:
    with pytest.raises(KeyError):
        model_registry.list_models("agrupamento")


def test_classificadores_declaram_suporte_a_probabilidade() -> None:
    for spec in model_registry.list_models("classification"):
        estimador = spec.build()
        assert spec.supports_probability == hasattr(estimador, "predict_proba")
