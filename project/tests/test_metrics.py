"""Testes do cálculo e da apresentação das métricas."""

from __future__ import annotations

import numpy as np

from scr.core.metrics import METRIC_GLOSSARY, classification_metrics, regression_metrics


def test_classificacao_perfeita_atinge_acuracia_maxima() -> None:
    y = ["a", "b", "a", "b"]
    linhas = {linha.key: linha for linha in classification_metrics(y, y)}

    assert linhas["accuracy"].value == 1.0
    assert linhas["accuracy"].formatted == "100,00%"
    assert linhas["accuracy"].higher_is_better is True


def test_metrica_indisponivel_nao_quebra_a_tabela() -> None:
    # Uma única classe no teste torna a ROC AUC indefinida.
    linhas = {linha.key: linha for linha in classification_metrics(["a", "a"], ["a", "a"])}

    assert linhas["roc_auc"].value is None
    assert linhas["roc_auc"].formatted == "não disponível"


def test_roc_auc_binaria_e_calculada_com_probabilidades() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "a", "b", "b"]
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    linhas = {
        linha.key: linha
        for linha in classification_metrics(y_true, y_pred, y_proba=proba, class_labels=["a", "b"])
    }
    assert linhas["roc_auc"].value == 1.0


def test_regressao_perfeita_atinge_r2_um() -> None:
    y = [1.0, 2.0, 3.0, 4.0]
    linhas = {linha.key: linha for linha in regression_metrics(y, y, n_features=1)}

    assert linhas["r2"].value == 1.0
    assert linhas["mae"].value == 0.0
    # Erros são métricas onde menor é melhor: a direção precisa ser informada.
    assert linhas["mae"].higher_is_better is False


def test_r2_ajustado_so_existe_quando_ha_amostras_suficientes() -> None:
    linhas = {linha.key: linha for linha in regression_metrics([1.0, 2.0], [1.1, 2.1], n_features=5)}
    assert linhas["adjusted_r2"].value is None


def test_todas_as_metricas_possuem_explicacao_no_glossario() -> None:
    y_class = ["a", "b", "a", "b"]
    y_reg = [1.0, 2.0, 3.0, 4.0]

    for linha in classification_metrics(y_class, y_class) + regression_metrics(y_reg, y_reg):
        assert linha.key in METRIC_GLOSSARY
        assert linha.explanation == METRIC_GLOSSARY[linha.key]
        assert linha.explanation.strip()


def test_numeros_usam_virgula_decimal() -> None:
    linhas = {linha.key: linha for linha in regression_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])}
    assert "," in linhas["mae"].formatted
