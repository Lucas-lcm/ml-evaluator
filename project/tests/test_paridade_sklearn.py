"""Paridade entre a ferramenta e o fluxo manual do scikit-learn.

Motivação (spec 3): ao comparar a tela de resultado com um treino feito "por
fora", os números precisam bater. Se não baterem, a ferramenta está ensinando
errado — e uma ferramenta didática que mente sobre a própria métrica é pior que
nenhuma ferramenta.

Estes testes reproduzem o fluxo canônico (`train_test_split` → `fit` →
`accuracy_score`) e exigem igualdade exata com o que `train_model` reporta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from scr.core import config, trainer


def test_acuracia_bate_com_o_fluxo_manual(classification_frame: pd.DataFrame) -> None:
    features = ["idade", "renda", "regiao"]

    resultado = trainer.train_model(
        classification_frame,
        target="aprovado",
        features=features,
        task_type="classification",
        model_key="decision_tree_classifier",
    )

    # Mesmo split, mesma semente, mesmo modelo — codificação one-hot equivalente
    # à do pipeline para a coluna categórica.
    X = pd.get_dummies(classification_frame[features], columns=["regiao"])
    y = classification_frame["aprovado"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.DEFAULT_TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    manual = DecisionTreeClassifier(random_state=config.RANDOM_STATE).fit(x_train, y_train)
    esperado = accuracy_score(y_test, manual.predict(x_test))

    assert resultado.metric("accuracy") == pytest.approx(esperado, abs=1e-12)
    assert resultado.n_train == len(x_train)
    assert resultado.n_test == len(x_test)


def test_r2_bate_com_o_fluxo_manual(regression_frame: pd.DataFrame) -> None:
    features = ["area", "quartos", "bairro"]

    resultado = trainer.train_model(
        regression_frame,
        target="preco",
        features=features,
        task_type="regression",
        model_key="linear_regression",
    )

    X = pd.get_dummies(regression_frame[features], columns=["bairro"])
    y = regression_frame["preco"].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=config.DEFAULT_TEST_SIZE, random_state=config.RANDOM_STATE
    )
    manual = LinearRegression().fit(x_train, y_train)
    esperado = r2_score(y_test, manual.predict(x_test))

    # A padronização do pipeline não altera o R² de uma regressão linear, mas
    # muda os últimos bits do cálculo; a tolerância cobre exatamente isso.
    assert resultado.metric("r2") == pytest.approx(esperado, abs=1e-6)


def test_metricas_por_classe_batem_com_o_scikit_learn(
    classification_frame: pd.DataFrame,
) -> None:
    resultado = trainer.train_model(
        classification_frame,
        target="aprovado",
        features=["idade", "renda", "regiao"],
        task_type="classification",
        model_key="random_forest_classifier",
    )

    labels = [linha.label for linha in resultado.per_class]
    x_test_predictions = resultado.pipeline.predict
    assert callable(x_test_predictions)

    # O suporte somado precisa ser exatamente o tamanho do conjunto de teste.
    assert sum(linha.support for linha in resultado.per_class) == resultado.n_test
    assert labels == sorted(labels)

    # Coerência interna: F1 é a média harmônica de precisão e revocação.
    for linha in resultado.per_class:
        if linha.precision + linha.recall > 0:
            esperado = (
                2 * linha.precision * linha.recall / (linha.precision + linha.recall)
            )
            assert linha.f1 == pytest.approx(esperado, abs=1e-9)


def test_acuracia_e_a_media_ponderada_das_revocacoes(
    classification_frame: pd.DataFrame,
) -> None:
    resultado = trainer.train_model(
        classification_frame,
        target="aprovado",
        features=["idade", "renda", "regiao"],
        task_type="classification",
        model_key="logistic_regression",
    )

    total = sum(linha.support for linha in resultado.per_class)
    ponderada = sum(linha.recall * linha.support for linha in resultado.per_class) / total

    assert resultado.metric("accuracy") == pytest.approx(ponderada, abs=1e-9)


def test_relatorio_por_classe_equivale_ao_do_scikit_learn() -> None:
    y_true = ["a", "a", "b", "b", "b", "a"]
    y_pred = ["a", "b", "b", "b", "a", "a"]

    from scr.core.metrics import per_class_metrics

    linhas = per_class_metrics(y_true, y_pred, class_labels=["a", "b"])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=["a", "b"], average=None, zero_division=0
    )

    for indice, linha in enumerate(linhas):
        assert linha.precision == pytest.approx(float(precision[indice]))
        assert linha.recall == pytest.approx(float(recall[indice]))
        assert linha.f1 == pytest.approx(float(f1[indice]))
        assert linha.support == int(support[indice])


def test_treino_e_reproduzivel(classification_frame: pd.DataFrame) -> None:
    # Semente fixa: repetir o experimento tem de devolver a mesma métrica, senão
    # o aprendiz não consegue atribuir a variação à mudança que ele fez.
    argumentos = dict(
        target="aprovado",
        features=["idade", "renda", "regiao"],
        task_type="classification",
        model_key="random_forest_classifier",
    )
    primeiro = trainer.train_model(classification_frame, **argumentos)
    segundo = trainer.train_model(classification_frame, **argumentos)

    assert primeiro.metric("accuracy") == segundo.metric("accuracy")
    assert np.allclose(
        [linha.f1 for linha in primeiro.per_class],
        [linha.f1 for linha in segundo.per_class],
    )
