"""Testes de inferência de tarefa, validação e treino."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scr.core import model_registry, trainer
from scr.core.trainer import TrainingError

_CLASSIFIERS = [spec.key for spec in model_registry.list_models("classification")]
_REGRESSORS = [spec.key for spec in model_registry.list_models("regression")]


# --------------------------------------------------------------------------- #
# Inferência de tipo de tarefa
# --------------------------------------------------------------------------- #


def test_infer_task_type_texto_e_classificacao() -> None:
    assert trainer.infer_task_type(pd.Series(["a", "b", "a", "c"])) == "classification"


def test_infer_task_type_binario_numerico_e_classificacao() -> None:
    assert trainer.infer_task_type(pd.Series([0, 1, 0, 1, 1])) == "classification"


def test_infer_task_type_continuo_e_regressao() -> None:
    valores = pd.Series(np.linspace(0.5, 100.5, 300))
    assert trainer.infer_task_type(valores) == "regression"


def test_infer_task_type_inteiros_com_poucas_categorias_e_classificacao() -> None:
    valores = pd.Series(list(range(5)) * 100)  # 5 valores distintos em 500 linhas
    assert trainer.infer_task_type(valores) == "classification"


def test_infer_task_type_com_coluna_vazia_nao_quebra() -> None:
    assert trainer.infer_task_type(pd.Series([None, None], dtype="object")) == "classification"


# --------------------------------------------------------------------------- #
# Sugestão de colunas de entrada
# --------------------------------------------------------------------------- #


def test_suggest_features_descarta_constantes_e_identificadores() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"cod-{i}" for i in range(120)],  # cardinalidade alta
            "constante": ["x"] * 120,
            "util": np.arange(120, dtype=float),
            "alvo": np.arange(120, dtype=float) * 2,
        }
    )
    sugeridas, motivos = trainer.suggest_features(frame, "alvo")

    assert sugeridas == ["util"]
    assert len(motivos) == 2


# --------------------------------------------------------------------------- #
# Vazamento de alvo (spec 3)
# --------------------------------------------------------------------------- #


def test_detecta_coluna_que_espelha_o_alvo(classification_frame: pd.DataFrame) -> None:
    frame = classification_frame.copy()
    frame["parecer_final"] = frame["aprovado"]

    suspeitas = trainer.detect_leakage(
        frame, "aprovado", ["idade", "renda", "regiao", "parecer_final"], "classification"
    )
    assert set(suspeitas) == {"parecer_final"}
    assert suspeitas["parecer_final"] > 0.95


def test_detecta_copia_do_alvo_em_regressao(regression_frame: pd.DataFrame) -> None:
    frame = regression_frame.copy()
    frame["preco_em_milhares"] = frame["preco"] / 1000  # cópia reescalada

    suspeitas = trainer.detect_leakage(
        frame, "preco", ["area", "quartos", "preco_em_milhares"], "regression"
    )
    assert set(suspeitas) == {"preco_em_milhares"}


def test_coluna_util_nao_e_confundida_com_vazamento(regression_frame: pd.DataFrame) -> None:
    # 'area' é fortemente correlacionada com o preço, mas legítima: a heurística
    # precisa ser conservadora o bastante para não desmarcá-la.
    suspeitas = trainer.detect_leakage(
        regression_frame, "preco", ["area", "quartos", "bairro"], "regression"
    )
    assert suspeitas == {}


def test_coluna_vazando_e_desmarcada_com_explicacao(classification_frame: pd.DataFrame) -> None:
    frame = classification_frame.copy()
    frame["parecer_final"] = frame["aprovado"]

    sugeridas, motivos = trainer.suggest_features(frame, "aprovado", "classification")

    assert "parecer_final" not in sugeridas
    assert any("vazamento" in motivo for motivo in motivos)


def test_vazamento_inflaria_a_acuracia_para_cem_por_cento(
    classification_frame: pd.DataFrame,
) -> None:
    # Registra o comportamento que motivou a checagem: com a coluna vazando, a
    # acurácia vai a 100%; sem ela, fica em patamar realista.
    frame = classification_frame.copy()
    frame["parecer_final"] = frame["aprovado"]

    com_vazamento = trainer.train_model(
        frame, "aprovado", ["idade", "parecer_final"], "classification", "decision_tree_classifier"
    )
    sem_vazamento = trainer.train_model(
        frame, "aprovado", ["idade", "renda", "regiao"], "classification", "decision_tree_classifier"
    )

    assert (com_vazamento.metric("accuracy") or 0) == 1.0
    assert (sem_vazamento.metric("accuracy") or 0) < 1.0


# --------------------------------------------------------------------------- #
# Validação
# --------------------------------------------------------------------------- #


def test_validacao_rejeita_alvo_entre_as_entradas(classification_frame: pd.DataFrame) -> None:
    with pytest.raises(TrainingError, match="copiar a resposta"):
        trainer.validate_training_request(
            classification_frame, "aprovado", ["aprovado", "idade"], "classification"
        )


def test_validacao_rejeita_alvo_textual_em_regressao(classification_frame: pd.DataFrame) -> None:
    with pytest.raises(TrainingError, match="precisa ser numérico"):
        trainer.validate_training_request(
            classification_frame, "regiao", ["idade"], "regression"
        )


def test_validacao_rejeita_base_pequena_demais() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [1, 0, 1]})
    with pytest.raises(TrainingError, match="ao menos"):
        trainer.validate_training_request(frame, "y", ["x"], "classification")


def test_validacao_rejeita_alvo_com_uma_unica_classe() -> None:
    frame = pd.DataFrame({"x": range(40), "y": ["a"] * 40})
    with pytest.raises(TrainingError, match="duas categorias"):
        trainer.validate_training_request(frame, "y", ["x"], "classification")


def test_validacao_rejeita_coluna_inexistente(classification_frame: pd.DataFrame) -> None:
    with pytest.raises(TrainingError):
        trainer.validate_training_request(
            classification_frame, "aprovado", ["coluna_fantasma"], "classification"
        )


# --------------------------------------------------------------------------- #
# Treino
# --------------------------------------------------------------------------- #


def test_treino_de_classificacao_produz_metricas_uteis(classification_frame: pd.DataFrame) -> None:
    resultado = trainer.train_model(
        classification_frame,
        target="aprovado",
        features=["idade", "renda", "regiao"],
        task_type="classification",
        model_key="random_forest_classifier",
    )

    assert resultado.task_type == "classification"
    assert resultado.class_labels is not None and len(resultado.class_labels) == 2
    assert resultado.n_train > resultado.n_test > 0
    # Base sintética separável: um modelo funcional precisa superar o acaso.
    assert (resultado.metric("accuracy") or 0) > 0.6
    assert resultado.metric("f1") is not None
    assert resultado.metric("roc_auc") is not None
    # A spec 2 exige o painel completo, não apenas a acurácia.
    assert len(resultado.metrics) >= 8


def test_treino_de_regressao_produz_metricas_uteis(regression_frame: pd.DataFrame) -> None:
    resultado = trainer.train_model(
        regression_frame,
        target="preco",
        features=["area", "quartos", "bairro"],
        task_type="regression",
        model_key="linear_regression",
    )

    assert resultado.task_type == "regression"
    assert (resultado.metric("r2") or 0) > 0.8  # relação sintética é linear
    assert resultado.metric("mae") is not None
    assert resultado.metric("rmse") is not None
    assert resultado.residual_std is not None and resultado.residual_std > 0
    assert resultado.target_range is not None and resultado.target_range > 0


def test_toda_metrica_tem_explicacao_em_portugues(classification_frame: pd.DataFrame) -> None:
    resultado = trainer.train_model(
        classification_frame,
        target="aprovado",
        features=["idade", "renda"],
        task_type="classification",
        model_key="logistic_regression",
    )
    for linha in resultado.metrics:
        assert linha.explanation.strip(), f"métrica {linha.key} sem explicação"
        assert linha.formatted.strip()


def test_treino_lida_com_valores_ausentes(classification_frame: pd.DataFrame) -> None:
    frame = classification_frame.copy()
    frame.loc[frame.index[:20], "renda"] = np.nan
    frame.loc[frame.index[20:35], "regiao"] = None
    frame.loc[frame.index[35:45], "aprovado"] = None

    resultado = trainer.train_model(
        frame,
        target="aprovado",
        features=["idade", "renda", "regiao"],
        task_type="classification",
        model_key="decision_tree_classifier",
    )

    assert resultado.n_train + resultado.n_test == len(frame) - 10
    assert any("alvo" in aviso for aviso in resultado.warnings)


def test_column_specs_acompanham_as_features(regression_frame: pd.DataFrame) -> None:
    resultado = trainer.train_model(
        regression_frame,
        target="preco",
        features=["area", "bairro"],
        task_type="regression",
        model_key="decision_tree_regressor",
    )
    assert [spec.name for spec in resultado.column_specs] == ["area", "bairro"]


@pytest.mark.parametrize("model_key", _CLASSIFIERS)
def test_todos_os_classificadores_treinam(classification_frame: pd.DataFrame, model_key: str) -> None:
    resultado = trainer.train_model(
        classification_frame,
        target="aprovado",
        features=["idade", "renda", "regiao"],
        task_type="classification",
        model_key=model_key,
    )
    assert resultado.metric("accuracy") is not None


@pytest.mark.parametrize("model_key", _REGRESSORS)
def test_todos_os_regressores_treinam(regression_frame: pd.DataFrame, model_key: str) -> None:
    resultado = trainer.train_model(
        regression_frame,
        target="preco",
        features=["area", "quartos", "bairro"],
        task_type="regression",
        model_key=model_key,
    )
    assert resultado.metric("mae") is not None
