"""Contratos de dados trocados entre as camadas (SDD).

Estas estruturas são a *interface* entre `core` e `ui`. A camada de
apresentação nunca inspeciona objetos scikit-learn diretamente: ela lê apenas
os campos declarados aqui. Isso mantém a UI substituível e o core testável.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

TaskType = Literal["classification", "regression"]
ColumnKind = Literal["numeric", "categorical"]
ConfidenceKind = Literal["probability", "interval", "unavailable"]


@dataclass(frozen=True)
class Dataset:
    """Tabela carregada e já validada.

    Attributes:
        frame: Dados em memória.
        source_name: Nome do arquivo de origem, exibido ao usuário.
        detected_separator: Separador identificado (apenas para formatos texto).
        detected_encoding: Codificação identificada (apenas para formatos texto).
        warnings: Mensagens não fatais geradas durante a leitura.
    """

    frame: pd.DataFrame
    source_name: str
    detected_separator: Optional[str] = None
    detected_encoding: Optional[str] = None
    warnings: Sequence[str] = field(default_factory=tuple)

    @property
    def n_rows(self) -> int:
        """Número de linhas da tabela."""
        return int(self.frame.shape[0])

    @property
    def n_cols(self) -> int:
        """Número de colunas da tabela."""
        return int(self.frame.shape[1])


@dataclass(frozen=True)
class ColumnSpec:
    """Descrição de uma coluna usada para gerar o formulário de predição.

    A UI constrói widgets a partir desta especificação sem conhecer pandas:
    `numeric` vira campo numérico com limites; `categorical` vira seletor com
    as categorias observadas no treino.

    Attributes:
        name: Nome original da coluna.
        kind: ``"numeric"`` ou ``"categorical"``.
        default: Valor pré-preenchido (mediana para numérico, moda para categórico).
        minimum: Menor valor observado no treino (apenas numérico).
        maximum: Maior valor observado no treino (apenas numérico).
        categories: Categorias observadas no treino (apenas categórico).
        has_missing: Se a coluna continha valores ausentes no conjunto original.
    """

    name: str
    kind: ColumnKind
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    categories: Sequence[str] = field(default_factory=tuple)
    has_missing: bool = False


@dataclass(frozen=True)
class MetricRow:
    """Uma linha da tabela de métricas exibida ao aprendiz.

    Attributes:
        key: Identificador estável da métrica.
        label: Nome em pt-BR.
        value: Valor numérico bruto (pode ser ``None`` quando não aplicável).
        formatted: Valor já formatado para leitura humana.
        explanation: Explicação de uma linha, em pt-BR.
        higher_is_better: Se valores maiores indicam desempenho melhor.
    """

    key: str
    label: str
    value: Optional[float]
    formatted: str
    explanation: str
    higher_is_better: bool


@dataclass(frozen=True)
class ClassMetricRow:
    """Desempenho do modelo em uma categoria específica.

    Métrica agregada esconde o caso mais comum de fracasso em classificação: o
    modelo acerta muito a classe majoritária e erra quase tudo na minoritária,
    e a acurácia geral continua alta. Quebrar por classe torna isso visível.

    Attributes:
        label: Nome da categoria.
        precision: Entre os casos apontados como desta categoria, quantos eram.
        recall: Entre os casos que eram desta categoria, quantos foram achados.
        f1: Média harmônica entre precisão e revocação.
        support: Quantidade de casos desta categoria no conjunto de teste.
    """

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class TrainingResult:
    """Resultado completo de um treino, suficiente para prever e explicar.

    Attributes:
        pipeline: Pipeline scikit-learn ajustado (pré-processamento + modelo).
        task_type: Tipo de tarefa resolvido.
        model_key: Chave do modelo no catálogo.
        model_display_name: Nome completo do modelo em pt-BR.
        target: Nome da coluna alvo.
        features: Colunas efetivamente usadas como entrada.
        column_specs: Especificações para montar o formulário de predição.
        metrics: Métricas calculadas no conjunto de teste.
        per_class: Desempenho por categoria (apenas classificação).
        class_labels: Rótulos de classe (apenas classificação).
        residual_std: Desvio-padrão dos resíduos no teste (apenas regressão).
        target_range: Amplitude observada do alvo (apenas regressão).
        n_train: Tamanho do conjunto de treino.
        n_test: Tamanho do conjunto de teste.
        training_seconds: Duração do ajuste.
        warnings: Avisos didáticos coletados durante o treino.
    """

    pipeline: Pipeline
    task_type: TaskType
    model_key: str
    model_display_name: str
    target: str
    features: List[str]
    column_specs: List[ColumnSpec]
    metrics: List[MetricRow]
    per_class: List[ClassMetricRow] = field(default_factory=list)
    class_labels: Optional[List[str]] = None
    residual_std: Optional[float] = None
    target_range: Optional[float] = None
    n_train: int = 0
    n_test: int = 0
    training_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def metric(self, key: str) -> Optional[float]:
        """Recupera o valor bruto de uma métrica pelo identificador.

        Args:
            key: Identificador da métrica (ex.: ``"accuracy"``, ``"r2"``).

        Returns:
            O valor da métrica, ou ``None`` se ela não foi calculada.
        """
        for row in self.metrics:
            if row.key == key:
                return row.value
        return None


@dataclass(frozen=True)
class PredictionResult:
    """Saída de uma predição individual, com sua medida de confiança.

    Attributes:
        prediction: Valor previsto (rótulo em classificação, número em regressão).
        formatted_prediction: Valor pronto para exibição.
        confidence: Escore de confiança em [0, 1], ou ``None`` se indisponível.
        confidence_kind: Origem da confiança (probabilidade, intervalo, indisponível).
        confidence_explanation: Frase em pt-BR explicando como ler a confiança.
        class_probabilities: Distribuição de probabilidade por classe.
        interval: Intervalo estimado ``(inferior, superior)`` para regressão.
    """

    prediction: Any
    formatted_prediction: str
    confidence: Optional[float]
    confidence_kind: ConfidenceKind
    confidence_explanation: str
    class_probabilities: Optional[Dict[str, float]] = None
    interval: Optional[tuple[float, float]] = None
