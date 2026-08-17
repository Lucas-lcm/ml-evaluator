"""Ingestão segura de arquivos tabulares (csv, txt, xlsx, json).

Conceito de dados envolvido: *data ingestion* é a primeira etapa do ciclo de
vida de um projeto de ciência de dados, e também a mais frágil. Arquivos reais
chegam com separadores inconsistentes, acentuação em codificações antigas,
colunas vazias e cabeçalhos duplicados. Este módulo torna esse ruído visível e
tratável em vez de deixá-lo quebrar a aplicação silenciosamente.

Postura de segurança:
    * Extensão e tamanho são validados **antes** de qualquer parsing.
    * Nenhum formato executável é aceito (nada de ``pickle``/``joblib``).
    * O parsing acontece sobre bytes já em memória e limitados, nunca sobre um
      caminho arbitrário do sistema de arquivos.
    * ``eval``/``exec`` não são usados em nenhum ponto.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from scr.core import config
from scr.core.schema import ColumnSpec, Dataset


class DataLoadError(Exception):
    """Erro de carregamento com mensagem já legível para o usuário final.

    A mensagem é escrita em pt-BR e sem jargão de stack trace, porque ela é
    exibida diretamente na interface para um público não técnico.
    """


# --------------------------------------------------------------------------- #
# Validação de entrada
# --------------------------------------------------------------------------- #


def validate_upload(filename: str, size_bytes: int, mime_type: Optional[str] = None) -> str:
    """Valida nome, tamanho e tipo declarado de um arquivo enviado.

    Args:
        filename: Nome original do arquivo enviado.
        size_bytes: Tamanho do conteúdo em bytes.
        mime_type: Tipo MIME declarado pelo navegador, quando disponível.

    Returns:
        A extensão normalizada em minúsculas (ex.: ``".csv"``).

    Raises:
        DataLoadError: Se a extensão não for permitida, o arquivo estiver vazio,
            exceder o limite de tamanho ou declarar um MIME inesperado.
    """
    if not filename:
        raise DataLoadError("O arquivo enviado não possui nome.")

    # Remove qualquer componente de caminho: defesa contra path traversal caso o
    # nome seja reaproveitado para gravação em disco no futuro.
    safe_name = os.path.basename(filename.replace("\\", "/"))
    extension = os.path.splitext(safe_name)[1].lower()

    if extension not in config.ALLOWED_EXTENSIONS:
        permitidas = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise DataLoadError(
            f"Formato '{extension or 'desconhecido'}' não suportado. "
            f"Envie um arquivo com uma destas extensões: {permitidas}."
        )

    if size_bytes <= 0:
        raise DataLoadError("O arquivo enviado está vazio.")

    if size_bytes > config.MAX_UPLOAD_BYTES:
        raise DataLoadError(
            f"O arquivo tem {size_bytes / 1024 / 1024:.1f} MB e excede o limite "
            f"de {config.MAX_UPLOAD_MB} MB."
        )

    if mime_type and mime_type not in config.ALLOWED_MIME_TYPES:
        raise DataLoadError(
            f"O tipo de conteúdo '{mime_type}' não é aceito para arquivos tabulares."
        )

    return extension


# --------------------------------------------------------------------------- #
# Detecção de codificação e separador
# --------------------------------------------------------------------------- #


def decode_text(data: bytes) -> Tuple[str, str]:
    """Decodifica bytes tentando as codificações mais comuns em dados brasileiros.

    Args:
        data: Conteúdo bruto do arquivo.

    Returns:
        Tupla ``(texto, codificação_utilizada)``.

    Raises:
        DataLoadError: Se nenhuma codificação candidata conseguir decodificar.
    """
    for encoding in config.CANDIDATE_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError):
            continue
    raise DataLoadError(
        "Não foi possível ler o texto do arquivo. Salve-o novamente em UTF-8 e tente outra vez."
    )


def detect_separator(sample: str) -> str:
    """Identifica automaticamente o separador de um arquivo texto.

    Estratégia em duas camadas:
        1. ``csv.Sniffer`` da biblioteca padrão, que costuma acertar arquivos
           bem formados.
        2. Heurística de consistência: para cada separador candidato, conta os
           campos por linha e prefere aquele que produz **mais de uma coluna** e
           a contagem **mais estável** entre as linhas. Estabilidade é o sinal
           mais confiável, porque um separador errado gera contagens erráticas.

    Args:
        sample: Trecho inicial do arquivo (algumas dezenas de KB bastam).

    Returns:
        O caractere separador identificado. Cai em ``","`` quando o arquivo tem
        uma única coluna e nenhum candidato se aplica.
    """
    lines = [line for line in sample.splitlines() if line.strip()][:50]
    if not lines:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample[:8192], delimiters="".join(config.CANDIDATE_SEPARATORS))
        sniffed = dialect.delimiter
        if sniffed in config.CANDIDATE_SEPARATORS and _column_count(lines[0], sniffed) > 1:
            return sniffed
    except csv.Error:
        pass  # Sniffer falha com frequência; a heurística abaixo assume.

    best_separator = ","
    best_score = (-1.0, 0.0)  # (colunas médias, estabilidade)

    for candidate in config.CANDIDATE_SEPARATORS:
        counts = [_column_count(line, candidate) for line in lines]
        if max(counts) <= 1:
            continue
        mean_count = sum(counts) / len(counts)
        # Estabilidade: proporção de linhas que compartilham a contagem modal.
        modal = max(set(counts), key=counts.count)
        stability = counts.count(modal) / len(counts)
        score = (stability, mean_count)
        if score > best_score:
            best_score = score
            best_separator = candidate

    return best_separator


def _column_count(line: str, separator: str) -> int:
    """Conta campos de uma linha respeitando aspas duplas.

    Args:
        line: Linha bruta do arquivo.
        separator: Separador candidato.

    Returns:
        Número de campos detectados na linha.
    """
    try:
        return len(next(csv.reader([line], delimiter=separator)))
    except csv.Error:
        return len(line.split(separator))


# --------------------------------------------------------------------------- #
# Leitores por formato
# --------------------------------------------------------------------------- #


def _read_delimited(data: bytes) -> Tuple[pd.DataFrame, str, str]:
    """Lê csv/txt detectando codificação e separador.

    Args:
        data: Conteúdo bruto do arquivo.

    Returns:
        Tupla ``(dataframe, separador, codificação)``.

    Raises:
        DataLoadError: Se o conteúdo não puder ser interpretado como tabela.
    """
    text, encoding = decode_text(data)
    if not text.strip():
        raise DataLoadError("O arquivo não contém dados.")

    separator = detect_separator(text[: config.SNIFF_SAMPLE_BYTES])

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=separator,
            engine="python",
            nrows=config.MAX_ROWS,
            skip_blank_lines=True,
            on_bad_lines="skip",  # linhas corrompidas são descartadas, não fatais
        )
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("O arquivo não contém dados tabulares.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(
            "Não foi possível interpretar o arquivo como tabela. "
            "Verifique se todas as linhas têm o mesmo número de colunas."
        ) from exc

    return frame, separator, encoding


_DECIMAL_COMMA_PATTERN = re.compile(r"^-?\d{1,3}(?:\.\d{3})*,\d+$|^-?\d+,\d+$")


def coerce_decimal_comma(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Converte colunas numéricas escritas no padrão brasileiro.

    Planilhas exportadas em pt-BR gravam ``1.234,56``. Lidas ingenuamente, essas
    colunas viram texto e o modelo passa a tratar preço como categoria — um erro
    silencioso que destrói o resultado sem gerar nenhuma exceção. A conversão só
    acontece quando a esmagadora maioria dos valores da coluna segue o padrão,
    para não estragar colunas de texto legítimas.

    Args:
        frame: Tabela recém-lida de um arquivo delimitado.

    Returns:
        Tupla ``(tabela_convertida, colunas_convertidas)``.
    """
    converted: List[str] = []

    for column in frame.columns:
        series = frame[column]
        # pandas 2 entrega texto como ``object``; pandas 3 usa ``StringDtype``.
        # Aceitar os dois mantém a aplicação funcionando nas duas gerações.
        if not (
            pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)
        ):
            continue

        values = series.dropna().astype(str).str.strip()
        if values.empty:
            continue

        matches = values.map(lambda item: bool(_DECIMAL_COMMA_PATTERN.match(item)))
        if matches.mean() < 0.9:
            continue

        numeric = pd.to_numeric(
            values.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        # Reconstrói a coluna preservando as posições originalmente ausentes.
        rebuilt = pd.Series(float("nan"), index=series.index, dtype="float64")
        rebuilt.loc[numeric.index] = numeric.to_numpy(dtype="float64")
        frame[column] = rebuilt
        converted.append(column)

    return frame, converted


def _read_excel(data: bytes) -> pd.DataFrame:
    """Lê a primeira planilha de um arquivo xlsx.

    Args:
        data: Conteúdo bruto do arquivo.

    Returns:
        DataFrame com o conteúdo da primeira aba.

    Raises:
        DataLoadError: Se o arquivo não for um xlsx válido.
    """
    try:
        return pd.read_excel(io.BytesIO(data), engine="openpyxl", nrows=config.MAX_ROWS)
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise DataLoadError(
            "Suporte a Excel indisponível. Instale a biblioteca 'openpyxl'."
        ) from exc
    except Exception as exc:
        raise DataLoadError(
            "Não foi possível abrir a planilha. Confirme que é um arquivo .xlsx válido."
        ) from exc


def _read_json(data: bytes) -> pd.DataFrame:
    """Lê JSON em formato de registros, achatando um nível de aninhamento.

    Aceita tanto uma lista de objetos quanto um objeto único ou um dicionário
    cujo valor seja a lista de registros. Estruturas aninhadas são achatadas com
    ``json_normalize``, o que é suficiente para os dados didáticos esperados.

    Args:
        data: Conteúdo bruto do arquivo.

    Returns:
        DataFrame com os registros.

    Raises:
        DataLoadError: Se o JSON for inválido ou não representar uma tabela.
    """
    text, _ = decode_text(data)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"JSON inválido: {exc.msg} (linha {exc.lineno}).") from exc

    records: object = payload
    if isinstance(payload, dict):
        list_values = [value for value in payload.values() if isinstance(value, list)]
        # Um dicionário com uma única lista dentro é o formato de envelope comum
        # em APIs ({"data": [...]}). Caso contrário, tratamos como registro único.
        records = list_values[0] if len(list_values) == 1 else [payload]

    if not isinstance(records, list) or not records:
        raise DataLoadError("O JSON não contém uma lista de registros para montar a tabela.")

    try:
        frame = pd.json_normalize(records, max_level=1)
    except Exception as exc:
        raise DataLoadError("Não foi possível converter o JSON em tabela.") from exc

    return frame.head(config.MAX_ROWS)


# --------------------------------------------------------------------------- #
# Normalização e API pública
# --------------------------------------------------------------------------- #


def _sanitize_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Normaliza nomes de colunas e remove linhas/colunas inúteis.

    Args:
        frame: Tabela recém-lida.

    Returns:
        Tupla ``(tabela_normalizada, avisos)``.

    Raises:
        DataLoadError: Se, após a limpeza, não restar tabela utilizável.
    """
    warnings: List[str] = []

    if frame.empty:
        raise DataLoadError("O arquivo foi lido, mas não contém nenhuma linha de dados.")

    if frame.shape[1] > config.MAX_COLUMNS:
        raise DataLoadError(
            f"A tabela tem {frame.shape[1]} colunas e excede o limite de {config.MAX_COLUMNS}."
        )

    # Nomes de coluna sempre string, sem espaços nas pontas e sem caracteres de
    # controle (que quebrariam a renderização e os rótulos de widgets).
    frame.columns = [
        re.sub(r"[\x00-\x1f\x7f]", "", str(column)).strip() or f"coluna_{index + 1}"
        for index, column in enumerate(frame.columns)
    ]

    # Colunas "Unnamed: N" são artefato de índices exportados pelo Excel/pandas.
    unnamed = [column for column in frame.columns if column.lower().startswith("unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
        warnings.append(f"{len(unnamed)} coluna(s) sem nome foram descartadas.")

    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        frame = frame.loc[:, ~frame.columns.duplicated()]
        warnings.append(
            f"Colunas com nome repetido foram mantidas apenas uma vez: {', '.join(sorted(set(duplicated)))}."
        )

    empty_columns = [column for column in frame.columns if frame[column].isna().all()]
    if empty_columns:
        frame = frame.drop(columns=empty_columns)
        warnings.append(f"{len(empty_columns)} coluna(s) totalmente vazias foram descartadas.")

    before = len(frame)
    frame = frame.dropna(how="all")
    if len(frame) < before:
        warnings.append(f"{before - len(frame)} linha(s) totalmente vazias foram descartadas.")

    if frame.empty or frame.shape[1] == 0:
        raise DataLoadError("Depois da limpeza não restaram dados utilizáveis no arquivo.")

    return frame.reset_index(drop=True), warnings


def load_tabular_file(
    filename: str,
    data: bytes,
    mime_type: Optional[str] = None,
) -> Dataset:
    """Carrega um arquivo tabular enviado pelo usuário.

    Ponto de entrada único da camada de ingestão. Valida, escolhe o leitor pelo
    formato, normaliza a tabela e devolve o contrato :class:`Dataset`.

    Args:
        filename: Nome original do arquivo (usado para inferir o formato).
        data: Conteúdo bruto em bytes.
        mime_type: Tipo MIME declarado pelo navegador, quando disponível.

    Returns:
        O :class:`Dataset` pronto para exibição e modelagem.

    Raises:
        DataLoadError: Para qualquer falha de validação ou parsing, sempre com
            mensagem em pt-BR adequada ao usuário final.
    """
    extension = validate_upload(filename, len(data), mime_type)

    separator: Optional[str] = None
    encoding: Optional[str] = None

    decimal_notes: List[str] = []

    if extension in {".csv", ".txt"}:
        frame, separator, encoding = _read_delimited(data)
        frame, converted = coerce_decimal_comma(frame)
        if converted:
            decimal_notes.append(
                "Coluna(s) convertidas do formato numérico brasileiro (vírgula decimal): "
                + ", ".join(converted)
                + "."
            )
    elif extension == ".xlsx":
        frame = _read_excel(data)
    elif extension == ".json":
        frame = _read_json(data)
    else:  # pragma: no cover - impossível após validate_upload
        raise DataLoadError(f"Formato '{extension}' não suportado.")

    frame, warnings = _sanitize_frame(frame)
    warnings = decimal_notes + warnings

    if len(frame) >= config.MAX_ROWS:
        warnings.append(
            f"Apenas as primeiras {config.MAX_ROWS:,} linhas foram carregadas.".replace(",", ".")
        )

    return Dataset(
        frame=frame,
        source_name=os.path.basename(filename.replace("\\", "/")),
        detected_separator=separator,
        detected_encoding=encoding,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# Perfil de colunas
# --------------------------------------------------------------------------- #


def is_numeric(series: pd.Series) -> bool:
    """Indica se a coluna deve ser tratada como numérica.

    Args:
        series: Coluna a inspecionar.

    Returns:
        ``True`` para dtypes numéricos (exceto booleano, tratado como categoria).
    """
    return bool(pd.api.types.is_numeric_dtype(series)) and not bool(
        pd.api.types.is_bool_dtype(series)
    )


def profile_columns(frame: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Resume tipo, ausências e cardinalidade de cada coluna.

    Usado na etapa 1 da interface para que o aprendiz veja a qualidade dos dados
    antes de escolher o alvo — decisão que depende justamente disso.

    Args:
        frame: Tabela carregada.

    Returns:
        Dicionário ``{coluna: {tipo, ausentes, percentual_ausente, distintos}}``.
    """
    profile: Dict[str, Dict[str, object]] = {}
    total = max(len(frame), 1)
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        profile[column] = {
            "tipo": "numérica" if is_numeric(series) else "categórica",
            "ausentes": missing,
            "percentual_ausente": round(100.0 * missing / total, 2),
            "distintos": int(series.nunique(dropna=True)),
        }
    return profile


def build_column_specs(frame: pd.DataFrame, features: Sequence[str]) -> List[ColumnSpec]:
    """Deriva as especificações de formulário a partir dos dados de treino.

    Os limites e categorias vêm do conjunto de treino, e não de suposições: é
    assim que a interface consegue avisar o aprendiz quando ele digita um valor
    fora da faixa que o modelo realmente viu — o problema de *extrapolação*.

    Args:
        frame: Tabela usada no treino.
        features: Colunas de entrada do modelo.

    Returns:
        Lista de :class:`ColumnSpec` na mesma ordem de ``features``.
    """
    specs: List[ColumnSpec] = []

    for column in features:
        series = frame[column]
        has_missing = bool(series.isna().any())

        if is_numeric(series):
            clean = series.dropna()
            default = float(clean.median()) if not clean.empty else 0.0
            specs.append(
                ColumnSpec(
                    name=column,
                    kind="numeric",
                    default=default,
                    minimum=float(clean.min()) if not clean.empty else None,
                    maximum=float(clean.max()) if not clean.empty else None,
                    has_missing=has_missing,
                )
            )
        else:
            clean = series.dropna().astype(str)
            categories = sorted(clean.unique().tolist())
            mode = clean.mode()
            default = str(mode.iloc[0]) if not mode.empty else (categories[0] if categories else "")
            specs.append(
                ColumnSpec(
                    name=column,
                    kind="categorical",
                    default=default,
                    categories=tuple(categories),
                    has_missing=has_missing,
                )
            )

    return specs
