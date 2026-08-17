"""Testes da camada de ingestão de dados."""

from __future__ import annotations

import io
import json
from typing import Tuple

import pandas as pd
import pytest

from scr.core import config
from scr.core.data_loader import (
    DataLoadError,
    build_column_specs,
    detect_separator,
    load_tabular_file,
    profile_columns,
    validate_upload,
)


# --------------------------------------------------------------------------- #
# Validação de upload
# --------------------------------------------------------------------------- #


def test_validate_upload_aceita_extensoes_permitidas() -> None:
    assert validate_upload("dados.CSV", 1024) == ".csv"
    assert validate_upload("planilha.xlsx", 1024) == ".xlsx"


def test_validate_upload_rejeita_extensao_perigosa() -> None:
    with pytest.raises(DataLoadError, match="não suportado"):
        validate_upload("modelo.pkl", 1024)


def test_validate_upload_rejeita_arquivo_vazio() -> None:
    with pytest.raises(DataLoadError, match="vazio"):
        validate_upload("dados.csv", 0)


def test_validate_upload_rejeita_arquivo_grande_demais() -> None:
    with pytest.raises(DataLoadError, match="excede o limite"):
        validate_upload("dados.csv", config.MAX_UPLOAD_BYTES + 1)


def test_validate_upload_remove_componentes_de_caminho() -> None:
    # Defesa contra path traversal: só a extensão do nome-base importa.
    assert validate_upload("../../etc/passwd.csv", 10) == ".csv"


# --------------------------------------------------------------------------- #
# Detecção de separador
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("a,b,c\n1,2,3\n4,5,6\n", ","),
        ("a;b;c\n1;2;3\n4;5;6\n", ";"),
        ("a\tb\tc\n1\t2\t3\n4\t5\t6\n", "\t"),
        ("a|b|c\n1|2|3\n4|5|6\n", "|"),
    ],
)
def test_detect_separator_identifica_delimitadores_comuns(sample: str, expected: str) -> None:
    assert detect_separator(sample) == expected


def test_detect_separator_nao_confunde_virgula_decimal_com_separador() -> None:
    # Números com vírgula decimal em arquivo delimitado por ponto e vírgula:
    # o caso clássico de planilha exportada no padrão brasileiro.
    sample = "produto;preco\ncafe;10,50\nleite;7,25\narroz;22,90\n"
    assert detect_separator(sample) == ";"


def test_detect_separator_com_entrada_vazia_usa_padrao() -> None:
    assert detect_separator("") == ","


# --------------------------------------------------------------------------- #
# Leitura por formato
# --------------------------------------------------------------------------- #


def test_load_csv_com_virgula_e_com_ponto_e_virgula(csv_bytes: Tuple[bytes, bytes]) -> None:
    virgula, ponto_virgula = csv_bytes

    primeiro = load_tabular_file("dados.csv", virgula)
    segundo = load_tabular_file("dados.csv", ponto_virgula)

    assert primeiro.detected_separator == ","
    assert segundo.detected_separator == ";"
    assert list(primeiro.frame.columns) == list(segundo.frame.columns)
    assert primeiro.n_rows == segundo.n_rows == 30


def test_load_txt_tabulado_e_tratado_como_tabela() -> None:
    payload = "nome\tnota\nana\t9\nbruno\t7\ncarla\t8\n".encode("utf-8")
    dataset = load_tabular_file("notas.txt", payload)

    assert dataset.detected_separator == "\t"
    assert list(dataset.frame.columns) == ["nome", "nota"]
    assert dataset.n_rows == 3


def test_load_json_lista_de_registros() -> None:
    registros = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
    dataset = load_tabular_file("dados.json", json.dumps(registros).encode("utf-8"))

    assert dataset.n_rows == 3
    assert set(dataset.frame.columns) == {"a", "b"}


def test_load_json_com_envelope_de_dados() -> None:
    payload = json.dumps({"data": [{"a": 1}, {"a": 2}]}).encode("utf-8")
    dataset = load_tabular_file("dados.json", payload)

    assert dataset.n_rows == 2


def test_load_json_invalido_gera_mensagem_amigavel() -> None:
    with pytest.raises(DataLoadError, match="JSON inválido"):
        load_tabular_file("dados.json", b"{isto nao e json}")


def test_load_xlsx(classification_frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    classification_frame.head(20).to_excel(buffer, index=False)
    dataset = load_tabular_file("dados.xlsx", buffer.getvalue())

    assert dataset.n_rows == 20
    assert dataset.detected_separator is None


def test_csv_brasileiro_com_virgula_decimal_vira_coluna_numerica() -> None:
    payload = (
        "produto;preco;quantidade\n"
        "cafe;10,50;3\n"
        "leite;7,25;5\n"
        "arroz;1.234,90;2\n"
        "feijao;22,00;7\n"
    ).encode("utf-8")
    dataset = load_tabular_file("compras.csv", payload)

    assert dataset.detected_separator == ";"
    assert pd.api.types.is_numeric_dtype(dataset.frame["preco"])
    assert dataset.frame["preco"].iloc[2] == pytest.approx(1234.90)
    # A conversão é comunicada ao usuário, nunca feita em silêncio.
    assert any("vírgula decimal" in aviso for aviso in dataset.warnings)


def test_coluna_de_texto_legitima_nao_e_convertida() -> None:
    payload = "cidade,estado\nSao Paulo,SP\nRio,RJ\nCuritiba,PR\n".encode("utf-8")
    dataset = load_tabular_file("cidades.csv", payload)

    assert not pd.api.types.is_numeric_dtype(dataset.frame["cidade"])


def test_load_csv_latin1_com_acentuacao() -> None:
    payload = "cidade,populacao\nSão Paulo,12000000\nBrasília,3000000\n".encode("latin-1")
    dataset = load_tabular_file("cidades.csv", payload)

    assert "São Paulo" in dataset.frame["cidade"].tolist()


# --------------------------------------------------------------------------- #
# Normalização
# --------------------------------------------------------------------------- #


def test_colunas_vazias_e_duplicadas_sao_removidas_com_aviso() -> None:
    payload = "a,a,vazia\n1,2,\n3,4,\n5,6,\n".encode("utf-8")
    dataset = load_tabular_file("dados.csv", payload)

    assert dataset.warnings  # o descarte é comunicado, nunca silencioso
    assert "vazia" not in dataset.frame.columns


def test_arquivo_sem_linhas_de_dados_falha_com_mensagem() -> None:
    with pytest.raises(DataLoadError):
        load_tabular_file("dados.csv", b"a,b,c\n")


def test_arquivo_em_branco_falha() -> None:
    with pytest.raises(DataLoadError):
        load_tabular_file("dados.csv", b"   \n  \n")


# --------------------------------------------------------------------------- #
# Perfil e especificações de coluna
# --------------------------------------------------------------------------- #


def test_profile_columns_reporta_ausentes_e_cardinalidade() -> None:
    frame = pd.DataFrame({"n": [1.0, None, 3.0, 4.0], "c": ["a", "a", "b", None]})
    profile = profile_columns(frame)

    assert profile["n"]["tipo"] == "numérica"
    assert profile["n"]["ausentes"] == 1
    assert profile["n"]["percentual_ausente"] == 25.0
    assert profile["c"]["tipo"] == "categórica"
    assert profile["c"]["distintos"] == 2


def test_build_column_specs_deriva_limites_e_categorias(classification_frame: pd.DataFrame) -> None:
    specs = build_column_specs(classification_frame, ["idade", "regiao"])
    por_nome = {spec.name: spec for spec in specs}

    idade = por_nome["idade"]
    assert idade.kind == "numeric"
    assert idade.minimum is not None and idade.maximum is not None
    assert idade.minimum <= idade.default <= idade.maximum

    regiao = por_nome["regiao"]
    assert regiao.kind == "categorical"
    assert set(regiao.categories) == {"norte", "sul", "sudeste"}
    assert regiao.default in regiao.categories
