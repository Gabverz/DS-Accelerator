"""Funções reutilizáveis do projeto de previsão de inadimplência."""

import unicodedata

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

PLACEHOLDERS_PADRAO = frozenset({
    "", "?", "-", "n/a", "na", "nan", "none", "null", "xna", "unknown",
    "desconhecido", "nao informado", "sem informacao",
})


def _normalizar(valor: str) -> str:
    """Normaliza um texto para comparação de variantes de digitação.

    Remove acentos, converte para minúsculas, elimina espaços nas pontas e
    colapsa espaços repetidos no meio do texto.

    Args:
        valor: Texto original.

    Returns:
        Texto normalizado. Ex.: "  Não  Informado " -> "nao informado".
    """
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", valor) if not unicodedata.combining(c)
    )
    return " ".join(sem_acento.lower().split())


def _eh_categorica(serie: pd.Series) -> bool:
    """Indica se a série é de texto (object/str) ou categórica do pandas."""
    return (
        is_object_dtype(serie)
        or is_string_dtype(serie)
        or isinstance(serie.dtype, pd.CategoricalDtype)
    )


def auditar_categoricas(
    df: pd.DataFrame,
    colunas: list[str] | None = None,
    excluir: list[str] | tuple = (),
    limite_rara: float = 0.01,
    placeholders: frozenset | set = PLACEHOLDERS_PADRAO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Varre colunas categóricas de um DataFrame em busca de problemas de digitação.

    Para cada coluna, calcula contagem de valores (incluindo nulos), detecta
    valores distintos que colapsam na mesma forma normalizada (espaços,
    maiúsculas/minúsculas, acentos), identifica placeholders que representam
    ausência disfarçada (ex.: "XNA", "Unknown", "?") e sinaliza categorias raras.
    Nada é alterado no DataFrame original: a função apenas diagnostica.

    Args:
        df: DataFrame a ser auditado.
        colunas: Colunas a varrer. Se None, usa todas as colunas de texto ou
            categóricas do DataFrame.
        excluir: Colunas a ignorar. Útil para datas ou números lidos como texto
            (ex.: "data_nascimento", "renda_anual").
        limite_rara: Proporção abaixo da qual uma categoria é considerada rara.
            Padrão: 0.01 (1% das linhas).
        placeholders: Valores que indicam ausência disfarçada. A comparação é
            feita após a normalização, então "XNA" e " xna " são equivalentes.

    Returns:
        Uma tupla com três DataFrames:

        - resumo: uma linha por coluna, com n_nulos, pct_nulos, n_distintos,
          n_distintos_normalizado, n_grupos_com_variantes, placeholders
          (valores encontrados), n_raras e menor_pct.
        - frequencias: uma linha por (coluna, valor), com n, pct e as flags
          rara e placeholder. Os nulos aparecem como valor NaN.
        - variantes: uma linha por grupo de valores que diferem só pela
          digitação, com a lista de variantes e o total de linhas afetadas.
          Vazio se não houver problemas.

    Example:
        >>> resumo, freq, variantes = auditar_categoricas(
        ...     cadastral, excluir=["data_nascimento", "renda_anual"]
        ... )
        >>> variantes  # vazio se não houver variantes de digitação
    """
    if colunas is None:
        colunas = [c for c in df.columns if c not in excluir and _eh_categorica(df[c])]
    placeholders_norm = {_normalizar(p) for p in placeholders}

    linhas_resumo, tabelas_freq, linhas_variantes = [], [], []

    for col in colunas:
        serie = df[col]
        n_total = len(serie)
        contagem = serie.value_counts(dropna=True)
        chave = pd.Series(
            [_normalizar(str(v)) for v in contagem.index],
            index=contagem.index,
            dtype="object",
        )

        # Valores brutos distintos que colapsam na mesma forma normalizada
        tamanho_grupo = chave.value_counts()
        chaves_duplicadas = tamanho_grupo[tamanho_grupo > 1].index
        for k in chaves_duplicadas:
            brutos = contagem[chave == k]
            linhas_variantes.append({
                "coluna": col,
                "valor_normalizado": k,
                "variantes": list(brutos.index),
                "n_linhas": int(brutos.sum()),
            })

        # Tabela de frequências, com nulos
        freq = serie.value_counts(dropna=False).rename_axis("valor").reset_index(name="n")
        freq["pct"] = freq["n"] / n_total
        nao_nulo = freq["valor"].notna()
        freq["rara"] = nao_nulo & (freq["pct"] < limite_rara)
        freq["placeholder"] = freq["valor"].map(
            lambda v: pd.notna(v) and _normalizar(str(v)) in placeholders_norm
        )
        freq.insert(0, "coluna", col)
        tabelas_freq.append(freq)

        n_nulos = int(serie.isna().sum())
        linhas_resumo.append({
            "coluna": col,
            "n_nulos": n_nulos,
            "pct_nulos": n_nulos / n_total,
            "n_distintos": int(contagem.size),
            "n_distintos_normalizado": int(chave.nunique()),
            "n_grupos_com_variantes": int(len(chaves_duplicadas)),
            "placeholders": ", ".join(map(str, freq.loc[freq["placeholder"], "valor"])),
            "n_raras": int(freq["rara"].sum()),
            "menor_pct": float(freq.loc[nao_nulo, "pct"].min()) if nao_nulo.any() else float("nan"),
        })

    resumo = pd.DataFrame(linhas_resumo)
    frequencias = pd.concat(tabelas_freq, ignore_index=True) if tabelas_freq else pd.DataFrame()
    variantes = pd.DataFrame(
        linhas_variantes, columns=["coluna", "valor_normalizado", "variantes", "n_linhas"]
    )
    return resumo, frequencias, variantes