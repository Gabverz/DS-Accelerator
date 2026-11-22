"""Funções reutilizáveis do projeto de previsão de inadimplência."""

import unicodedata

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

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
        freq["placeholder"] = _normalizar_serie(freq["valor"]).isin(placeholders_norm
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

def _texto_para_numero(
    serie: pd.Series, decimal: str = ".", milhar: str | None = None
) -> pd.Series:
    """Converte uma série de texto numérico em float.

    Remove tudo que não seja dígito, sinal de menos, vírgula ou ponto (o que
    elimina símbolos como "R$", espaços e textos como "não informado"), tira o
    separador de milhar, troca o separador decimal por ponto e converte.
    Valores que não puderem ser convertidos viram NaN.

    Args:
        serie: Série de texto, ex.: "R$ 1.234,56".
        decimal: Separador decimal usado no texto original.
        milhar: Separador de milhar usado no texto original, ou None se não houver.

    Returns:
        Série float64. Ex.: "R$ 1.234,56" com decimal="," e milhar="." -> 1234.56.
    """
    s = serie.astype("string")
    s = s.str.replace(r"[^\d\-,.]", "", regex=True)
    if milhar:
        s = s.str.replace(milhar, "", regex=False)
    if decimal != ".":
        s = s.str.replace(decimal, ".", regex=False)
    return pd.to_numeric(s, errors="coerce").astype("float64")


def converter_tipos(
    df: pd.DataFrame,
    datas: dict[str, str | list[str] | None] | None = None,
    numericas_texto: dict[str, dict] | None = None,
    inteiros: list[str] | tuple = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converte colunas para os tipos corretos e relata o que foi perdido.

    Não altera o DataFrame original. Valores que não puderem ser convertidos
    viram nulo, e o relatório informa quantos foram e exemplos deles, para
    que a conversão nunca descarte dados silenciosamente.

    Args:
        df: DataFrame de entrada.
        datas: Mapa coluna -> formato de data. O formato pode ser um texto
            (ex.: "%Y-%m-%d"), uma lista de textos tentados em ordem (cada
            formato preenche só o que os anteriores não converteram) ou None,
            caso em que o pandas tenta inferir.
        numericas_texto: Mapa coluna -> opções de _texto_para_numero (ex.:
            {"renda_anual": {"decimal": ",", "milhar": "."}}).
        inteiros: Colunas numéricas a converter para inteiro com suporte a nulos
            (Int64). Levanta ValueError se houver valores não inteiros.

    Returns:
        Tupla (df_convertido, relatorio). O relatório tem uma linha por coluna
        convertida, com dtype_antes, dtype_depois, nulos_antes, nulos_depois,
        falhas (valores não nulos que viraram nulo) e exemplos_falha.

    Raises:
        KeyError: Se alguma coluna pedida não existir no DataFrame.
        ValueError: Se uma coluna de `inteiros` tiver valores fracionários.

    Example:
        >>> cadastral_t, rel = converter_tipos(
        ...     cadastral,
        ...     datas={"data_nascimento": ["%Y-%m-%d", "%d/%m/%Y"]},
        ...     numericas_texto={"renda_anual": {"decimal": ",", "milhar": "."}},
        ...     inteiros=["qtd_membros_familia"],
        ... )
    """
    pedidas = [*(datas or {}), *(numericas_texto or {}), *inteiros]
    ausentes = [c for c in pedidas if c not in df.columns]
    if ausentes:
        raise KeyError(f"Colunas ausentes no DataFrame: {ausentes}")

    out = df.copy()
    linhas = []

    def registrar(col: str, depois: pd.Series) -> None:
        """Compara a coluna original com a convertida e guarda uma linha no relatório."""
        antes = df[col]
        virou_nulo = depois.isna() & antes.notna()
        linhas.append({
            "coluna": col,
            "dtype_antes": str(antes.dtype),
            "dtype_depois": str(depois.dtype),
            "nulos_antes": int(antes.isna().sum()),
            "nulos_depois": int(depois.isna().sum()),
            "falhas": int(virou_nulo.sum()),
            "exemplos_falha": list(antes[virou_nulo].unique()[:5]),
        })

    for col, formato in (datas or {}).items():
        formatos = [formato] if formato is None or isinstance(formato, str) else list(formato)
        convertida = pd.to_datetime(df[col], format=formatos[0], errors="coerce")
        for f in formatos[1:]:
            convertida = convertida.fillna(pd.to_datetime(df[col], format=f, errors="coerce"))
        out[col] = convertida
        registrar(col, out[col])

    for col, opcoes in (numericas_texto or {}).items():
        out[col] = _texto_para_numero(df[col], **opcoes)
        registrar(col, out[col])

    for col in inteiros:
        num = pd.to_numeric(df[col], errors="coerce")
        n_fracionarios = int((num.dropna() % 1 != 0).sum())
        if n_fracionarios:
            raise ValueError(f"{col}: {n_fracionarios} valores não inteiros")
        out[col] = num.astype("Int64")
        registrar(col, out[col])

    return out, pd.DataFrame(linhas)

def _normalizar_serie(serie: pd.Series) -> pd.Series:
    """Normaliza uma série de texto de forma vetorizada.

    Remove acentos, converte para minúsculas, tira espaços nas pontas e
    colapsa espaços repetidos. Nulos continuam nulos.

    Args:
        serie: Série de texto.

    Returns:
        Série de dtype string. Ex.: "  Não  Informado " -> "nao informado".
    """
    return (
        serie.astype("string")
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

def padronizar_categoricas(
    df: pd.DataFrame,
    colunas: list[str],
    placeholders: frozenset | set = PLACEHOLDERS_PADRAO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Padroniza colunas categóricas e relata o efeito.

    Aplica _normalizar a cada valor (minúsculas, sem acento, sem espaços
    sobrando) e converte placeholders de ausência (ex.: "não informado") em
    nulo. Não altera o DataFrame original.

    Args:
        df: DataFrame de entrada.
        colunas: Colunas de texto a padronizar.
        placeholders: Valores que representam ausência disfarçada; a
            comparação é feita após a normalização.

    Returns:
        Tupla (df_padronizado, relatorio). O relatório tem uma linha por
        coluna, com n_distintos_antes, n_distintos_depois, nulos_antes e
        nulos_depois.

    Raises:
        KeyError: Se alguma coluna não existir no DataFrame.
    """
    ausentes = [c for c in colunas if c not in df.columns]
    if ausentes:
        raise KeyError(f"Colunas ausentes no DataFrame: {ausentes}")

    placeholders_norm = {_normalizar(p) for p in placeholders}
    out = df.copy()
    linhas = []

    for col in colunas:
        antes = df[col]
        # Normaliza só os valores não nulos (o dtype string preserva os nulos).
        normal = _normalizar_serie(antes)
        normal = normal.mask(normal.isin(placeholders_norm))
        out[col] = normal
        linhas.append({
            "coluna": col,
            "n_distintos_antes": int(antes.nunique()),
            "n_distintos_depois": int(out[col].nunique()),
            "nulos_antes": int(antes.isna().sum()),
            "nulos_depois": int(out[col].isna().sum()),
        })

    return out, pd.DataFrame(linhas)

"""Funções para etapa de transformação e modelagem do projeto de previsão de inadimplência."""

PARAMS_BASELINE = dict(
    learning_rate=0.05, max_iter=200, max_depth=4, min_samples_leaf=200,
    l2_regularization=1.0, categorical_features="from_dtype",
    early_stopping=False, random_state=42,
)


def selecionar_features(df, excluir):
    """Devolve as colunas de df que não estão em `excluir`, validando que todas as exclusões existem."""
    ausentes = set(excluir) - set(df.columns)
    assert not ausentes, f"Colunas de exclusão inexistentes: {ausentes}"
    return [c for c in df.columns if c not in excluir]


def preparar_X(df, features):
    """Seleciona features e converte inteiros anuláveis (Int*) para float64 (NA -> NaN)."""
    X = df[features].copy()
    cols_int = X.select_dtypes(include=["Int8", "Int16", "Int32", "Int64"]).columns
    X[cols_int] = X[cols_int].astype("float64")
    return X


def ks_stat(y, p):
    """KS: maior distância entre as curvas acumuladas de maus (tpr) e bons (fpr)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metricas(y, p):
    """AUC e KS; devolve (NaN, NaN) se o recorte tiver uma só classe."""
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, p)), ks_stat(y, p)


def features_sem_variacao(X):
    """Colunas com menos de 2 valores distintos não nulos (quebram o binning do HistGradientBoosting)."""
    n_distintos = X.nunique(dropna=True)
    return n_distintos.index[n_distintos < 2].tolist()


def treinar_modelo(df_treino, features, col_target, params=PARAMS_BASELINE):
    """Treina o HistGradientBoosting removendo features constantes/vazias.

    Devolve (modelo, features_usadas, features_removidas) para rastreabilidade.
    """
    X = preparar_X(df_treino, features)
    removidas = features_sem_variacao(X)
    usadas = [c for c in features if c not in removidas]
    modelo = HistGradientBoostingClassifier(**params)
    modelo.fit(X[usadas], df_treino[col_target].astype("int8"))
    return modelo, usadas, removidas


def dividir_oot(df, col_data, prop_treino, embargo_dias):
    """Split temporal: teste = decisões após o quantil `prop_treino`;
    treino = decisões até (corte - embargo_dias). Devolve (treino, teste, data_corte)."""
    data_corte = df[col_data].quantile(prop_treino)
    limite_treino = data_corte - pd.Timedelta(days=embargo_dias)
    treino = df.loc[df[col_data] <= limite_treino]
    teste = df.loc[df[col_data] > data_corte]
    return treino, teste, data_corte


def avaliar_oot(treinos, teste, features, col_target, col_cliente, params=PARAMS_BASELINE):
    """Treina um modelo por cenário (dict nome -> treino) e avalia no teste idêntico.

    Recortes: completo, novos (cliente fora do treino) e recorrentes (cliente no treino),
    além do AUC/KS in-sample para medir overfitting.
    """
    y = teste[col_target].astype("int8").to_numpy()
    linhas = []
    for nome, tr in treinos.items():
        modelo, usadas, _ = treinar_modelo(tr, features, col_target, params)
        p = modelo.predict_proba(preparar_X(teste, usadas))[:, 1]

        recorrente = teste[col_cliente].isin(tr[col_cliente].unique()).to_numpy()
        recortes = {"completo": np.ones(len(teste), dtype=bool),
                    "novos": ~recorrente, "recorrentes": recorrente}
        for rec, m in recortes.items():
            auc, ks = metricas(y[m], p[m])
            linhas.append({"cenario": nome, "recorte": rec, "n": int(m.sum()),
                           "taxa_maus": float(y[m].mean()), "auc": auc, "ks": ks})

        y_tr = tr[col_target].astype("int8").to_numpy()
        auc_tr, ks_tr = metricas(y_tr, modelo.predict_proba(preparar_X(tr, usadas))[:, 1])
        linhas.append({"cenario": nome, "recorte": "treino (in-sample)", "n": len(tr),
                       "taxa_maus": float(y_tr.mean()), "auc": auc_tr, "ks": ks_tr})
    return pd.DataFrame(linhas)


def walk_forward(df, features, col_data, col_cliente, col_target,
                 n_folds, gap_dias, min_maus_treino, min_maus_val, params=PARAMS_BASELINE):
    """Walk-forward expansivo com gap e separação por cliente.

    Folds com poucos maus são registrados como ignorados (com motivo), não descartados.
    """
    limites = df[col_data].quantile(np.linspace(0, 1, n_folds + 2)).to_numpy()
    registros = []
    for k in range(1, n_folds + 1):
        ini, fim = pd.Timestamp(limites[k]), pd.Timestamp(limites[k + 1])
        val = df.loc[(df[col_data] > ini) & (df[col_data] <= fim)]
        tr = df.loc[df[col_data] <= ini - pd.Timedelta(days=gap_dias)]
        tr = tr.loc[~tr[col_cliente].isin(val[col_cliente].unique())]
        assert set(tr[col_cliente]).isdisjoint(set(val[col_cliente]))

        maus_tr = int(tr[col_target].astype("float").sum())
        maus_val = int(val[col_target].astype("float").sum())
        reg = {"fold": k, "val_inicio": ini.date(), "val_fim": fim.date(),
               "n_treino": len(tr), "maus_treino": maus_tr,
               "n_val": len(val), "maus_val": maus_val,
               "auc": np.nan, "ks": np.nan, "features_removidas": "", "status": "ok"}
        if maus_tr < min_maus_treino or maus_val < min_maus_val:
            reg["status"] = "ignorado: maus insuficientes"
        else:
            modelo, usadas, removidas = treinar_modelo(tr, features, col_target, params)
            pv = modelo.predict_proba(preparar_X(val, usadas))[:, 1]
            reg["auc"], reg["ks"] = metricas(val[col_target].astype("int8"), pv)
            reg["features_removidas"] = ", ".join(removidas)
        registros.append(reg)
    return pd.DataFrame(registros)

def criar_features_grupo_a(df, col_ref="data_ref", col_nasc="data_nascimento"):
    """
    Grupo A: features sem histórico, calculadas só com colunas do contrato atual e do cadastral.

    - idade_anos: (data_ref - data_nascimento) em anos; nula se qualquer data for nula.
    - razao_*: divisão protegida; resultado NaN quando o denominador é nulo ou <= 0.
    Retorna uma cópia de df com as colunas novas (não altera o original).
    """
  
    out = df.copy()

    # Idade em anos na data de referência (data_nascimento pode estar como texto)
    nasc = pd.to_datetime(out[col_nasc], errors="coerce")
    out["idade_anos"] = (out[col_ref] - nasc).dt.days / 365.25

    def razao(num, den):
        den_ok = den.notna() & (den > 0)
        return np.where(den_ok, num / den.where(den_ok), np.nan)

    out["razao_credito_parcela"] = razao(out["valor_credito"], out["valor_parcela"])
    out["razao_credito_bem"] = razao(out["valor_credito"], out["valor_bem"])
    out["razao_credito_renda"] = razao(out["valor_credito"], out["renda_anual"])
    out["razao_parcela_renda"] = razao(out["valor_parcela"], out["renda_anual"])

    return out