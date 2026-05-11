from __future__ import annotations

import numpy as np
import pandas as pd
import skfuzzy as fuzz


def _para_numpy(X) -> np.ndarray:
    # Converte a entrada para numpy e valida o formato.
    X_array = np.asarray(X, dtype=float)

    if X_array.ndim != 2:
        raise ValueError("Os dados de entrada devem estar no formato 2D.")

    if not np.isfinite(X_array).all():
        raise ValueError(
            "Os dados de entrada possuem NaN ou infinito. "
            "Verifique o pre-processamento antes do agrupamento."
        )

    return X_array


def _para_tipo_python(valor):
    # Converte tipos numpy para tipos nativos do Python.
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def aplicar_fuzzy_c_means(
    # Aplica o Fuzzy C-Means nos dados de treino normalizados.
    X_train_norm,
    n_clusters: int = 4,
    m: float = 2.0,
    error: float = 0.005,
    maxiter: int = 1000,
    seed: int = 42,
):
    """
    Retorna:
    - centros dos clusters
    - matriz de pertinencia
    - rotulo dominante de cluster para cada amostra
    - FPC
    """
    X_array = _para_numpy(X_train_norm)

    dados_transpostos = X_array.T

    centros, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        data=dados_transpostos,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        seed=seed,
    )

    rotulos_clusters = np.argmax(u, axis=0)

    return centros, u, rotulos_clusters, fpc


def associar_clusters(u, rotulos_clusters, y_train):
    # Associa cada cluster à classe predominante com base nas amostras de treino.
    
    y_array = np.asarray(y_train)
    classes_unicas = np.unique(y_array)

    mapeamento = {}
    detalhes_clusters = []

    for cluster_idx in range(u.shape[0]):
        mascara_cluster = rotulos_clusters == cluster_idx
        indices_cluster = np.where(mascara_cluster)[0]

        contagem_classes = {}
        pesos_classes = {}

        for classe in classes_unicas:
            mascara_classe = y_array == classe
            contagem_classes[_para_tipo_python(classe)] = int(
                np.sum(y_array[indices_cluster] == classe)
            )
            pesos_classes[_para_tipo_python(classe)] = float(
                np.sum(u[cluster_idx, mascara_classe])
            )

        classe_predominante = max(pesos_classes, key=pesos_classes.get)
        mapeamento[cluster_idx] = classe_predominante

        detalhes_clusters.append(
            {
                "cluster": cluster_idx,
                "classe_predominante": classe_predominante,
                "total_amostras_dominantes": int(indices_cluster.size),
                "contagem_classes": contagem_classes,
                "pesos_classes": pesos_classes,
            }
        )

    return mapeamento, detalhes_clusters


def gerar_regras_clusters(centros, mapeamento_clusters, nomes_atributos):
    # Gera uma regra fuzzy para cada cluster encontrado.
    
    regras = []

    for cluster_idx, centro in enumerate(centros):
        antecedentes = {
            nome_atributo: float(valor_centro)
            for nome_atributo, valor_centro in zip(nomes_atributos, centro)
        }

        regras.append(
            {
                "cluster": cluster_idx,
                "classe": mapeamento_clusters[cluster_idx],
                "centro": centro.astype(float).tolist(),
                "antecedentes": antecedentes,
            }
        )

    return regras


def criar_resumo_clusters(regras, detalhes_clusters) -> pd.DataFrame:
    # Cria uma tabela simples para analise dos clusters.
    detalhes_por_cluster = {
        detalhe["cluster"]: detalhe for detalhe in detalhes_clusters
    }

    linhas = []

    for regra in regras:
        detalhe = detalhes_por_cluster[regra["cluster"]]
        linhas.append(
            {
                "cluster": regra["cluster"],
                "classe": regra["classe"],
                "total_amostras_dominantes": detalhe["total_amostras_dominantes"],
                "contagem_classes": detalhe["contagem_classes"],
                "pesos_classes": detalhe["pesos_classes"],
            }
        )

    return pd.DataFrame(linhas)


def executar_agrupamento(
    #Executa todo o fluxo de agrupamento e gera as regras fuzzy.
    X_train_norm,
    y_train,
    nomes_atributos,
    n_clusters: int = 4,
    m: float = 2.0,
    error: float = 0.005,
    maxiter: int = 1000,
    seed: int = 42,
):
    centros, u, rotulos_clusters, fpc = aplicar_fuzzy_c_means(
        X_train_norm=X_train_norm,
        n_clusters=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        seed=seed,
    )

    mapeamento_clusters, detalhes_clusters = associar_clusters(
        u=u,
        rotulos_clusters=rotulos_clusters,
        y_train=y_train,
    )

    regras = gerar_regras_clusters(
        centros=centros,
        mapeamento_clusters=mapeamento_clusters,
        nomes_atributos=nomes_atributos,
    )

    resumo = criar_resumo_clusters(
        regras=regras,
        detalhes_clusters=detalhes_clusters,
    )

    return {
        "centros": centros,
        "matriz_pertinencia": u,
        "rotulos_clusters": rotulos_clusters,
        "fpc": float(fpc),
        "mapeamento_clusters": mapeamento_clusters,
        "detalhes_clusters": detalhes_clusters,
        "regras": regras,
        "resumo": resumo,
    }
