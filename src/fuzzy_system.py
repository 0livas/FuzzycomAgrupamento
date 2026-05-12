from __future__ import annotations

import numpy as np


def _para_numpy(X) -> np.ndarray:
    X_array = np.asarray(X, dtype=float)

    if X_array.ndim == 1:
        X_array = X_array.reshape(1, -1)

    if X_array.ndim != 2:
        raise ValueError("Os dados de entrada devem estar no formato 2D.")

    if not np.isfinite(X_array).all():
        raise ValueError(
            "Os dados de entrada possuem NaN ou infinito. "
            "Verifique o pre-processamento antes da classificação fuzzy."
        )

    return X_array


def calcular_distancias(X, centros):
    X_array = _para_numpy(X)
    centros_array = np.asarray(centros, dtype=float)

    distancias = np.linalg.norm(
        X_array[:, np.newaxis, :] - centros_array[np.newaxis, :, :],
        axis=2
    )

    return distancias


def calcular_ativacoes(X, centros, sigma: float | None = None):
    # Calcula o grau de ativação de cada regra fuzzy.
    """
    Cada centro de cluster representa uma regra.
    Quanto menor a distância da amostra ao centro, maior a ativação da regra.
    """

    distancias = calcular_distancias(X, centros)

    if sigma is None:
        distancias_centros = np.linalg.norm(
            np.asarray(centros)[:, np.newaxis, :] - np.asarray(centros)[np.newaxis, :, :],
            axis=2
        )

        distancias_validas = distancias_centros[distancias_centros > 0]

        if distancias_validas.size == 0:
            sigma = 1.0
        else:
            sigma = np.mean(distancias_validas)

    ativacoes = np.exp(-(distancias ** 2) / (2 * sigma ** 2))

    soma_ativacoes = ativacoes.sum(axis=1, keepdims=True)

    soma_ativacoes[soma_ativacoes == 0] = 1

    ativacoes_normalizadas = ativacoes / soma_ativacoes

    return ativacoes_normalizadas


def prever_amostra(amostra, centros, mapeamento_clusters):
    # Classifica uma única amostra usando as regras fuzzy geradas pelos clusters.
    
    ativacoes = calcular_ativacoes(amostra, centros)

    cluster_mais_ativado = int(np.argmax(ativacoes[0]))

    classe_prevista = mapeamento_clusters[cluster_mais_ativado]

    return classe_prevista


def prever_conjunto(X, centros, mapeamento_clusters):
    # Classifica um conjunto de amostras.

    X_array = _para_numpy(X)

    ativacoes = calcular_ativacoes(X_array, centros)

    clusters_mais_ativados = np.argmax(ativacoes, axis=1)

    previsoes = [
        mapeamento_clusters[int(cluster)]
        for cluster in clusters_mais_ativados
    ]

    return np.asarray(previsoes)


def criar_sistema_fuzzy(resultado_agrupamento):
    # Cria uma estrutura simples para representar o sistema fuzzy.

    sistema = {
        "centros": resultado_agrupamento["centros"],
        "regras": resultado_agrupamento["regras"],
        "mapeamento_clusters": resultado_agrupamento["mapeamento_clusters"],
        "tipo": "Takagi-Sugeno ordem 0",
    }

    return sistema


def prever_com_sistema(sistema_fuzzy, X):
    # Usa o sistema fuzzy criado para prever as classes de um conjunto de amostras.

    return prever_conjunto(
        X=X,
        centros=sistema_fuzzy["centros"],
        mapeamento_clusters=sistema_fuzzy["mapeamento_clusters"]
    )