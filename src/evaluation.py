from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def avaliar_modelo(y_true, y_pred):
    """
    Calcula as principais métricas de avaliação do sistema fuzzy.
    """

    acuracia = accuracy_score(y_true, y_pred)
    matriz = confusion_matrix(y_true, y_pred)
    relatorio = classification_report(y_true, y_pred, zero_division=0)

    return {
        "acuracia": acuracia,
        "matriz_confusao": matriz,
        "relatorio_classificacao": relatorio,
    }


def imprimir_avaliacao(resultados_avaliacao):
    """
    Exibe os resultados da avaliação no terminal.
    """

    print("\n======================================================================")
    print("ETAPA 5 - AVALIACAO DO SISTEMA FUZZY")
    print("======================================================================")

    print(f"Acuracia: {resultados_avaliacao['acuracia']:.4f}")

    print("\nMatriz de confusao:")
    print(resultados_avaliacao["matriz_confusao"])

    print("\nRelatorio de classificacao:")
    print(resultados_avaliacao["relatorio_classificacao"])


def salvar_avaliacao(resultados_avaliacao, output_dir):
    """
    Salva a matriz de confusão e o relatório de classificação.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    caminho_matriz = output_dir / "matriz_confusao.csv"
    caminho_relatorio = output_dir / "relatorio_classificacao.txt"

    pd.DataFrame(resultados_avaliacao["matriz_confusao"]).to_csv(
        caminho_matriz,
        index=False
    )

    with caminho_relatorio.open("w", encoding="utf-8") as arquivo:
        arquivo.write(f"Acuracia: {resultados_avaliacao['acuracia']:.4f}\n\n")
        arquivo.write("Matriz de confusao:\n")
        arquivo.write(str(resultados_avaliacao["matriz_confusao"]))
        arquivo.write("\n\nRelatorio de classificacao:\n")
        arquivo.write(resultados_avaliacao["relatorio_classificacao"])

    return caminho_matriz, caminho_relatorio