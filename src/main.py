from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clustering import executar_agrupamento
from fuzzy_system import criar_sistema_fuzzy, prever_com_sistema
from preprocessing import preparar_dados


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "base_sintetica_media.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def imprimir_linha(tamanho: int = 70):
    print("=" * tamanho)


def imprimir_titulo(texto: str):
    print()
    imprimir_linha()
    print(texto)
    imprimir_linha()


def mostrar_resumo_preprocessamento(
    X_train_norm,
    X_test_norm,
    y_train,
    y_test,
    nomes_atributos,
):
    imprimir_titulo("ETAPA 1 - PRE-PROCESSAMENTO")
    print(f"Arquivo lido: {DATASET_PATH}")
    print(f"Quantidade de atributos: {len(nomes_atributos)}")
    print(f"Nomes dos atributos: {', '.join(nomes_atributos)}")
    print(f"Amostras de treino: {X_train_norm.shape[0]}")
    print(f"Amostras de teste: {X_test_norm.shape[0]}")
    print(f"Numero de classes em treino: {y_train.nunique()}")
    print(f"Numero de classes em teste: {y_test.nunique()}")


def mostrar_resultados_clustering(resultado_agrupamento):
    imprimir_titulo("ETAPA 2 - AGRUPAMENTO FUZZY")
    print(f"FPC (Fuzzy Partition Coefficient): {resultado_agrupamento['fpc']:.6f}")
    print(
        "Quantidade de regras geradas automaticamente: "
        f"{len(resultado_agrupamento['regras'])}"
    )

    print("\nMapeamento cluster -> classe:")
    for cluster, classe in resultado_agrupamento["mapeamento_clusters"].items():
        print(f"Cluster {cluster} -> Classe {classe}")

    print("\nCentros dos clusters:")
    for regra in resultado_agrupamento["regras"]:
        centro_formatado = ", ".join(f"{valor:.4f}" for valor in regra["centro"])
        print(f"Cluster {regra['cluster']}: [{centro_formatado}]")


def mostrar_regras_fuzzy(regras):
    imprimir_titulo("ETAPA 3 - REGRAS FUZZY GERADAS DOS CLUSTERS")

    for regra in regras:
        antecedentes_texto = []

        for nome_atributo, valor in regra["antecedentes"].items():
            antecedentes_texto.append(f"{nome_atributo} ~ {valor:.4f}")

        texto_regra = " E ".join(antecedentes_texto)

        print(
            f"Regra {regra['cluster'] + 1}: "
            f"SE {texto_regra} ENTAO classe = {regra['classe']}"
        )


def mostrar_resultados_sistema_fuzzy(y_test, y_pred):
    imprimir_titulo("ETAPA 4 - CLASSIFICACAO COM SISTEMA FUZZY")

    total = len(y_test)
    acertos = (y_test.to_numpy() == y_pred).sum()
    acuracia = acertos / total

    print(f"Total de amostras testadas: {total}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {total - acertos}")
    print(f"Acuracia: {acuracia:.4f}")

    print("\nPrimeiras previsoes:")
    comparacao = pd.DataFrame(
        {
            "classe_real": y_test.to_numpy()[:10],
            "classe_prevista": y_pred[:10],
        }
    )

    print(comparacao)


def salvar_resultados(resultado_agrupamento, y_test, y_pred):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    caminho_resumo = OUTPUT_DIR / "resumo_clusters.csv"
    caminho_regras = OUTPUT_DIR / "regras_fuzzy.json"
    caminho_previsoes = OUTPUT_DIR / "previsoes_teste.csv"

    resultado_agrupamento["resumo"].to_csv(caminho_resumo, index=False)

    with caminho_regras.open("w", encoding="utf-8") as arquivo:
        json.dump(resultado_agrupamento["regras"], arquivo, indent=4, ensure_ascii=False)

    previsoes_df = pd.DataFrame(
        {
            "classe_real": y_test.to_numpy(),
            "classe_prevista": y_pred,
        }
    )

    previsoes_df.to_csv(caminho_previsoes, index=False)

    imprimir_titulo("ARQUIVOS GERADOS")
    print(f"Resumo dos clusters salvo em: {caminho_resumo}")
    print(f"Regras fuzzy salvas em: {caminho_regras}")
    print(f"Previsoes do teste salvas em: {caminho_previsoes}")


def main():
    imprimir_titulo("SISTEMA DE AGRUPAMENTO + REGRAS FUZZY")

    X_train_norm, X_test_norm, y_train, y_test, scaler, nomes_atributos = preparar_dados(
        caminho_csv=str(DATASET_PATH)
    )

    mostrar_resumo_preprocessamento(
        X_train_norm=X_train_norm,
        X_test_norm=X_test_norm,
        y_train=y_train,
        y_test=y_test,
        nomes_atributos=nomes_atributos,
    )

    resultado_agrupamento = executar_agrupamento(
        X_train_norm=X_train_norm,
        y_train=y_train,
        nomes_atributos=nomes_atributos,
        n_clusters=4,
    )

    mostrar_resultados_clustering(resultado_agrupamento)
    mostrar_regras_fuzzy(resultado_agrupamento["regras"])

    sistema_fuzzy = criar_sistema_fuzzy(resultado_agrupamento)

    y_pred = prever_com_sistema(
        sistema_fuzzy=sistema_fuzzy,
        X=X_test_norm,
    )

    mostrar_resultados_sistema_fuzzy(
        y_test=y_test,
        y_pred=y_pred,
    )

    salvar_resultados(
        resultado_agrupamento=resultado_agrupamento,
        y_test=y_test,
        y_pred=y_pred,
    )

    imprimir_titulo("EXECUCAO FINALIZADA")
    print("O agrupamento foi concluido, as regras fuzzy foram geradas e o sistema fuzzy foi avaliado.")


if __name__ == "__main__":
    main()