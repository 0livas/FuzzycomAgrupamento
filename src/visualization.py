from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from preprocessing import preparar_dados
from clustering import executar_agrupamento

# Configurações visuais p/os gráficos

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.05
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

def _ensure_output_dir(output_dir: Path) -> None:
    """Garante que o diretório de saída exista."""
    output_dir.mkdir(parents=True, exist_ok=True)


def _salvar_figura(save_path: Path) -> None:
    """Salva a figura atual em alta resolução e fecha o gráfico."""
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _obter_colunas_previsao(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Obtém as colunas de classe real e classe prevista.

    Prioriza os nomes esperados:
    - classe_real
    - classe_prevista

    Caso eles não existam, usa as duas primeiras colunas do CSV.
    """
    if "classe_real" in df.columns and "classe_prevista" in df.columns:
        y_true = df["classe_real"]
        y_pred = df["classe_prevista"]
    else:
        if len(df.columns) < 2:
            raise ValueError(
                "O arquivo de previsões precisa ter pelo menos duas colunas: "
                "classe real e classe prevista."
            )

        cols = df.columns.tolist()
        y_true = df[cols[0]]
        y_pred = df[cols[1]]

    return y_true, y_pred

def _ordenar_labels(y_true: pd.Series, y_pred: pd.Series) -> list:
    """
    Cria uma lista ordenada de labels presentes nas classes reais e previstas.

    A ordenação por string evita erro quando há mistura de tipos,
    por exemplo: classes numéricas e classes textuais.
    """
    labels = set(y_true.dropna().unique()).union(set(y_pred.dropna().unique()))
    return sorted(labels, key=lambda x: str(x))

def plot_confusion_matrix_from_csv(previsoes_csv: Path, save_path: Path) -> None:
    """
    Gera uma matriz de confusão a partir de um CSV de previsões.

    O gráfico exibe:
    - quantidade absoluta;
    - percentual por linha;
    - rótulos de classes nos eixos.
    """
    df = pd.read_csv(previsoes_csv)

    y_true, y_pred = _obter_colunas_previsao(df)
    labels = _ordenar_labels(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )

    annot = np.empty_like(cm).astype(str)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1%})"

    plt.figure(figsize=(8, 6.5))

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Quantidade"}
    )

    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.title("Matriz de Confusão — Classes Reais vs. Previstas", pad=12)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    _salvar_figura(save_path)

def plot_class_distribution_from_csv(previsoes_csv: Path, save_path: Path) -> None:
    """
    Gera um gráfico de barras com a distribuição das classes reais.

    O gráfico exibe:
    - quantidade por classe;
    - percentual por classe;
    - barras com rótulos.
    """
    df = pd.read_csv(previsoes_csv)

    if "classe_real" in df.columns:
        y = df["classe_real"]
    else:
        y = df[df.columns[0]]

    counts = y.value_counts().sort_index()
    total = counts.sum()

    plt.figure(figsize=(7.5, 4.8))

    ax = sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        hue=counts.index.astype(str),
        palette="muted",
        legend=False
    )

    for i, value in enumerate(counts.values):
        percentual = value / total if total > 0 else 0

        ax.text(
            i,
            value,
            f"{value}\n({percentual:.1%})",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xlabel("Classe")
    plt.ylabel("Quantidade de amostras")
    plt.title("Distribuição das Classes no Conjunto de Teste", pad=12)

    if len(counts) > 0:
        plt.ylim(0, counts.max() * 1.18)

    plt.xticks(rotation=0)

    _salvar_figura(save_path)

def plot_pca_clusters(dataset_csv: str, n_clusters: int, save_path: Path) -> None:
    """
    Gera uma visualização 2D dos clusters usando PCA.

    O gráfico exibe:
    - amostras projetadas nos dois primeiros componentes principais;
    - cores por cluster;
    - centroides dos clusters;
    - percentual de variância explicada em cada eixo.
    """
    X_train_norm, X_test_norm, y_train, y_test, scaler, nomes_atributos = preparar_dados(
        caminho_csv=dataset_csv
    )

    resultado = executar_agrupamento(
        X_train_norm=X_train_norm,
        y_train=y_train,
        nomes_atributos=nomes_atributos,
        n_clusters=n_clusters,
    )

    X = np.asarray(X_train_norm)
    rotulos = np.asarray(resultado["rotulos_clusters"])
    centros = np.asarray(resultado["centros"])

    if X.shape[0] < 2:
        raise ValueError("É necessário ter pelo menos duas amostras para aplicar PCA.")

    if X.shape[1] < 2:
        raise ValueError("É necessário ter pelo menos duas variáveis para visualizar PCA em 2D.")

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    centros2 = pca.transform(centros)

    var1, var2 = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(8.5, 6.5))

    palette = sns.color_palette("tab10", n_colors=max(2, n_clusters))

    for cluster_idx in range(n_clusters):
        mask = rotulos == cluster_idx

        plt.scatter(
            X2[mask, 0],
            X2[mask, 1],
            s=24,
            alpha=0.75,
            color=palette[cluster_idx % len(palette)],
            label=f"Cluster {cluster_idx}"
        )

    plt.scatter(
        centros2[:, 0],
        centros2[:, 1],
        s=170,
        facecolors="white",
        edgecolors="black",
        marker="D",
        linewidths=1.5,
        label="Centroides"
    )

    for i, (cx, cy) in enumerate(centros2):
        plt.text(
            cx + 0.02,
            cy + 0.02,
            str(i),
            fontsize=9,
            fontweight="bold",
            color="black"
        )

    plt.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.axvline(0, color="gray", linewidth=0.8, alpha=0.4)

    plt.xlabel(f"PCA 1 ({var1:.1f}% da variância)")
    plt.ylabel(f"PCA 2 ({var2:.1f}% da variância)")
    plt.title("Visualização dos Clusters em 2D via PCA", pad=12)

    plt.legend(
        title="Agrupamentos",
        loc="best",
        frameon=True
    )

    _salvar_figura(save_path)


def gerar_graficos(output_dir: Path, dataset_csv: str, n_clusters: int = 4):
    """
    Gera os gráficos principais do projeto.

    Saídas geradas:
    - matriz_confusao.png
    - distribuicao_classes.png
    - pca_clusters.png

    A matriz de confusão e a distribuição das classes dependem do arquivo:
    - previsoes_teste.csv

    O PCA dos clusters é gerado a partir do dataset original.
    """
    output_dir = Path(output_dir)
    _ensure_output_dir(output_dir)

    previsoes_csv = output_dir / "previsoes_teste.csv"

    cm_path = output_dir / "matriz_confusao.png"
    dist_path = output_dir / "distribuicao_classes.png"
    pca_path = output_dir / "pca_clusters.png"

    if previsoes_csv.exists():
        plot_confusion_matrix_from_csv(previsoes_csv, cm_path)
        plot_class_distribution_from_csv(previsoes_csv, dist_path)
    else:
        print(
            f"Aviso: arquivo '{previsoes_csv}' não encontrado. "
            "Matriz de confusão e distribuição das classes não foram geradas."
        )

    plot_pca_clusters(
        dataset_csv=dataset_csv,
        n_clusters=n_clusters,
        save_path=pca_path
    )

    return cm_path, dist_path, pca_path