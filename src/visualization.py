from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from preprocessing import preparar_dados
from clustering import executar_agrupamento


def _ensure_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix_from_csv(previsoes_csv: Path, save_path: Path):
    df = pd.read_csv(previsoes_csv)

    if 'classe_real' in df.columns and 'classe_prevista' in df.columns:
        y_true = df['classe_real']
        y_pred = df['classe_prevista']
    else:
        # tentativa genérica
        cols = df.columns.tolist()
        y_true = df[cols[0]]
        y_pred = df[cols[1]]

    from sklearn.metrics import confusion_matrix

    labels = np.unique(np.concatenate([y_true.unique(), y_pred.unique()]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_class_distribution_from_csv(previsoes_csv: Path, save_path: Path):
    df = pd.read_csv(previsoes_csv)

    if 'classe_real' in df.columns:
        y = df['classe_real']
    else:
        y = df[df.columns[0]]

    counts = y.value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette='muted')
    plt.xlabel('Classe')
    plt.ylabel('Contagem')
    plt.title('Distribuição das Classes')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pca_clusters(dataset_csv: str, n_clusters: int, save_path: Path):
    # Recria pré-processamento e agrupamento para obter rótulos por amostra.
    X_train_norm, X_test_norm, y_train, y_test, scaler, nomes_atributos = preparar_dados(caminho_csv=dataset_csv)

    resultado = executar_agrupamento(
        X_train_norm=X_train_norm,
        y_train=y_train,
        nomes_atributos=nomes_atributos,
        n_clusters=n_clusters,
    )

    X = np.asarray(X_train_norm)
    rotulos = np.asarray(resultado['rotulos_clusters'])
    centros = np.asarray(resultado['centros'])

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    centros2 = pca.transform(centros)

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette('tab10', n_colors=max(2, n_clusters))

    for cluster_idx in range(n_clusters):
        mask = rotulos == cluster_idx
        plt.scatter(X2[mask, 0], X2[mask, 1], s=10, color=palette[cluster_idx % len(palette)], label=f'Cluster {cluster_idx}')

    plt.scatter(centros2[:, 0], centros2[:, 1], s=120, facecolors='white', edgecolors='black', marker='D', linewidths=1.2, label='Centros')
    for i, (cx, cy) in enumerate(centros2):
        plt.text(cx + 0.02, cy + 0.02, str(i), fontsize=9, fontweight='bold')

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Visualização 2D dos Clusters (PCA)')
    plt.legend(markerscale=1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def gerar_graficos(output_dir: Path, dataset_csv: str, n_clusters: int = 4):
    output_dir = Path(output_dir)
    _ensure_output_dir(output_dir)

    previsoes_csv = output_dir / 'previsoes_teste.csv'

    cm_path = output_dir / 'matriz_confusao.png'
    dist_path = output_dir / 'distribuicao_classes.png'
    pca_path = output_dir / 'pca_clusters.png'

    if previsoes_csv.exists():
        plot_confusion_matrix_from_csv(previsoes_csv, cm_path)
        plot_class_distribution_from_csv(previsoes_csv, dist_path)

    plot_pca_clusters(dataset_csv=dataset_csv, n_clusters=n_clusters, save_path=pca_path)

    return cm_path, dist_path, pca_path
