import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def carregar_dataset(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)
    return df


def separar_entradas_saida(df: pd.DataFrame, coluna_classe: str = "classe"):
    # Separa os atributos de entrada X e a saída y.
    if coluna_classe not in df.columns:
        raise ValueError(f"A coluna '{coluna_classe}' não foi encontrada no dataset.")

    X = df.drop(columns=[coluna_classe])
    y = df[coluna_classe]

    return X, y


def dividir_treino_teste(X, y, tamanho_teste: float = 0.2, random_state: int = 42):
    # Divide os dados em treino e teste (80% para treino e 20% para teste).
    # A estratificação mantém a proporção das classes.
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=tamanho_teste,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def normalizar_dados(X_train, X_test):
    # Normaliza os dados usando StandardScaler.

    scaler = StandardScaler()

    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_test_norm, scaler


def preparar_dados(caminho_csv: str, coluna_classe: str = "classe"):
    
    df = carregar_dataset(caminho_csv)

    X, y = separar_entradas_saida(df, coluna_classe)

    nomes_atributos = X.columns.tolist()

    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)

    X_train_norm, X_test_norm, scaler = normalizar_dados(X_train, X_test)

    return X_train_norm, X_test_norm, y_train, y_test, scaler, nomes_atributos