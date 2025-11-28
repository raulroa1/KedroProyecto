import logging
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
logger = logging.getLogger(__name__)

def preparar_datos_clasificacion(df: pd.DataFrame, max_samples: int = 10000):
    df = df.copy()
    print("Iniciando preparacion de datos para clasificacion...")
    
    # Muestreo si el dataset es muy grande
    if len(df) > max_samples:
        print(f"Dataset grande ({len(df)} filas). Muestreando a {max_samples} filas...")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Dataset muestreado: {len(df)} filas")

    # ======================
    # LIMPIEZA BASICA
    # ======================
    df.columns = df.columns.str.strip()
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df = df.dropna(subset=["FECHA"])
    print(f"Fechas validas: {df.shape[0]} filas restantes")

    # ======================
    # VARIABLES BASE
    # ======================
    df["MES"] = df["FECHA"].dt.month
    df["PRODUCTO_ID"] = df["PRODUCTO"].astype("category").cat.codes
    df["COMUNA_ID"] = df["COMUNA"].astype("category").cat.codes

    # ======================
    # VARIABLES AGRUPADAS
    # ======================
    df = df.sort_values(["PRODUCTO_ID", "COMUNA_ID", "FECHA"])
    grp = df.groupby(["PRODUCTO_ID", "COMUNA_ID"])["CANTIDAD"]
    df["VENTA_MES_ANTERIOR"] = grp.shift(1).astype("float32")
    df["PROM_3_MESES"] = grp.shift(1).rolling(3, min_periods=1).mean().astype("float32")
    df["PROM_6_MESES"] = grp.shift(1).rolling(6, min_periods=1).mean().astype("float32")
    mask = df["VENTA_MES_ANTERIOR"] != 0
    df["DELTA_MES"] = 0.0
    df.loc[mask, "DELTA_MES"] = (
        (df.loc[mask, "CANTIDAD"] - df.loc[mask, "VENTA_MES_ANTERIOR"]) 
        / df.loc[mask, "VENTA_MES_ANTERIOR"]
    ).astype("float32")

    # ======================
    # CLASIFICACION DE VENTAS
    # ======================
    bins = [0, 10, 50, df["CANTIDAD"].max()]
    labels = [0, 1, 2]
    df["VENTA_CLASE"] = pd.cut(df["CANTIDAD"], bins=bins, labels=labels, include_lowest=True).astype("category")

    # ======================
    # LIMPIEZA FINAL DE NULOS
    # ======================
    antes = len(df)
    df = df.dropna(subset=["VENTA_MES_ANTERIOR", "VENTA_CLASE"])
    despues = len(df)
    print(f"Filas eliminadas por nulos: {antes - despues}")
    print(f"Tamano despues de limpieza: {df.shape}")

    # ======================
    # CONVERSION A ENTERO
    # ======================
    columnas_a_convertir = [
        "MES", "PRODUCTO_ID", "COMUNA_ID",
        "VENTA_MES_ANTERIOR", "PROM_3_MESES",
        "PROM_6_MESES", "DELTA_MES"
    ]

    faltantes = [c for c in columnas_a_convertir if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")

    for col in columnas_a_convertir:
        df[col] = df[col].fillna(0).round().astype("int32")

    # ======================
    # VARIABLES X E Y
    # ======================
    X = df[columnas_a_convertir].copy().reset_index(drop=True)
    y = df["VENTA_CLASE"].cat.codes.astype("int8").reset_index(drop=True)
    
    # Convertir y a DataFrame para que pueda guardarse como Parquet
    y = pd.DataFrame(y, columns=["VENTA_CLASE"])

    # ======================
    # INFO FINAL
    # ======================
    print("\nColumnas convertidas a entero:")
    print(df[columnas_a_convertir].dtypes)
    print("\nResumen final:")
    print(f"Filas totales: {df.shape[0]}")
    print(f"Columnas totales: {df.shape[1]}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Nulos restantes:\n{df.isna().sum()}")

    return X, y

def preparar_datos_regresion(df: pd.DataFrame, max_samples: int = 10000):
    df = df.copy()
    
    if len(df) > max_samples:
        print(f"Dataset grande ({len(df)} filas). Muestreando a {max_samples} filas...")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Dataset muestreado: {len(df)} filas")
    
    df.columns = df.columns.str.strip()
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df = df.dropna(subset=["FECHA"])
    print(f"Fechas validas: {df.shape[0]} filas restantes")
    
    df["MES"] = df["FECHA"].dt.month
    df["PRODUCTO_ID"] = df["PRODUCTO"].astype("category").cat.codes
    df["COMUNA_ID"] = df["COMUNA"].astype("category").cat.codes

    df.sort_values(['PRODUCTO_ID','COMUNA_ID','MES'], inplace=True)
    grp = df.groupby(['PRODUCTO_ID','COMUNA_ID'])['CANTIDAD']
    df['VENTA_MES_ANTERIOR'] = grp.shift(1).astype("float32")
    df['PROM_3_MESES'] = grp.shift(1).rolling(3,min_periods=1).mean().astype("float32")
    df['PROM_6_MESES'] = grp.shift(1).rolling(6,min_periods=1).mean().astype("float32")
    mask = df['VENTA_MES_ANTERIOR'] != 0
    df['DELTA_MES'] = 0.0
    df.loc[mask, 'DELTA_MES'] = (
        (df.loc[mask, 'CANTIDAD'] - df.loc[mask, 'VENTA_MES_ANTERIOR']) 
        / df.loc[mask, 'VENTA_MES_ANTERIOR']
    ).astype("float32")

    df = df.dropna(subset=['VENTA_MES_ANTERIOR'])
    X = df[['MES','PRODUCTO_ID','COMUNA_ID','VENTA_MES_ANTERIOR','PROM_3_MESES','PROM_6_MESES','DELTA_MES']]
    y = df['CANTIDAD']
    
    # Convertir y a DataFrame para que pueda guardarse como Parquet
    y = pd.DataFrame(y, columns=["CANTIDAD"])

    print("\nColumnas convertidas a entero:")
    print(df.dtypes)
    print("\nResumen final:")
    print(f"Filas totales: {df.shape[0]}")
    print(f"Columnas totales: {df.shape[1]}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Nulos restantes:\n{df.isna().sum()}")

    return X, y

def prepare_clustering_data(data: pd.DataFrame, feature_columns: list = None, max_samples: int = 10000) -> pd.DataFrame:
    """
    Prepara datos para clustering seleccionando features numéricas relevantes.
    
    Args:
        data: DataFrame con datos de ventas preprocesadas
        feature_columns: Lista de columnas a usar (None = automático)
        max_samples: Máximo de muestras a usar (para optimizar tiempo)
        
    Returns:
        DataFrame con solo features numéricas seleccionadas
    """
    logger.info("Preparando datos para clustering...")
    logger.info(f"Datos de entrada: {len(data)} filas, {data.shape[1]} columnas")
    
    # Muestreo si el dataset es muy grande
    if len(data) > max_samples:
        logger.warning(f"Dataset grande ({len(data)} filas). Muestreando a {max_samples} filas...")
        data = data.sample(n=max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"Dataset muestreado: {len(data)} filas")
    
    # Selección de features
    if feature_columns is None:
        # Seleccionar automáticamente columnas numéricas
        # Excluir IDs categóricos que no aportan información para clustering
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        # Excluir columnas que son IDs categóricos (si existen), pero mantener las importantes
        feature_columns = [col for col in numeric_cols if not col.endswith('_ID') or col in ['MES', 'CANTIDAD', 'VENTA_MES_ANTERIOR', 'AUMENTA']]
        # Simplificar: usar todas las numéricas por ahora
        feature_columns = numeric_cols
        # Si no hay suficientes, usar todas las numéricas
        if len(feature_columns) < 2:
            feature_columns = numeric_cols
        logger.info(f"Features seleccionadas automáticamente: {feature_columns}")
    else:
        logger.info(f"Usando features especificadas: {feature_columns}")
    
    # Seleccionar y validar features
    X = data[feature_columns].select_dtypes(include=[np.number]).copy()
    
    # Eliminar columnas con varianza cero (no aportan información)
    X = X.loc[:, X.var() != 0]
    
    # Validar que haya suficientes features
    if len(X.columns) < 1:
        raise ValueError("No hay suficientes features numéricas para clustering")
    
    logger.info(f"Features finales seleccionadas: {len(X.columns)} columnas: {list(X.columns)}")
    logger.info(f"Muestras: {len(X)} filas")
    logger.info(f"Shape final: {X.shape}")
    
    return X

# ==========================
# ===== Clasificacion ======
# ==========================
def pre_proceso_clf(X,y):
    # Asegurar que X e y tengan el mismo tamaño
    min_size = min(len(X), len(y))
    X = X.iloc[:min_size].reset_index(drop=True)
    y = y.iloc[:min_size].reset_index(drop=True)
    print(f"Tamanos ajustados: X={len(X)}, y={len(y)}")
    return X,y

def dividir_datos_clf(X, y):
    # Convertir y a Series si es DataFrame para train_test_split
    if isinstance(y, pd.DataFrame):
        y_series = y.iloc[:, 0] if len(y.columns) > 0 else y.squeeze()
    else:
        y_series = y
    
    print("Entradas dividir_datos_clf:", len(X), len(y_series))
    X_train, X_test, y_train, y_test = train_test_split(X, y_series, test_size=0.2, random_state=42)
    
    # Convertir y_train e y_test a DataFrame para guardar como Parquet
    y_train = pd.DataFrame(y_train, columns=["VENTA_CLASE"])
    y_test = pd.DataFrame(y_test, columns=["VENTA_CLASE"])
    
    print("Salidas dividir_datos_clf:", len(X_train), len(y_train))
    return X_train, X_test, y_train, y_test

# ==========================
# === Entrenamiento modelos ==
# ==========================
def entrenar_modelos_clasificacion_cv(X_train_clf, y_train_clf, output_path="models_clf"):
    # Convertir y_train_clf a Series si es DataFrame
    if isinstance(y_train_clf, pd.DataFrame):
        y_train_series = y_train_clf.iloc[:, 0] if len(y_train_clf.columns) > 0 else y_train_clf.squeeze()
    else:
        y_train_series = y_train_clf
    
    modelos = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {'C':[0.01,0.1,1,10]}),
        "RandomForest": (RandomForestClassifier(random_state=42), {'n_estimators':[50,100,200]}),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {'n_estimators':[50,100]}),
        "SVC": (SVC(probability=True), {'C':[0.1,1,10]}),
        "KNN": (KNeighborsClassifier(), {'n_neighbors':[3,5,7]})
    }
    
    resultados_clf = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for nombre, (modelo, params) in modelos.items():
        print(f"\nEntrenando {nombre}...")
        gs = GridSearchCV(modelo, params, cv=cv, n_jobs=-1)
        gs.fit(X_train_clf, y_train_series)
        resultados_clf[nombre] = {
            "modelo": gs.best_estimator_,
            "mejor_params": gs.best_params_,
            "score_cv": gs.best_score_
        }
        # Los modelos se guardan en resultados_clf a través del catálogo de Kedro
        # No es necesario guardarlos individualmente aquí
    
    return resultados_clf


# ==========================
# === Evaluacion modelos ===
# ==========================
def evaluar_modelos_clasificacion(resultados, X_test, y_test):
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Convertir y_test a Series si es DataFrame
    if isinstance(y_test, pd.DataFrame):
        y_test_series = y_test.iloc[:, 0] if len(y_test.columns) > 0 else y_test.squeeze()
    else:
        y_test_series = y_test

    metricas = {}
    for nombre, res in resultados.items():
        y_pred = res['modelo'].predict(X_test)
        metricas[nombre] = {
            "Accuracy": accuracy_score(y_test_series, y_pred),
            "Precision": precision_score(y_test_series, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test_series, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_test_series, y_pred, average='weighted', zero_division=0)
        }
    return pd.DataFrame(metricas).T


# ==========================
# ===== Regresion ==========
# ==========================
def pre_proceso_rg(X, y):
    # Asegurar que X e y tengan el mismo tamaño
    min_size = min(len(X), len(y))
    X = X.iloc[:min_size].reset_index(drop=True)
    y = y.iloc[:min_size].reset_index(drop=True)
    print(f"Tamanos ajustados: X={len(X)}, y={len(y)}")
    return X, y

def dividir_datos_reg(X, y, test_size=0.2, random_state=42):
    # Convertir y a Series si es DataFrame para train_test_split
    if isinstance(y, pd.DataFrame):
        y_series = y.iloc[:, 0] if len(y.columns) > 0 else y.squeeze()
    else:
        y_series = y
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_series, test_size=test_size, random_state=random_state)
    
    # Convertir y_train e y_test a DataFrame para guardar como Parquet
    y_train = pd.DataFrame(y_train, columns=["CANTIDAD"])
    y_test = pd.DataFrame(y_test, columns=["CANTIDAD"])
    
    return X_train, X_test, y_train, y_test

def entrenar_modelos_regresion_cv(X_train, y_train, output_path="models_reg"):
    # Convertir y_train a Series si es DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_train_series = y_train.iloc[:, 0] if len(y_train.columns) > 0 else y_train.squeeze()
    else:
        y_train_series = y_train
    
    modelos = {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {'alpha':[0.1,1.0,10.0]}),
        "Lasso": (Lasso(), {'alpha':[0.01,0.1,1.0]}),
        "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=-1), {'n_estimators':[50,100]}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42), {'n_estimators':[50,100]}),
    }
    
    resultados_rg = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for nombre, (modelo, params) in modelos.items():
        if params:
            gs = GridSearchCV(modelo, params, cv=cv, n_jobs=-1)
            gs.fit(X_train, y_train_series)
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            score_cv = gs.best_score_
        else:
            modelo.fit(X_train, y_train_series)
            best_model = modelo
            best_params = {}
            score_cv = None
        
        resultados_rg[nombre] = {
            "modelo": best_model,
            "mejor_params": best_params,
            "score_cv": score_cv
        }
        # Los modelos se guardan en resultados_rg a través del catálogo de Kedro
        # No es necesario guardarlos individualmente aquí
        print(f"Modelo entrenado: {nombre}")
    
    return resultados_rg

def evaluar_modelos_regresion(resultados, X_test, y_test):
    # Convertir y_test a Series si es DataFrame
    if isinstance(y_test, pd.DataFrame):
        y_test_series = y_test.iloc[:, 0] if len(y_test.columns) > 0 else y_test.squeeze()
    else:
        y_test_series = y_test
    
    metricas = {}
    for nombre, res in resultados.items():
        y_pred = res['modelo'].predict(X_test)
        metricas[nombre] = {
            "RMSE": mean_squared_error(y_test_series, y_pred, squared=False),
            "MAE": mean_absolute_error(y_test_series, y_pred),
            "R2": r2_score(y_test_series, y_pred)
        }
    return pd.DataFrame(metricas).T

# ==========================
# ===== Clustering ==========
# ==========================

def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    logger.info("Estandarizando features...")
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    logger.info("Features estandarizadas exitosamente")
    return X_scaled, scaler


def train_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    logger.info(f"Entrenando modelo K-Means con {n_clusters} clusters...")
    
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    
    labels = model.fit_predict(X)
    
    logger.info("Modelo K-Means entrenado exitosamente")
    return model, labels


def train_dbscan(
    X: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Tuple[DBSCAN, np.ndarray]:
    logger.info(f"Entrenando modelo DBSCAN (eps={eps}, min_samples={min_samples})...")
    
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"DBSCAN completado: {n_clusters} clusters, {n_noise} puntos de ruido")
    
    return model, labels


def train_agglomerative_fast(X: pd.DataFrame, n_clusters: int = 3, linkage: str = "ward") -> Tuple[AgglomerativeClustering, np.ndarray]:
    logger.info(f"Entrenando modelo Agglomerative Clustering con {n_clusters} clusters...")
    logger.info(f"Reduciendo dimensiones con PCA para optimizar memoria...")
    
    n_components = min(10, X.shape[1])
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    logger.info(f"Dimensiones reducidas: {X.shape[1]} -> {n_components}")
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(X_reduced)
    
    logger.info("Modelo Agglomerative Clustering entrenado exitosamente")
    return model, labels


def train_agglomerative_with_fallback(
    X: pd.DataFrame,
    n_clusters: int = None,
    fallback_n_clusters: int = 3,
    linkage: str = "ward",
) -> Tuple[AgglomerativeClustering, np.ndarray]:

    if n_clusters is None or not isinstance(n_clusters, int) or n_clusters < 1:
        logger.warning(f"n_clusters es None o invalido ({n_clusters}), usando fallback: {fallback_n_clusters}")
        n_clusters = fallback_n_clusters
    
    return train_agglomerative_fast(X, n_clusters, linkage)


def train_gmm(
    X: pd.DataFrame,
    n_components: int = 3,
    random_state: int = 42,
) -> Tuple[GaussianMixture, np.ndarray]:

    logger.info(f"Entrenando modelo GMM con {n_components} componentes...")
    
    model = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=10,
    )
    
    labels = model.fit_predict(X)
    
    logger.info("Modelo GMM entrenado exitosamente")
    return model, labels


def train_gmm_with_fallback(
    X: pd.DataFrame,
    n_components: int = None,
    fallback_n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[GaussianMixture, np.ndarray]:

    if n_components is None or not isinstance(n_components, int) or n_components < 1:
        logger.warning(f"n_components es None o invalido ({n_components}), usando fallback: {fallback_n_clusters}")
        n_components = fallback_n_clusters
    
    return train_gmm(X, n_components, random_state)


def evaluate_clustering(
    X: pd.DataFrame,
    labels: np.ndarray,
) -> Dict[str, Any]:

    logger.info("Evaluando modelo de clustering...")
    
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        logger.warning("No hay suficientes puntos validos para calcular metricas")
        return {
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise": list(labels).count(-1),
            "silhouette_score": None,
            "davies_bouldin_score": None,
        }
    
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    n_clusters = len(set(labels_valid))
    
    if n_clusters < 2:
        logger.warning("Solo hay 1 cluster, no se pueden calcular metricas")
        metrics = {
            "n_clusters": n_clusters,
            "n_noise": list(labels).count(-1),
            "silhouette_score": None,
            "davies_bouldin_score": None,
        }
    else:
        metrics = {
            "n_clusters": n_clusters,
            "n_noise": list(labels).count(-1),
            "silhouette_score": silhouette_score(X_valid, labels_valid),
            "davies_bouldin_score": davies_bouldin_score(X_valid, labels_valid),
            "calinski_harabasz_score": calinski_harabasz_score(X_valid, labels_valid),
        }
        
        logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        logger.info(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
    
    return metrics


def add_cluster_labels_to_data(
    data: pd.DataFrame,
    labels: np.ndarray,
    cluster_column_name: str = "cluster",
) -> pd.DataFrame:
    logger.info(f"Anadiendo etiquetas de cluster al DataFrame...")
    
    min_len = min(len(data), len(labels))
    if len(data) != len(labels):
        logger.warning(
            f"Tamanos no coinciden: data={len(data)}, labels={len(labels)}. "
            f"Usando primeros {min_len} elementos."
        )
        data = data.iloc[:min_len].reset_index(drop=True)
        labels = labels[:min_len]
    
    data_with_clusters = data.copy()
    data_with_clusters[cluster_column_name] = labels
    
    logger.info(f"Clusters anadidos: {len(set(labels))} clusters unicos")
    
    return data_with_clusters
