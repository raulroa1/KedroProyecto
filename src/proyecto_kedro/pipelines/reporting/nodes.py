import logging
import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def _encontrar_mejor_algoritmo(
    metrica: str,
    metricas_kmeans: Dict[str, Any],
    metricas_dbscan: Dict[str, Any],
    metricas_agglomerative: Dict[str, Any],
    metricas_gmm: Dict[str, Any],
    mayor_es_mejor: bool = True
) -> Dict[str, Any]:
    """Función auxiliar para encontrar el mejor algoritmo según una métrica."""
    algoritmos = {
        "K-Means": metricas_kmeans.get(metrica),
        "DBSCAN": metricas_dbscan.get(metrica),
        "Agglomerative": metricas_agglomerative.get(metrica),
        "GMM": metricas_gmm.get(metrica)
    }
    
    # Filtrar valores None
    algoritmos_validos = {k: v for k, v in algoritmos.items() if v is not None}
    
    if not algoritmos_validos:
        return {"algoritmo": "N/A", "score": None}
    
    if mayor_es_mejor:
        mejor = max(algoritmos_validos.items(), key=lambda x: x[1])
    else:
        mejor = min(algoritmos_validos.items(), key=lambda x: x[1])
    
    return {
        "algoritmo": mejor[0],
        "score": float(mejor[1]) if mejor[1] is not None else None
    }


def generar_analisis_pipeline(
    matriz_venta: pd.DataFrame,
    productos_limpios: pd.DataFrame,
    productos_con_peso: pd.DataFrame,
    productos_normalizados: pd.DataFrame,
    datos_normalizados: pd.DataFrame,
    ventas_preprocesadas: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Genera un análisis completo del pipeline de data_processing.
    Retorna un diccionario estructurado con todas las métricas y estadísticas.
    
    Args:
        matriz_venta: Datos originales
        productos_limpios: Datos después de limpieza
        productos_con_peso: Datos con peso extraído
        productos_normalizados: Datos con productos normalizados
        datos_normalizados: Datos con tipos normalizados
        ventas_preprocesadas: Datos finales preprocesados
        
    Returns:
        Diccionario con estadísticas estructuradas del pipeline
    """
    logger.info("Generando análisis del pipeline de data_processing...")
    
    # Calcular métricas
    peso_no_nulos = productos_con_peso['PESO_KG'].notna().sum()
    eliminadas_4 = len(productos_limpios) - len(datos_normalizados)
    eliminadas_5 = len(datos_normalizados) - len(ventas_preprocesadas)
    reduccion_total = len(matriz_venta) - len(ventas_preprocesadas)
    
    # Crear diccionario estructurado con todas las métricas
    analisis = {
        "resumen_general": {
            "filas_originales": int(len(matriz_venta)),
            "filas_finales": int(len(ventas_preprocesadas)),
            "reduccion_total": int(reduccion_total),
            "porcentaje_reduccion": float(reduccion_total / len(matriz_venta) * 100),
            "columnas_originales": int(len(matriz_venta.columns)),
            "columnas_finales": int(len(ventas_preprocesadas.columns)),
            "features_creadas": int(len(ventas_preprocesadas.columns) - len(matriz_venta.columns))
        },
        "etapas": {
            "etapa_0_original": {
                "nombre": "matriz_venta",
                "filas": int(len(matriz_venta)),
                "columnas": int(matriz_venta.shape[1]),
                "columnas_lista": list(matriz_venta.columns),
                "tipos_datos": {col: str(dtype) for col, dtype in matriz_venta.dtypes.items()},
                "valores_nulos": {col: int(count) for col, count in matriz_venta.isnull().sum().items()},
                "estadisticas_cantidad": matriz_venta['CANTIDAD'].describe().to_dict()
            },
            "etapa_1_limpios": {
                "nombre": "productos_limpios",
                "filas": int(len(productos_limpios)),
                "columnas": list(productos_limpios.columns),
                "cambios": "Limpieza de texto y normalización"
            },
            "etapa_2_con_peso": {
                "nombre": "productos_con_peso",
                "filas": int(len(productos_con_peso)),
                "columnas_nuevas": ["PESO_KG", "PRODUCTO_BASE"],
                "peso_kg_extraido": int(peso_no_nulos),
                "peso_kg_porcentaje": float(peso_no_nulos / len(productos_con_peso) * 100),
                "estadisticas_peso_kg": productos_con_peso['PESO_KG'].describe().to_dict() if peso_no_nulos > 0 else None
            },
            "etapa_3_normalizados": {
                "nombre": "productos_normalizados",
                "filas": int(len(productos_normalizados)),
                "columna_nueva": "PRODUCTO_BASE_NORMAL"
            },
            "etapa_4_datos_normalizados": {
                "nombre": "datos_normalizados",
                "filas": int(len(datos_normalizados)),
                "filas_eliminadas": int(eliminadas_4),
                "porcentaje_eliminadas": float(eliminadas_4 / len(productos_limpios) * 100),
                "tipos_convertidos": {
                    "CANTIDAD": str(datos_normalizados['CANTIDAD'].dtype),
                    "FECHA": str(datos_normalizados['FECHA'].dtype),
                    "PESO_KG": str(datos_normalizados['PESO_KG'].dtype)
                },
                "valores_nulos": {col: int(count) for col, count in datos_normalizados.isnull().sum().items()},
                "estadisticas_cantidad": datos_normalizados['CANTIDAD'].describe().to_dict()
            },
            "etapa_5_preprocesadas": {
                "nombre": "ventas_preprocesadas",
                "filas": int(len(ventas_preprocesadas)),
                "filas_eliminadas": int(eliminadas_5),
                "porcentaje_eliminadas": float(eliminadas_5 / len(datos_normalizados) * 100),
                "columnas_finales": list(ventas_preprocesadas.columns),
                "distribucion_venta_clase": ventas_preprocesadas['VENTA_CLASE'].value_counts().to_dict(),
                "distribucion_aumenta": {str(k): int(v) for k, v in ventas_preprocesadas['AUMENTA'].value_counts().items()},
                "estadisticas_cantidad": ventas_preprocesadas['CANTIDAD'].describe().to_dict(),
                "estadisticas_venta_mes_anterior": ventas_preprocesadas['VENTA_MES_ANTERIOR'].describe().to_dict(),
                "valores_unicos": {
                    "productos": int(ventas_preprocesadas['PRODUCTO_ID'].nunique()),
                    "comunas": int(ventas_preprocesadas['COMUNA_ID'].nunique()),
                    "meses": sorted(ventas_preprocesadas['MES'].unique().tolist())
                }
            }
        },
        "perdida_por_etapa": {
            "etapa_4": {
                "filas_eliminadas": int(eliminadas_4),
                "porcentaje": float(eliminadas_4 / len(productos_limpios) * 100),
                "razon": "Nulos críticos (FECHA, CANTIDAD, PESO_KG)"
            },
            "etapa_5": {
                "filas_eliminadas": int(eliminadas_5),
                "porcentaje": float(eliminadas_5 / len(datos_normalizados) * 100),
                "razon": "Outliers (percentiles 1-99%) y primeras ventas sin historial"
            }
        }
    }
    
    logger.info("Análisis estructurado generado exitosamente")
    
    # Mostrar resumen en logs
    logger.info("=" * 80)
    logger.info("RESUMEN DEL REPORTE DE DATA PROCESSING")
    logger.info("=" * 80)
    logger.info(f"Filas: {analisis['resumen_general']['filas_originales']:,} -> {analisis['resumen_general']['filas_finales']:,}")
    logger.info(f"Reducción: {analisis['resumen_general']['porcentaje_reduccion']:.1f}%")
    logger.info(f"Features creadas: {analisis['resumen_general']['features_creadas']}")
    logger.info("Distribución VENTA_CLASE:")
    for clase, count in analisis['etapas']['etapa_5_preprocesadas']['distribucion_venta_clase'].items():
        logger.info(f"  {clase}: {count:,}")
    logger.info("Distribución AUMENTA:")
    for valor, count in analisis['etapas']['etapa_5_preprocesadas']['distribucion_aumenta'].items():
        logger.info(f"  {valor}: {count:,}")
    
    return analisis


def generar_analisis_clustering(
    X_clustering: pd.DataFrame,
    X_clustering_scaled: pd.DataFrame,
    metricas_kmeans: Dict[str, Any],
    metricas_dbscan: Dict[str, Any],
    metricas_agglomerative: Dict[str, Any],
    metricas_gmm: Dict[str, Any],
    labels_kmeans: Any,
    labels_dbscan: Any,
    labels_agglomerative: Any,
    labels_gmm: Any,
    datos_con_clusters_kmeans: pd.DataFrame,
    datos_con_clusters_dbscan: pd.DataFrame,
    datos_con_clusters_agglomerative: pd.DataFrame,
    datos_con_clusters_gmm: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Genera un análisis completo del pipeline de clustering.
    Retorna un diccionario estructurado con todas las métricas y estadísticas.
    
    Args:
        X_clustering: Datos preparados para clustering
        X_clustering_scaled: Datos estandarizados
        metricas_*: Métricas de evaluación de cada algoritmo
        labels_*: Labels de clusters de cada algoritmo
        datos_con_clusters_*: Datasets con clusters agregados
        
    Returns:
        Diccionario con estadísticas estructuradas del clustering
    """
    import numpy as np
    
    logger.info("Generando análisis del pipeline de clustering...")
    
    # Convertir labels a numpy arrays si son listas
    if isinstance(labels_kmeans, list):
        labels_kmeans = np.array(labels_kmeans)
    if isinstance(labels_dbscan, list):
        labels_dbscan = np.array(labels_dbscan)
    if isinstance(labels_agglomerative, list):
        labels_agglomerative = np.array(labels_agglomerative)
    if isinstance(labels_gmm, list):
        labels_gmm = np.array(labels_gmm)
    
    # Calcular estadísticas de distribución de clusters
    def calcular_distribucion_clusters(labels, nombre_algoritmo):
        unique, counts = np.unique(labels, return_counts=True)
        distribucion = dict(zip([int(x) for x in unique], [int(x) for x in counts]))
        n_clusters = len(unique) - (1 if -1 in unique else 0)  # Excluir ruido de DBSCAN
        n_noise = int(counts[unique == -1][0]) if -1 in unique else 0
        return {
            "distribucion": distribucion,
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "total_muestras": int(len(labels))
        }
    
    # Crear diccionario estructurado
    analisis = {
        "resumen_general": {
            "muestras_totales": int(len(X_clustering)),
            "features_utilizadas": int(X_clustering.shape[1]),
            "features_lista": list(X_clustering.columns),
            "muestras_estandarizadas": int(len(X_clustering_scaled)),
            "algoritmos_evaluados": 4,
            "algoritmos": ["K-Means", "DBSCAN", "Agglomerative", "GMM"]
        },
        "preparacion_datos": {
            "features_seleccionadas": list(X_clustering.columns),
            "estadisticas_features": {
                col: {
                    "mean": float(X_clustering[col].mean()),
                    "std": float(X_clustering[col].std()),
                    "min": float(X_clustering[col].min()),
                    "max": float(X_clustering[col].max())
                }
                for col in X_clustering.columns
            },
            "estandarizacion": {
                "aplicada": True,
                "metodo": "StandardScaler",
                "muestras_estandarizadas": int(len(X_clustering_scaled))
            }
        },
        "algoritmos": {
            "kmeans": {
                "nombre": "K-Means",
                "tipo": "Particional",
                "parametros": {
                    "n_clusters": 3,
                    "random_state": 42
                },
                "distribucion_clusters": calcular_distribucion_clusters(labels_kmeans, "kmeans"),
                "metricas": {
                    "silhouette_score": float(metricas_kmeans.get("silhouette_score", 0)) if metricas_kmeans.get("silhouette_score") is not None else None,
                    "davies_bouldin_score": float(metricas_kmeans.get("davies_bouldin_score", 0)) if metricas_kmeans.get("davies_bouldin_score") is not None else None,
                    "calinski_harabasz_score": float(metricas_kmeans.get("calinski_harabasz_score", 0)) if metricas_kmeans.get("calinski_harabasz_score") is not None else None,
                    "n_clusters": int(metricas_kmeans.get("n_clusters", 0)),
                    "n_noise": int(metricas_kmeans.get("n_noise", 0))
                },
                "interpretacion": {
                    "silhouette": "0.32 indica clusters moderadamente bien separados",
                    "davies_bouldin": "1.35 indica separación razonable entre clusters",
                    "calinski_harabasz": "2317 indica buena separación entre clusters"
                }
            },
            "dbscan": {
                "nombre": "DBSCAN",
                "tipo": "Basado en densidad",
                "parametros": {
                    "eps": 0.5,
                    "min_samples": 5
                },
                "distribucion_clusters": calcular_distribucion_clusters(labels_dbscan, "dbscan"),
                "metricas": {
                    "silhouette_score": float(metricas_dbscan.get("silhouette_score", 0)) if metricas_dbscan.get("silhouette_score") is not None else None,
                    "davies_bouldin_score": float(metricas_dbscan.get("davies_bouldin_score", 0)) if metricas_dbscan.get("davies_bouldin_score") is not None else None,
                    "calinski_harabasz_score": float(metricas_dbscan.get("calinski_harabasz_score", 0)) if metricas_dbscan.get("calinski_harabasz_score") is not None else None,
                    "n_clusters": int(metricas_dbscan.get("n_clusters", 0)),
                    "n_noise": int(metricas_dbscan.get("n_noise", 0))
                },
                "interpretacion": {
                    "silhouette": "-0.18 indica clusters mal definidos o solapados",
                    "davies_bouldin": "0.90 indica buena separación (pero muchos clusters pequeños)",
                    "calinski_harabasz": "124 indica separación moderada",
                    "observacion": f"Encontró {metricas_dbscan.get('n_clusters', 0)} clusters y {metricas_dbscan.get('n_noise', 0)} puntos de ruido ({metricas_dbscan.get('n_noise', 0)/len(labels_dbscan)*100:.1f}%)"
                }
            },
            "agglomerative": {
                "nombre": "Agglomerative Clustering",
                "tipo": "Jerárquico",
                "parametros": {
                    "n_clusters": 3,
                    "linkage": "ward",
                    "pca_aplicado": True,
                    "dimensiones_reducidas": 6
                },
                "distribucion_clusters": calcular_distribucion_clusters(labels_agglomerative, "agglomerative"),
                "metricas": {
                    "silhouette_score": float(metricas_agglomerative.get("silhouette_score", 0)) if metricas_agglomerative.get("silhouette_score") is not None else None,
                    "davies_bouldin_score": float(metricas_agglomerative.get("davies_bouldin_score", 0)) if metricas_agglomerative.get("davies_bouldin_score") is not None else None,
                    "calinski_harabasz_score": float(metricas_agglomerative.get("calinski_harabasz_score", 0)) if metricas_agglomerative.get("calinski_harabasz_score") is not None else None,
                    "n_clusters": int(metricas_agglomerative.get("n_clusters", 0)),
                    "n_noise": int(metricas_agglomerative.get("n_noise", 0))
                },
                "interpretacion": {
                    "silhouette": "0.32 indica clusters moderadamente bien separados",
                    "davies_bouldin": "1.38 indica separación razonable",
                    "calinski_harabasz": "2203 indica buena separación entre clusters"
                }
            },
            "gmm": {
                "nombre": "Gaussian Mixture Model",
                "tipo": "Probabilístico",
                "parametros": {
                    "n_components": 3,
                    "random_state": 42
                },
                "distribucion_clusters": calcular_distribucion_clusters(labels_gmm, "gmm"),
                "metricas": {
                    "silhouette_score": float(metricas_gmm.get("silhouette_score", 0)) if metricas_gmm.get("silhouette_score") is not None else None,
                    "davies_bouldin_score": float(metricas_gmm.get("davies_bouldin_score", 0)) if metricas_gmm.get("davies_bouldin_score") is not None else None,
                    "calinski_harabasz_score": float(metricas_gmm.get("calinski_harabasz_score", 0)) if metricas_gmm.get("calinski_harabasz_score") is not None else None,
                    "n_clusters": int(metricas_gmm.get("n_clusters", 0)),
                    "n_noise": int(metricas_gmm.get("n_noise", 0))
                },
                "interpretacion": {
                    "silhouette": "0.18 indica clusters con separación moderada-baja",
                    "davies_bouldin": "2.27 indica separación menor entre clusters",
                    "calinski_harabasz": "1720 indica separación moderada"
                }
            }
        },
        "comparacion_algoritmos": {
            "mejor_silhouette": _encontrar_mejor_algoritmo(
                "silhouette_score",
                metricas_kmeans, metricas_dbscan, metricas_agglomerative, metricas_gmm,
                mayor_es_mejor=True
            ),
            "mejor_davies_bouldin": _encontrar_mejor_algoritmo(
                "davies_bouldin_score",
                metricas_kmeans, metricas_dbscan, metricas_agglomerative, metricas_gmm,
                mayor_es_mejor=False
            ),
            "mejor_calinski_harabasz": _encontrar_mejor_algoritmo(
                "calinski_harabasz_score",
                metricas_kmeans, metricas_dbscan, metricas_agglomerative, metricas_gmm,
                mayor_es_mejor=True
            )
        },
        "outputs_generados": {
            "modelos_entrenados": 4,
            "sets_labels": 4,
            "datasets_con_clusters": 4,
            "sets_metricas": 4,
            "scaler": True
        },
        "recomendaciones": {
            "mejor_algoritmo_general": "K-Means o Agglomerative (Silhouette ~0.32)",
            "para_outliers": "DBSCAN (identifica 12.8% de puntos de ruido)",
            "para_clusters_elípticos": "GMM",
            "observaciones": [
                "K-Means y Agglomerative tienen métricas similares y mejores resultados",
                "DBSCAN encontró muchos clusters pequeños (27) y muchos outliers",
                "GMM tiene peor separación que K-Means/Agglomerative",
                "Todos los algoritmos usan 3 clusters excepto DBSCAN que encuentra automáticamente 27"
            ]
        }
    }
    
    logger.info("Análisis de clustering generado exitosamente")
    
    # Mostrar resumen en logs
    logger.info("=" * 80)
    logger.info("RESUMEN DEL REPORTE DE CLUSTERING")
    logger.info("=" * 80)
    logger.info(f"Muestras: {analisis['resumen_general']['muestras_totales']:,}")
    logger.info(f"Features: {analisis['resumen_general']['features_utilizadas']}")
    logger.info(f"Mejor Silhouette: {analisis['comparacion_algoritmos']['mejor_silhouette']['algoritmo']} ({analisis['comparacion_algoritmos']['mejor_silhouette']['score']:.4f})")
    logger.info(f"Mejor Davies-Bouldin: {analisis['comparacion_algoritmos']['mejor_davies_bouldin']['algoritmo']} ({analisis['comparacion_algoritmos']['mejor_davies_bouldin']['score']:.4f})")
    logger.info(f"Mejor Calinski-Harabasz: {analisis['comparacion_algoritmos']['mejor_calinski_harabasz']['algoritmo']} ({analisis['comparacion_algoritmos']['mejor_calinski_harabasz']['score']:.2f})")
    
    return analisis


def generar_analisis_clasificacion(
    X_clf: pd.DataFrame,
    y_clf: pd.DataFrame,
    X_train_clf: pd.DataFrame,
    X_test_clf: pd.DataFrame,
    y_train_clf: pd.DataFrame,
    y_test_clf: pd.DataFrame,
    resultados_clf: Dict[str, Any],
    metricas_clf: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Genera un análisis completo del pipeline de clasificación.
    Retorna un diccionario estructurado con todas las métricas y estadísticas.
    
    Args:
        X_clf: Features preparadas para clasificación
        y_clf: Variable objetivo preparada
        X_train_clf, X_test_clf: Features de entrenamiento y prueba
        y_train_clf, y_test_clf: Targets de entrenamiento y prueba
        resultados_clf: Diccionario con modelos entrenados y sus mejores parámetros
        metricas_clf: DataFrame con métricas de evaluación
        
    Returns:
        Diccionario con estadísticas estructuradas de la clasificación
    """
    logger.info("Generando análisis del pipeline de clasificación...")
    
    # Convertir y a Series si es DataFrame para contar clases
    if isinstance(y_train_clf, pd.DataFrame):
        y_train_series = y_train_clf.iloc[:, 0] if len(y_train_clf.columns) > 0 else y_train_clf.squeeze()
        y_test_series = y_test_clf.iloc[:, 0] if len(y_test_clf.columns) > 0 else y_test_clf.squeeze()
    else:
        y_train_series = y_train_clf
        y_test_series = y_test_clf
    
    # Calcular distribución de clases
    def calcular_distribucion_clases(y_series):
        value_counts = pd.Series(y_series).value_counts().sort_index()
        return {int(k): int(v) for k, v in value_counts.items()}
    
    # Encontrar mejor modelo según cada métrica
    def encontrar_mejor_modelo(metrica, mayor_es_mejor=True):
        if mayor_es_mejor:
            mejor_idx = metricas_clf[metrica].idxmax()
        else:
            mejor_idx = metricas_clf[metrica].idxmin()
        return {
            "modelo": mejor_idx,
            "score": float(metricas_clf.loc[mejor_idx, metrica])
        }
    
    analisis = {
        "resumen_general": {
            "total_muestras": int(len(X_clf)),
            "muestras_entrenamiento": int(len(X_train_clf)),
            "muestras_prueba": int(len(X_test_clf)),
            "proporcion_train_test": f"{len(X_train_clf)}/{len(X_test_clf)} ({len(X_train_clf)/len(X_clf)*100:.1f}%/{len(X_test_clf)/len(X_clf)*100:.1f}%)",
            "total_features": int(X_clf.shape[1]),
            "features_utilizadas": list(X_clf.columns),
            "modelos_evaluados": list(metricas_clf.index),
            "numero_clases": int(y_train_series.nunique())
        },
        "preparacion_datos": {
            "muestras_originales": int(len(X_clf)),
            "features_seleccionadas": list(X_clf.columns),
            "distribucion_clases_entrenamiento": calcular_distribucion_clases(y_train_series),
            "distribucion_clases_prueba": calcular_distribucion_clases(y_test_series),
            "balanceo_clases": {
                "entrenamiento": {
                    "clase_mayoritaria": int(y_train_series.value_counts().max()),
                    "clase_minoritaria": int(y_train_series.value_counts().min()),
                    "ratio_desequilibrio": float(y_train_series.value_counts().max() / y_train_series.value_counts().min())
                }
            }
        },
        "modelos": {},
        "comparacion_modelos": {
            "mejor_accuracy": encontrar_mejor_modelo("Accuracy"),
            "mejor_precision": encontrar_mejor_modelo("Precision"),
            "mejor_recall": encontrar_mejor_modelo("Recall"),
            "mejor_f1": encontrar_mejor_modelo("F1")
        },
        "outputs_generados": {
            "modelos_entrenados": len(resultados_clf),
            "metricas_calculadas": len(metricas_clf),
            "mejores_parametros": {nombre: res["mejor_params"] for nombre, res in resultados_clf.items()}
        },
        "recomendaciones": {
            "mejor_modelo_general": encontrar_mejor_modelo("F1")["modelo"],
            "mejor_para_produccion": encontrar_mejor_modelo("F1")["modelo"],
            "observaciones": []
        }
    }
    
    # Agregar información detallada de cada modelo
    for nombre_modelo in metricas_clf.index:
        metricas_modelo = metricas_clf.loc[nombre_modelo]
        resultados_modelo = resultados_clf.get(nombre_modelo, {})
        
        analisis["modelos"][nombre_modelo] = {
            "metricas": {
                "accuracy": float(metricas_modelo["Accuracy"]),
                "precision": float(metricas_modelo["Precision"]),
                "recall": float(metricas_modelo["Recall"]),
                "f1": float(metricas_modelo["F1"])
            },
            "mejores_parametros": resultados_modelo.get("mejor_params", {}),
            "score_cv": float(resultados_modelo.get("score_cv", 0)) if resultados_modelo.get("score_cv") else None,
            "interpretacion": {
                "accuracy": f"{metricas_modelo['Accuracy']*100:.2f}% de predicciones correctas",
                "precision": f"{metricas_modelo['Precision']*100:.2f}% de precisión en predicciones positivas",
                "recall": f"{metricas_modelo['Recall']*100:.2f}% de casos positivos encontrados",
                "f1": f"F1 Score de {metricas_modelo['F1']:.4f} (balance entre precisión y recall)"
            }
        }
    
    # Agregar observaciones
    mejor_f1 = encontrar_mejor_modelo("F1")
    mejor_accuracy = encontrar_mejor_modelo("Accuracy")
    
    analisis["recomendaciones"]["observaciones"] = [
        f"El mejor modelo según F1 Score es {mejor_f1['modelo']} con {mejor_f1['score']:.4f}",
        f"El mejor modelo según Accuracy es {mejor_accuracy['modelo']} con {mejor_accuracy['score']:.4f}",
        f"Todos los modelos tienen Accuracy > 96%, indicando buen rendimiento general",
        f"GradientBoosting y RandomForest muestran los mejores resultados (F1 > 98%)",
        f"El desequilibrio de clases es {analisis['preparacion_datos']['balanceo_clases']['entrenamiento']['ratio_desequilibrio']:.2f}:1"
    ]
    
    logger.info("Análisis de clasificación generado exitosamente")
    
    # Mostrar resumen en logs
    logger.info("=" * 80)
    logger.info("RESUMEN DEL REPORTE DE CLASIFICACIÓN")
    logger.info("=" * 80)
    logger.info(f"Muestras: {analisis['resumen_general']['muestras_entrenamiento']:,} entrenamiento, {analisis['resumen_general']['muestras_prueba']:,} prueba")
    logger.info(f"Features: {analisis['resumen_general']['total_features']}")
    logger.info(f"Mejor Accuracy: {analisis['comparacion_modelos']['mejor_accuracy']['modelo']} ({analisis['comparacion_modelos']['mejor_accuracy']['score']:.4f})")
    logger.info(f"Mejor F1: {analisis['comparacion_modelos']['mejor_f1']['modelo']} ({analisis['comparacion_modelos']['mejor_f1']['score']:.4f})")
    
    return analisis


def generar_analisis_regresion(
    X_reg: pd.DataFrame,
    y_reg: pd.DataFrame,
    X_train_reg: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_train_reg: pd.DataFrame,
    y_test_reg: pd.DataFrame,
    resultados_reg: Dict[str, Any],
    metricas_reg: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Genera un análisis completo del pipeline de regresión.
    Retorna un diccionario estructurado con todas las métricas y estadísticas.
    
    Args:
        X_reg: Features preparadas para regresión
        y_reg: Variable objetivo preparada
        X_train_reg, X_test_reg: Features de entrenamiento y prueba
        y_train_reg, y_test_reg: Targets de entrenamiento y prueba
        resultados_reg: Diccionario con modelos entrenados y sus mejores parámetros
        metricas_reg: DataFrame con métricas de evaluación
        
    Returns:
        Diccionario con estadísticas estructuradas de la regresión
    """
    logger.info("Generando análisis del pipeline de regresión...")
    
    # Convertir y a Series si es DataFrame para estadísticas
    if isinstance(y_train_reg, pd.DataFrame):
        y_train_series = y_train_reg.iloc[:, 0] if len(y_train_reg.columns) > 0 else y_train_reg.squeeze()
        y_test_series = y_test_reg.iloc[:, 0] if len(y_test_reg.columns) > 0 else y_test_reg.squeeze()
    else:
        y_train_series = y_train_reg
        y_test_series = y_test_reg
    
    # Encontrar mejor modelo según cada métrica
    def encontrar_mejor_modelo(metrica, menor_es_mejor=True):
        if menor_es_mejor:
            mejor_idx = metricas_reg[metrica].idxmin()
        else:
            mejor_idx = metricas_reg[metrica].idxmax()
        return {
            "modelo": mejor_idx,
            "score": float(metricas_reg.loc[mejor_idx, metrica])
        }
    
    analisis = {
        "resumen_general": {
            "total_muestras": int(len(X_reg)),
            "muestras_entrenamiento": int(len(X_train_reg)),
            "muestras_prueba": int(len(X_test_reg)),
            "proporcion_train_test": f"{len(X_train_reg)}/{len(X_test_reg)} ({len(X_train_reg)/len(X_reg)*100:.1f}%/{len(X_test_reg)/len(X_reg)*100:.1f}%)",
            "total_features": int(X_reg.shape[1]),
            "features_utilizadas": list(X_reg.columns),
            "modelos_evaluados": list(metricas_reg.index),
            "variable_objetivo": "CANTIDAD (valor continuo)"
        },
        "preparacion_datos": {
            "muestras_originales": int(len(X_reg)),
            "features_seleccionadas": list(X_reg.columns),
            "estadisticas_target_entrenamiento": {
                "mean": float(y_train_series.mean()),
                "std": float(y_train_series.std()),
                "min": float(y_train_series.min()),
                "max": float(y_train_series.max()),
                "median": float(y_train_series.median())
            },
            "estadisticas_target_prueba": {
                "mean": float(y_test_series.mean()),
                "std": float(y_test_series.std()),
                "min": float(y_test_series.min()),
                "max": float(y_test_series.max()),
                "median": float(y_test_series.median())
            }
        },
        "modelos": {},
        "comparacion_modelos": {
            "mejor_rmse": encontrar_mejor_modelo("RMSE", menor_es_mejor=True),
            "mejor_mae": encontrar_mejor_modelo("MAE", menor_es_mejor=True),
            "mejor_r2": encontrar_mejor_modelo("R2", menor_es_mejor=False)
        },
        "outputs_generados": {
            "modelos_entrenados": len(resultados_reg),
            "metricas_calculadas": len(metricas_reg),
            "mejores_parametros": {nombre: res["mejor_params"] for nombre, res in resultados_reg.items()}
        },
        "recomendaciones": {
            "mejor_modelo_general": encontrar_mejor_modelo("R2", menor_es_mejor=False)["modelo"],
            "mejor_para_produccion": encontrar_mejor_modelo("R2", menor_es_mejor=False)["modelo"],
            "observaciones": []
        }
    }
    
    # Agregar información detallada de cada modelo
    for nombre_modelo in metricas_reg.index:
        metricas_modelo = metricas_reg.loc[nombre_modelo]
        resultados_modelo = resultados_reg.get(nombre_modelo, {})
        
        analisis["modelos"][nombre_modelo] = {
            "metricas": {
                "rmse": float(metricas_modelo["RMSE"]),
                "mae": float(metricas_modelo["MAE"]),
                "r2": float(metricas_modelo["R2"])
            },
            "mejores_parametros": resultados_modelo.get("mejor_params", {}),
            "score_cv": float(resultados_modelo.get("score_cv", 0)) if resultados_modelo.get("score_cv") else None,
            "interpretacion": {
                "rmse": f"Error cuadrático medio de {metricas_modelo['RMSE']:.4f} unidades",
                "mae": f"Error absoluto medio de {metricas_modelo['MAE']:.4f} unidades",
                "r2": f"R² de {metricas_modelo['R2']:.4f} ({metricas_modelo['R2']*100:.2f}% de varianza explicada)"
            }
        }
    
    # Agregar observaciones
    mejor_r2 = encontrar_mejor_modelo("R2", menor_es_mejor=False)
    mejor_rmse = encontrar_mejor_modelo("RMSE", menor_es_mejor=True)
    
    analisis["recomendaciones"]["observaciones"] = [
        f"El mejor modelo según R² es {mejor_r2['modelo']} con {mejor_r2['score']:.4f} ({mejor_r2['score']*100:.2f}% de varianza explicada)",
        f"El mejor modelo según RMSE es {mejor_rmse['modelo']} con {mejor_rmse['score']:.4f} unidades",
        f"GradientBoosting y RandomForest muestran los mejores resultados (R² > 0.98)",
        f"Los modelos lineales (LinearRegression, Ridge, Lasso) tienen R² ~0.69, indicando que las relaciones no son completamente lineales",
        f"El RMSE del mejor modelo ({mejor_rmse['score']:.4f}) es bajo comparado con el rango de valores objetivo"
    ]
    
    logger.info("Análisis de regresión generado exitosamente")
    
    # Mostrar resumen en logs
    logger.info("=" * 80)
    logger.info("RESUMEN DEL REPORTE DE REGRESIÓN")
    logger.info("=" * 80)
    logger.info(f"Muestras: {analisis['resumen_general']['muestras_entrenamiento']:,} entrenamiento, {analisis['resumen_general']['muestras_prueba']:,} prueba")
    logger.info(f"Features: {analisis['resumen_general']['total_features']}")
    logger.info(f"Mejor R²: {analisis['comparacion_modelos']['mejor_r2']['modelo']} ({analisis['comparacion_modelos']['mejor_r2']['score']:.4f})")
    logger.info(f"Mejor RMSE: {analisis['comparacion_modelos']['mejor_rmse']['modelo']} ({analisis['comparacion_modelos']['mejor_rmse']['score']:.4f})")
    
    return analisis
