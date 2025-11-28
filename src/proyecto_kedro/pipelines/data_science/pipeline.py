from kedro.pipeline import Node, Pipeline

from .nodes import (
    preparar_datos_clasificacion,
    preparar_datos_regresion,
    pre_proceso_clf,
    dividir_datos_clf,
    entrenar_modelos_clasificacion_cv,
    evaluar_modelos_clasificacion,
    pre_proceso_rg,
    dividir_datos_reg,
    entrenar_modelos_regresion_cv,
    evaluar_modelos_regresion,
    prepare_clustering_data,
    scale_features,
    train_kmeans,
    train_dbscan,
    train_agglomerative_with_fallback,
    train_gmm_with_fallback,
    evaluate_clustering,
    add_cluster_labels_to_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Preparación de datos para clasificación
            Node(
                func=preparar_datos_clasificacion,
                inputs="ventas_preprocesadas",
                outputs=["X_clf", "y_clf"],
                name="preparar_datos_clasificacion_node",
            ),
            # Preprocesamiento de datos de clasificación
            Node(
                func=pre_proceso_clf,
                inputs=["X_clf", "y_clf"],
                outputs=["X_clf_proc", "y_clf_proc"],
                name="pre_proceso_clf_node",
            ),
            # División de datos de clasificación
            Node(
                func=dividir_datos_clf,
                inputs=["X_clf_proc", "y_clf_proc"],
                outputs=["X_train_clf", "X_test_clf", "y_train_clf", "y_test_clf"],
                name="dividir_datos_clf_node",
            ),
            # Entrenamiento de modelos de clasificación
            Node(
                func=entrenar_modelos_clasificacion_cv,
                inputs=["X_train_clf", "y_train_clf"],
                outputs="resultados_clf",
                name="entrenar_modelos_clasificacion_node",
            ),
            # Evaluación de modelos de clasificación
            Node(
                func=evaluar_modelos_clasificacion,
                inputs=["resultados_clf", "X_test_clf", "y_test_clf"],
                outputs="metricas_clf",
                name="evaluar_modelos_clasificacion_node",
            ),
            # Preparación de datos para regresión
            Node(
                func=preparar_datos_regresion,
                inputs="ventas_preprocesadas",
                outputs=["X_reg", "y_reg"],
                name="preparar_datos_regresion_node",
            ),
            # Preprocesamiento de datos de regresión
            Node(
                func=pre_proceso_rg,
                inputs=["X_reg", "y_reg"],
                outputs=["X_reg_proc", "y_reg_proc"],
                name="pre_proceso_rg_node",
            ),
            # División de datos de regresión
            Node(
                func=dividir_datos_reg,
                inputs=["X_reg_proc", "y_reg_proc"],
                outputs=["X_train_reg", "X_test_reg", "y_train_reg", "y_test_reg"],
                name="dividir_datos_reg_node",
            ),
            # Entrenamiento de modelos de regresión
            Node(
                func=entrenar_modelos_regresion_cv,
                inputs=["X_train_reg", "y_train_reg"],
                outputs="resultados_reg",
                name="entrenar_modelos_regresion_node",
            ),
            # Evaluación de modelos de regresión
            Node(
                func=evaluar_modelos_regresion,
                inputs=["resultados_reg", "X_test_reg", "y_test_reg"],
                outputs="metricas_reg",
                name="evaluar_modelos_regresion_node",
            ),
            # Preparación de datos para clustering
            Node(
                func=prepare_clustering_data,
                inputs="ventas_preprocesadas",
                outputs="X_clustering",
                name="prepare_clustering_data_node",
            ),
            # Estandarización de features para clustering
            Node(
                func=scale_features,
                inputs="X_clustering",
                outputs=["X_clustering_scaled", "scaler_clustering"],
                name="scale_features_node",
            ),
            # Entrenamiento K-Means
            Node(
                func=train_kmeans,
                inputs="X_clustering_scaled",
                outputs=["modelo_kmeans", "labels_kmeans"],
                name="train_kmeans_node",
            ),
            # Evaluación K-Means
            Node(
                func=evaluate_clustering,
                inputs=["X_clustering_scaled", "labels_kmeans"],
                outputs="metricas_kmeans",
                name="evaluate_kmeans_node",
            ),
            # Agregar labels K-Means a datos
            Node(
                func=add_cluster_labels_to_data,
                inputs=["X_clustering", "labels_kmeans"],
                outputs="datos_con_clusters_kmeans",
                name="add_clusters_kmeans_node",
            ),
            # Entrenamiento DBSCAN
            Node(
                func=train_dbscan,
                inputs="X_clustering_scaled",
                outputs=["modelo_dbscan", "labels_dbscan"],
                name="train_dbscan_node",
            ),
            # Evaluación DBSCAN
            Node(
                func=evaluate_clustering,
                inputs=["X_clustering_scaled", "labels_dbscan"],
                outputs="metricas_dbscan",
                name="evaluate_dbscan_node",
            ),
            # Agregar labels DBSCAN a datos
            Node(
                func=add_cluster_labels_to_data,
                inputs=["X_clustering", "labels_dbscan"],
                outputs="datos_con_clusters_dbscan",
                name="add_clusters_dbscan_node",
            ),
            # Entrenamiento Agglomerative
            Node(
                func=train_agglomerative_with_fallback,
                inputs="X_clustering_scaled",
                outputs=["modelo_agglomerative", "labels_agglomerative"],
                name="train_agglomerative_node",
            ),
            # Evaluación Agglomerative
            Node(
                func=evaluate_clustering,
                inputs=["X_clustering_scaled", "labels_agglomerative"],
                outputs="metricas_agglomerative",
                name="evaluate_agglomerative_node",
            ),
            # Agregar labels Agglomerative a datos
            Node(
                func=add_cluster_labels_to_data,
                inputs=["X_clustering", "labels_agglomerative"],
                outputs="datos_con_clusters_agglomerative",
                name="add_clusters_agglomerative_node",
            ),
            # Entrenamiento GMM
            Node(
                func=train_gmm_with_fallback,
                inputs="X_clustering_scaled",
                outputs=["modelo_gmm", "labels_gmm"],
                name="train_gmm_node",
            ),
            # Evaluación GMM
            Node(
                func=evaluate_clustering,
                inputs=["X_clustering_scaled", "labels_gmm"],
                outputs="metricas_gmm",
                name="evaluate_gmm_node",
            ),
            # Agregar labels GMM a datos
            Node(
                func=add_cluster_labels_to_data,
                inputs=["X_clustering", "labels_gmm"],
                outputs="datos_con_clusters_gmm",
                name="add_clusters_gmm_node",
            ),
        ]
    )
