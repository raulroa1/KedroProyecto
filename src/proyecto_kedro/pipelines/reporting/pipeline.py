from kedro.pipeline import Node, Pipeline

from .nodes import (
    generar_analisis_pipeline,
    generar_analisis_clustering,
    generar_analisis_clasificacion,
    generar_analisis_regresion,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline de reporting que genera análisis de data_processing y clustering"""
    return Pipeline(
        [
            Node(
                func=generar_analisis_pipeline,
                inputs=[
                    "matriz_venta",
                    "productos_limpios",
                    "productos_con_peso",
                    "productos_normalizados",
                    "datos_normalizados",
                    "ventas_preprocesadas",
                ],
                outputs="analisis_pipeline_data_processing",
                name="generar_analisis_pipeline_node",
            ),
            Node(
                func=generar_analisis_clustering,
                inputs=[
                    "X_clustering",
                    "X_clustering_scaled",
                    "metricas_kmeans",
                    "metricas_dbscan",
                    "metricas_agglomerative",
                    "metricas_gmm",
                    "labels_kmeans",
                    "labels_dbscan",
                    "labels_agglomerative",
                    "labels_gmm",
                    "datos_con_clusters_kmeans",
                    "datos_con_clusters_dbscan",
                    "datos_con_clusters_agglomerative",
                    "datos_con_clusters_gmm",
                ],
                outputs="analisis_pipeline_clustering",
                name="generar_analisis_clustering_node",
            ),
            Node(
                func=generar_analisis_clasificacion,
                inputs=[
                    "X_clf",
                    "y_clf",
                    "X_train_clf",
                    "X_test_clf",
                    "y_train_clf",
                    "y_test_clf",
                    "resultados_clf",
                    "metricas_clf",
                ],
                outputs="analisis_pipeline_clasificacion",
                name="generar_analisis_clasificacion_node",
            ),
            Node(
                func=generar_analisis_regresion,
                inputs=[
                    "X_reg",
                    "y_reg",
                    "X_train_reg",
                    "X_test_reg",
                    "y_train_reg",
                    "y_test_reg",
                    "resultados_reg",
                    "metricas_reg",
                ],
                outputs="analisis_pipeline_regresion",
                name="generar_analisis_regresion_node",
            ),
        ]
    )
