from kedro.pipeline import Node, Pipeline

from .nodes import (
    limpiar_productos,
    extraer_peso_y_limpiar_productos_v3,
    normalizar_productos,
    normalizar_datos,
    preprocesar_ventas,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=limpiar_productos,
                inputs="matriz_venta",
                outputs="productos_limpios",
                name="limpiar_productos_node",
            ),
            Node(
                func=extraer_peso_y_limpiar_productos_v3,
                inputs="productos_limpios",
                outputs="productos_con_peso",
                name="extraer_peso_productos_node",
            ),
            Node(
                func=normalizar_productos,
                inputs="productos_con_peso",
                outputs="productos_normalizados",
                name="normalizar_productos_node",
            ),
            Node(
                func=normalizar_datos,
                inputs="productos_normalizados",
                outputs="datos_normalizados",
                name="normalizar_datos_node",
            ),
            Node(
                func=preprocesar_ventas,
                inputs="datos_normalizados",
                outputs="ventas_preprocesadas",
                name="preprocesar_ventas_node",
            ),
        ]
    )
