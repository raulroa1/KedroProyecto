"""
DAG para ejecutar los pipelines de Data Science de Kedro
(Clustering, Clasificación y Regresión)
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Argumentos por defecto
default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Definir el DAG
dag = DAG(
    'kedro_data_science',
    default_args=default_args,
    description='Pipelines de Data Science: Clustering, Clasificación y Regresión',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['kedro', 'data_science', 'ml', 'clustering', 'classification', 'regression'],
)

# Grupo de tareas para Clustering
with TaskGroup('clustering_tasks', dag=dag) as clustering_group:
    run_clustering = BashOperator(
        task_id='run_clustering_pipeline',
        bash_command='''
        cd /opt/airflow/kedro_project && \
        kedro run --pipeline data_science --from-nodes prepare_clustering_data_node --to-nodes add_clusters_gmm_node
        ''',
        dag=dag,
    )
    
    verify_clustering = BashOperator(
        task_id='verify_clustering_output',
        bash_command='test -f /opt/airflow/kedro_project/data/06_models/modelo_kmeans.pickle && echo "Clustering completado" || exit 1',
        dag=dag,
    )
    
    run_clustering >> verify_clustering

# Grupo de tareas para Clasificación
with TaskGroup('classification_tasks', dag=dag) as classification_group:
    run_classification = BashOperator(
        task_id='run_classification_pipeline',
        bash_command='''
        cd /opt/airflow/kedro_project && \
        kedro run --pipeline data_science --from-nodes preparar_datos_clasificacion_node --to-nodes evaluar_modelos_clasificacion_node
        ''',
        dag=dag,
    )
    
    verify_classification = BashOperator(
        task_id='verify_classification_output',
        bash_command='test -f /opt/airflow/kedro_project/data/06_models/resultados_clf.pickle && echo "Clasificación completada" || exit 1',
        dag=dag,
    )
    
    run_classification >> verify_classification

# Grupo de tareas para Regresión
with TaskGroup('regression_tasks', dag=dag) as regression_group:
    run_regression = BashOperator(
        task_id='run_regression_pipeline',
        bash_command='''
        cd /opt/airflow/kedro_project && \
        kedro run --pipeline data_science --from-nodes preparar_datos_regresion_node --to-nodes evaluar_modelos_regresion_node
        ''',
        dag=dag,
    )
    
    verify_regression = BashOperator(
        task_id='verify_regression_output',
        bash_command='test -f /opt/airflow/kedro_project/data/06_models/resultados_reg.pickle && echo "Regresión completada" || exit 1',
        dag=dag,
    )
    
    run_regression >> verify_regression

# Las tareas pueden ejecutarse en paralelo ya que no dependen entre sí
# clustering_group, classification_group, regression_group

