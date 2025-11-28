"""
DAG para ejecutar el pipeline de Data Processing de Kedro
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Definir el DAG
dag = DAG(
    'kedro_data_processing',
    default_args=default_args,
    description='Pipeline de procesamiento de datos con Kedro',
    schedule_interval=timedelta(days=1),  # Ejecutar diariamente
    catchup=False,
    tags=['kedro', 'data_processing', 'etl'],
)

def log_execution(context):
    """Función para logging de ejecución"""
    logger.info(f"Ejecutando pipeline data_processing")
    logger.info(f"DAG Run ID: {context.get('dag_run').run_id}")
    logger.info(f"Task Instance: {context.get('task_instance')}")

# Tarea para ejecutar el pipeline de data_processing
run_data_processing = BashOperator(
    task_id='run_data_processing_pipeline',
    bash_command='cd /opt/airflow/kedro_project && kedro run --pipeline data_processing',
    dag=dag,
    on_success_callback=log_execution,
)

# Tarea para verificar que el pipeline se ejecutó correctamente
verify_output = BashOperator(
    task_id='verify_data_processing_output',
    bash_command='test -f /opt/airflow/kedro_project/data/03_primary/ventas_preprocesadas.parquet && echo "Pipeline ejecutado exitosamente" || exit 1',
    dag=dag,
)

# Definir dependencias
run_data_processing >> verify_output

