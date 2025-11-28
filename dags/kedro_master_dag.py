"""
DAG Maestro que orquesta todos los pipelines de Kedro
Ejecuta: Data Processing -> Data Science -> Reporting
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Argumentos por defecto
default_args = {
    'owner': 'kedro_admin',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'start_date': days_ago(1),
}

# Definir el DAG maestro
dag = DAG(
    'kedro_master_pipeline',
    default_args=default_args,
    description='DAG maestro que ejecuta todos los pipelines de Kedro en orden',
    schedule_interval=timedelta(days=1),  # Ejecutar diariamente
    catchup=False,
    max_active_runs=1,  # Solo una ejecución a la vez
    tags=['kedro', 'master', 'orchestration'],
)

# Tarea inicial: Log de inicio
start_task = BashOperator(
    task_id='start_master_pipeline',
    bash_command='echo "Iniciando ejecución del pipeline maestro de Kedro - $(date)"',
    dag=dag,
)

# Trigger para Data Processing
trigger_data_processing = TriggerDagRunOperator(
    task_id='trigger_data_processing',
    trigger_dag_id='kedro_data_processing',
    wait_for_completion=True,
    poke_interval=30,
    dag=dag,
)

# Trigger para Data Science (después de data_processing)
trigger_data_science = TriggerDagRunOperator(
    task_id='trigger_data_science',
    trigger_dag_id='kedro_data_science',
    wait_for_completion=True,
    poke_interval=30,
    dag=dag,
)

# Trigger para Reporting (después de data_science)
trigger_reporting = TriggerDagRunOperator(
    task_id='trigger_reporting',
    trigger_dag_id='kedro_reporting',
    wait_for_completion=True,
    poke_interval=30,
    dag=dag,
)

# Tarea final: Log de finalización
end_task = BashOperator(
    task_id='end_master_pipeline',
    bash_command='echo "Pipeline maestro completado exitosamente - $(date)"',
    dag=dag,
)

# Definir el flujo de ejecución
start_task >> trigger_data_processing >> trigger_data_science >> trigger_reporting >> end_task

