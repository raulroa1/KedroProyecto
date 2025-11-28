"""
DAG para ejecutar el pipeline de Reporting de Kedro
Genera reportes de todos los pipelines ejecutados
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Argumentos por defecto
default_args = {
    'owner': 'reporting_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Definir el DAG
dag = DAG(
    'kedro_reporting',
    default_args=default_args,
    description='Pipeline de reporting que genera análisis de todos los pipelines',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['kedro', 'reporting', 'analytics'],
)

# Tarea para ejecutar el pipeline de reporting
run_reporting = BashOperator(
    task_id='run_reporting_pipeline',
    bash_command='cd /opt/airflow/kedro_project && kedro run --pipeline reporting',
    dag=dag,
)

# Tarea para verificar que todos los reportes se generaron
# Nota: Los archivos están versionados (versioned: true), así que se guardan en subdirectorios con timestamps
verify_reports = BashOperator(
    task_id='verify_reporting_output',
    bash_command='''
    cd /opt/airflow/kedro_project && \
    (test -f data/08_reporting/analisis_pipeline_data_processing.pickle || \
     (test -d data/08_reporting/analisis_pipeline_data_processing.pickle && \
      find data/08_reporting/analisis_pipeline_data_processing.pickle -name "*.pickle" -type f | head -1 | grep -q .)) && \
    (test -f data/08_reporting/analisis_pipeline_clustering.pickle || \
     (test -d data/08_reporting/analisis_pipeline_clustering.pickle && \
      find data/08_reporting/analisis_pipeline_clustering.pickle -name "*.pickle" -type f | head -1 | grep -q .)) && \
    (test -f data/08_reporting/analisis_pipeline_clasificacion.pickle || \
     (test -d data/08_reporting/analisis_pipeline_clasificacion.pickle && \
      find data/08_reporting/analisis_pipeline_clasificacion.pickle -name "*.pickle" -type f | head -1 | grep -q .)) && \
    (test -f data/08_reporting/analisis_pipeline_regresion.pickle || \
     (test -d data/08_reporting/analisis_pipeline_regresion.pickle && \
      find data/08_reporting/analisis_pipeline_regresion.pickle -name "*.pickle" -type f | head -1 | grep -q .)) && \
    echo "Todos los reportes generados exitosamente" || exit 1
    ''',
    dag=dag,
)

# Definir dependencias
run_reporting >> verify_reports

