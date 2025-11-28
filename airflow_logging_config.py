"""
Configuración de logging para Airflow y Kedro
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Directorio de logs
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configuración de formato de logs
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_kedro_logging():
    """Configurar logging para Kedro"""
    kedro_logger = logging.getLogger("kedro")
    kedro_logger.setLevel(logging.INFO)
    
    # Handler para archivo
    kedro_file_handler = RotatingFileHandler(
        LOG_DIR / "kedro.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    kedro_file_handler.setLevel(logging.INFO)
    kedro_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    # Handler para consola
    kedro_console_handler = logging.StreamHandler(sys.stdout)
    kedro_console_handler.setLevel(logging.INFO)
    kedro_console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    kedro_logger.addHandler(kedro_file_handler)
    kedro_logger.addHandler(kedro_console_handler)
    
    return kedro_logger

def setup_airflow_logging():
    """Configurar logging para Airflow"""
    airflow_logger = logging.getLogger("airflow")
    airflow_logger.setLevel(logging.INFO)
    
    # Handler para archivo
    airflow_file_handler = RotatingFileHandler(
        LOG_DIR / "airflow.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    airflow_file_handler.setLevel(logging.INFO)
    airflow_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    airflow_logger.addHandler(airflow_file_handler)
    
    return airflow_logger

def setup_dag_logging(dag_id: str):
    """Configurar logging específico para un DAG"""
    dag_logger = logging.getLogger(f"airflow.dag.{dag_id}")
    dag_logger.setLevel(logging.INFO)
    
    # Handler para archivo específico del DAG
    dag_file_handler = RotatingFileHandler(
        LOG_DIR / f"dag_{dag_id}.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3
    )
    dag_file_handler.setLevel(logging.INFO)
    dag_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    dag_logger.addHandler(dag_file_handler)
    
    return dag_logger

# Configurar logging por defecto
if __name__ != "__main__":
    setup_kedro_logging()
    setup_airflow_logging()

