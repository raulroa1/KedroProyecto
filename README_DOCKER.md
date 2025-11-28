# 🐳 Guía de Docker y Airflow para el Proyecto Kedro

Esta guía explica cómo usar Docker y Airflow para orquestar los pipelines de Kedro.

---

## 📋 Requisitos Previos

- Docker Desktop instalado y ejecutándose
- Docker Compose (incluido en Docker Desktop)
- Al menos 4GB de RAM disponible
- Al menos 10GB de espacio en disco

---

## 🚀 Inicio Rápido

### 1. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto (ya está creado) o ajusta las variables según necesites:

```bash
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
```

### 2. Construir las Imágenes Docker

```powershell
docker-compose build
```

Este comando construye la imagen de Airflow con todas las dependencias de Kedro.

### 3. Inicializar Airflow

```powershell
docker-compose up airflow-init
```

Esto inicializa la base de datos de Airflow y crea el usuario administrador.

### 4. Iniciar los Servicios

```powershell
docker-compose up -d
```

Esto inicia:
- **PostgreSQL**: Base de datos para Airflow
- **Airflow Webserver**: Interfaz web en http://localhost:8080
- **Airflow Scheduler**: Planificador de tareas

### 5. Acceder a la Interfaz de Airflow

Abre tu navegador y ve a:
```
http://localhost:8080
```

**Credenciales:**
- Usuario: `airflow`
- Contraseña: `airflow`

---

## 📊 DAGs Disponibles

Una vez que Airflow esté ejecutándose, verás los siguientes DAGs:

### 1. `kedro_data_processing`
- **Descripción**: Ejecuta el pipeline de procesamiento de datos
- **Frecuencia**: Diaria
- **Tareas**: 
  - `run_data_processing_pipeline`: Ejecuta el pipeline
  - `verify_data_processing_output`: Verifica que se generaron los outputs

### 2. `kedro_data_science`
- **Descripción**: Ejecuta pipelines de ML (Clustering, Clasificación, Regresión)
- **Frecuencia**: Diaria
- **Tareas**:
  - `clustering_tasks`: Grupo de tareas de clustering
  - `classification_tasks`: Grupo de tareas de clasificación
  - `regression_tasks`: Grupo de tareas de regresión

### 3. `kedro_reporting`
- **Descripción**: Genera reportes de todos los pipelines
- **Frecuencia**: Diaria
- **Tareas**:
  - `run_reporting_pipeline`: Ejecuta el pipeline de reporting
  - `verify_reporting_output`: Verifica que se generaron los reportes

### 4. `kedro_master_pipeline`
- **Descripción**: DAG maestro que orquesta todos los pipelines en orden
- **Frecuencia**: Diaria
- **Flujo**: Data Processing → Data Science → Reporting

---

## 🛠️ Comandos Útiles

### Ver Logs de los Servicios

```powershell
# Ver todos los logs
docker-compose logs

# Ver logs de un servicio específico
docker-compose logs airflow-scheduler
docker-compose logs airflow-webserver

# Seguir logs en tiempo real
docker-compose logs -f airflow-scheduler
```

### Detener los Servicios

```powershell
docker-compose down
```

### Detener y Eliminar Volúmenes (⚠️ Elimina datos)

```powershell
docker-compose down -v
```

### Reiniciar un Servicio Específico

```powershell
docker-compose restart airflow-scheduler
```

### Ejecutar Comandos en el Contenedor

```powershell
# Ejecutar un comando de Kedro
docker-compose exec airflow-webserver kedro run --pipeline data_processing

# Abrir una shell en el contenedor
docker-compose exec airflow-webserver bash
```

### Ver Estado de los Contenedores

```powershell
docker-compose ps
```

---

## 📁 Estructura de Directorios

```
proyecto-kedro/
├── dags/                          # DAGs de Airflow
│   ├── kedro_data_processing_dag.py
│   ├── kedro_data_science_dag.py
│   ├── kedro_reporting_dag.py
│   └── kedro_master_dag.py
├── logs/                          # Logs de Airflow y Kedro
│   ├── airflow.log
│   ├── kedro.log
│   └── dag_*.log
├── plugins/                       # Plugins de Airflow
├── Dockerfile.airflow             # Imagen de Airflow
├── docker-compose.yml             # Configuración de servicios
├── .env                          # Variables de entorno
└── airflow_logging_config.py     # Configuración de logging
```

---

## 🔍 Verificación de Logs

### Logs de Airflow

Los logs se encuentran en:
- **Dentro del contenedor**: `/opt/airflow/logs/`
- **En tu máquina**: `./logs/`

### Ver Logs de un DAG Específico

1. Ve a la interfaz de Airflow (http://localhost:8080)
2. Selecciona el DAG
3. Haz clic en "Graph View"
4. Selecciona una tarea
5. Haz clic en "Log"

### Ver Logs desde la Terminal

```powershell
# Ver logs de Airflow
Get-Content logs\airflow.log -Tail 50

# Ver logs de Kedro
Get-Content logs\kedro.log -Tail 50

# Ver logs de un DAG específico
Get-Content logs\dag_kedro_data_processing.log -Tail 50
```

---

## ⚙️ Configuración Avanzada

### Cambiar Frecuencia de Ejecución

Edita el archivo del DAG (ej: `dags/kedro_data_processing_dag.py`) y modifica:

```python
schedule_interval=timedelta(hours=6),  # Cada 6 horas
# o
schedule_interval='0 0 * * *',  # Diario a medianoche (cron)
```

### Agregar Notificaciones por Email

En `docker-compose.yml`, agrega configuración SMTP:

```yaml
environment:
  AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
  AIRFLOW__SMTP__SMTP_STARTTLS: 'true'
  AIRFLOW__SMTP__SMTP_SSL: 'false'
  AIRFLOW__SMTP__SMTP_USER: tu_email@gmail.com
  AIRFLOW__SMTP__SMTP_PASSWORD: tu_contraseña
  AIRFLOW__SMTP__SMTP_PORT: 587
  AIRFLOW__SMTP__SMTP_MAIL_FROM: tu_email@gmail.com
```

### Ajustar Recursos

En `docker-compose.yml`, puedes agregar límites de recursos:

```yaml
services:
  airflow-scheduler:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## 🐛 Solución de Problemas

### Problema: "Port 8080 is already in use"

**Solución**: Cambia el puerto en `docker-compose.yml`:
```yaml
ports:
  - "8081:8080"  # Cambiar 8080 por 8081
```

### Problema: "Permission denied" en logs

**Solución**: Ajusta permisos:
```powershell
# En Windows, asegúrate de que el directorio logs existe y tiene permisos
New-Item -ItemType Directory -Force -Path logs
```

### Problema: DAGs no aparecen en la interfaz

**Solución**:
1. Verifica que los archivos están en `dags/`
2. Revisa los logs del scheduler: `docker-compose logs airflow-scheduler`
3. Reinicia el scheduler: `docker-compose restart airflow-scheduler`

### Problema: Error al ejecutar pipelines de Kedro

**Solución**:
1. Verifica que el proyecto Kedro está correctamente copiado al contenedor
2. Ejecuta manualmente: `docker-compose exec airflow-webserver kedro run`
3. Revisa los logs: `docker-compose logs airflow-scheduler`

---

## 📝 Notas Importantes

- **Persistencia de Datos**: Los datos se guardan en volúmenes Docker. Si eliminas los volúmenes (`docker-compose down -v`), perderás los datos.
- **Logs**: Los logs se rotan automáticamente (máximo 10MB por archivo, 5 backups).
- **Rendimiento**: El primer build puede tardar varios minutos. Las ejecuciones posteriores serán más rápidas.
- **Recursos**: Asegúrate de tener suficientes recursos asignados a Docker Desktop (mínimo 4GB RAM, 2 CPUs).

---

## ✅ Checklist de Verificación

- [ ] Docker Desktop instalado y ejecutándose
- [ ] Imagen construida exitosamente (`docker-compose build`)
- [ ] Airflow inicializado (`docker-compose up airflow-init`)
- [ ] Servicios ejecutándose (`docker-compose ps`)
- [ ] Interfaz web accesible (http://localhost:8080)
- [ ] DAGs visibles en la interfaz
- [ ] Logs generándose correctamente

---

¡Listo! Con esta configuración podrás orquestar todos tus pipelines de Kedro usando Airflow. 🚀

