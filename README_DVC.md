# 📦 Guía de DVC (Data Version Control)

Esta guía explica cómo usar DVC en el proyecto Kedro para versionar datos, modelos y reproducir experimentos.

---

## 📋 ¿Qué es DVC?

DVC (Data Version Control) es un sistema de control de versiones para datos y modelos de Machine Learning. Permite:

- ✅ **Versionar datos grandes** sin guardarlos en Git
- ✅ **Reproducir experimentos** de forma exacta
- ✅ **Gestionar pipelines** de ML de forma declarativa
- ✅ **Comparar métricas** entre diferentes ejecuciones
- ✅ **Compartir datos y modelos** de forma eficiente

---

## 🚀 Comandos Básicos

### Ver el estado de los pipelines

```powershell
python -m dvc status
```

Muestra qué stages tienen cambios en sus dependencias o outputs.

### Ver el DAG de dependencias

```powershell
python -m dvc dag
```

Muestra la estructura de dependencias entre los stages.

### Ejecutar un stage específico

```powershell
# Ejecutar solo el stage de data_processing
python -m dvc repro data_processing

# Ejecutar con dry-run (simulación)
python -m dvc repro --dry data_processing
```

### Ejecutar todos los stages

```powershell
# Ejecutar todos los stages en orden
python -m dvc repro

# Ejecutar solo los stages que han cambiado
python -m dvc repro --force
```

### Ver métricas

```powershell
# Ver todas las métricas
python -m dvc metrics show

# Comparar métricas entre ejecuciones
python -m dvc metrics diff
```

### Ver gráficos

```powershell
# Ver todos los plots
python -m dvc plots show

# Comparar plots entre ejecuciones
python -m dvc plots diff
```

---

## 📊 Stages Definidos en `dvc.yaml`

El proyecto tiene 5 stages configurados:

### 1. `data_processing`
- **Comando**: `kedro run --pipeline data_processing`
- **Dependencias**: 
  - `data/01_raw/matriz-venta.csv`
  - Código del pipeline `data_processing`
- **Outputs**: 
  - `data/03_primary/ventas_preprocesadas.parquet`
  - `metrics/data_processing.json`
  - `plots/data_processing/`

### 2. `clustering`
- **Comando**: `kedro run --pipeline data_science --from-nodes prepare_clustering_data_node --to-nodes add_clusters_gmm_node`
- **Dependencias**: 
  - `data/03_primary/ventas_preprocesadas.parquet`
  - Código del pipeline `data_science`
- **Outputs**: 
  - Modelos: `modelo_kmeans.pickle`, `modelo_dbscan.pickle`, `modelo_agglomerative.pickle`, `modelo_gmm.pickle`
  - Datos con clusters
  - Métricas y plots

### 3. `classification`
- **Comando**: `kedro run --pipeline data_science --from-nodes preparar_datos_clasificacion_node --to-nodes evaluar_modelos_clasificacion_node`
- **Outputs**: 
  - `data/06_models/resultados_clf.pickle`
  - `data/07_model_output/metricas_clf.parquet`
  - Métricas y plots

### 4. `regression`
- **Comando**: `kedro run --pipeline data_science --from-nodes preparar_datos_regresion_node --to-nodes evaluar_modelos_regresion_node`
- **Outputs**: 
  - `data/06_models/resultados_reg.pickle`
  - `data/07_model_output/metricas_reg.parquet`
  - Métricas y plots

### 5. `reporting`
- **Comando**: `kedro run --pipeline reporting`
- **Dependencias**: 
  - Datos procesados y modelos de los stages anteriores
- **Outputs**: 
  - Análisis de todos los pipelines
  - `metrics/reporting.json`

---

## 🔄 Flujo de Trabajo con DVC

### 1. Hacer cambios en el código o datos

```powershell
# Editar código o datos
# ...

# Ver qué ha cambiado
python -m dvc status
```

### 2. Ejecutar los pipelines afectados

```powershell
# Ejecutar solo los stages que necesitan actualizarse
python -m dvc repro

# O ejecutar un stage específico
python -m dvc repro data_processing
```

### 3. Verificar métricas y resultados

```powershell
# Ver métricas actuales
python -m dvc metrics show

# Comparar con ejecución anterior
python -m dvc metrics diff
```

### 4. Guardar cambios (si usas Git)

```powershell
# DVC crea archivos .dvc que deben committearse
git add .dvc data/*.dvc metrics/*.json
git commit -m "Actualizar pipeline con nuevos datos"
```

---

## 🔗 Integración con Airflow

Actualmente, los DAGs de Airflow ejecutan Kedro directamente. Si quieres usar DVC desde Airflow, puedes modificar los DAGs para ejecutar:

```python
# En lugar de:
kedro run --pipeline data_processing

# Usar:
dvc repro data_processing
```

Esto te dará:
- ✅ Versionado automático de outputs
- ✅ Tracking de métricas
- ✅ Reproducibilidad garantizada

---

## 📁 Estructura de Archivos DVC

```
proyecto-kedro/
├── .dvc/                    # Configuración de DVC
│   ├── config              # Configuración principal
│   └── tmp/                # Archivos temporales
├── .dvcignore              # Archivos a ignorar (similar a .gitignore)
├── dvc.yaml                # Definición de stages y pipelines
├── data/                    # Datos versionados
│   └── *.dvc              # Archivos de metadatos (uno por dataset)
├── metrics/                # Métricas en JSON
└── plots/                  # Gráficos y visualizaciones
```

---

## ⚙️ Configuración Avanzada

### Configurar almacenamiento remoto

Para compartir datos con el equipo o hacer backup:

```powershell
# Ejemplo con Google Drive
python -m dvc remote add -d myremote gdrive://tu-id-de-google-drive

# Ejemplo con S3
python -m dvc remote add -d myremote s3://tu-bucket/dvc-storage

# Subir datos
python -m dvc push

# Descargar datos
python -m dvc pull
```

### Ver historial de cambios

```powershell
# Ver historial de un archivo específico
python -m dvc diff HEAD~1 data/03_primary/ventas_preprocesadas.parquet
```

---

## 🐛 Solución de Problemas

### Problema: "Path is ignored by .dvcignore"

**Solución**: Revisa el archivo `.dvcignore` y asegúrate de que los archivos definidos como outputs en `dvc.yaml` no estén siendo ignorados.

### Problema: "Stage dependencies changed"

**Solución**: Esto es normal cuando cambias código o datos. Ejecuta `python -m dvc repro` para actualizar los outputs.

### Problema: "Outputs were not found"

**Solución**: Los outputs no existen aún. Ejecuta el stage correspondiente:
```powershell
python -m dvc repro nombre_del_stage
```

---

## 📝 Notas Importantes

- **DVC no reemplaza a Git**: DVC trabaja junto con Git. Los metadatos (archivos `.dvc`) se versionan en Git, pero los datos grandes se almacenan por separado.

- **Reproducibilidad**: DVC garantiza que puedas reproducir exactamente los mismos resultados si tienes las mismas dependencias.

- **Métricas y Plots**: Los archivos JSON en `metrics/` y los plots en `plots/` se versionan automáticamente.

- **Integración con Kedro**: DVC ejecuta los pipelines de Kedro, por lo que necesitas tener Kedro configurado correctamente.

---

## ✅ Checklist de Verificación

- [x] DVC instalado (`python -m dvc version`)
- [x] DVC inicializado (`python -m dvc status` funciona)
- [x] `dvc.yaml` configurado con todos los stages
- [x] `.dvcignore` configurado correctamente
- [x] DVC agregado a `requirements.txt`

---

## 🔗 Recursos Adicionales

- [Documentación oficial de DVC](https://dvc.org/doc)
- [Guía de DVC con Kedro](https://dvc.org/doc/use-cases/versioning-data-and-model-files)
- [Comandos de DVC](https://dvc.org/doc/command-reference)

---

¡Listo! Con DVC configurado, puedes versionar tus datos y modelos de forma eficiente. 🚀

