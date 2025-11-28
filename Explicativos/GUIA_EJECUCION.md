# 📚 Guía de Ejecución del Proyecto Kedro

Esta guía te mostrará paso a paso cómo ejecutar el proyecto completo desde la terminal.

---

## 📋 Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Configuración Inicial](#configuración-inicial)
3. [Ejecución de Pipelines](#ejecución-de-pipelines)
4. [Visualización de Resultados](#visualización-de-resultados)
5. [Comandos Útiles](#comandos-útiles)

---

## 🔧 Requisitos Previos

- Python 3.8 o superior
- Git (opcional, para clonar el repositorio)
- Terminal (PowerShell, CMD, o Git Bash en Windows)

---

## 🚀 Configuración Inicial

### Paso 1: Navegar al Directorio del Proyecto

```powershell
cd "C:\Users\raulr\OneDrive\Escritorio\Proyecto definitivo\proyecto-kedro"
cd "C:Ruta de tu archivo"
```

### Paso 2: Crear y Activar el Entorno Virtual

**Crear el entorno virtual:**
```powershell
python -m venv venv
```

**Activar el entorno virtual:**

En PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

En CMD:
```cmd
venv\Scripts\activate.bat
```

**Verificar que el entorno está activo:**
```powershell
python --version
```

Deberías ver la versión de Python y el prefijo `(venv)` en tu terminal.

### Paso 3: Instalar Dependencias

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Verificar instalación de Kedro:**
```powershell
kedro --version
```

---

## 🔄 Ejecución de Pipelines

### Opción A: Ejecutar Todo el Proyecto Completo

Este comando ejecuta todos los pipelines en orden:
- `data_processing`
- `data_science` (clustering, clasificación, regresión)
- `reporting`

```powershell
kedro run
```

### Opción B: Ejecutar Pipelines Individuales

#### 1. Pipeline de Data Processing

```powershell
kedro run --pipeline data_processing
```

**Qué hace:**
- Limpia y normaliza los datos de `matriz-venta.csv`
- Crea features nuevas (MES, PRODUCTO_ID, COMUNA_ID, VENTA_MES_ANTERIOR, AUMENTA, VENTA_CLASE)
- Genera el dataset `ventas_preprocesadas`

#### 2. Pipeline de Clustering

```powershell
kedro run --pipeline data_science --node prepare_clustering_data_node
kedro run --pipeline data_science --node scale_features_node
kedro run --pipeline data_science --node train_kmeans_node
kedro run --pipeline data_science --node train_dbscan_node
kedro run --pipeline data_science --node train_agglomerative_node
kedro run --pipeline data_science --node train_gmm_node
```

**O ejecutar todo el clustering de una vez:**
```powershell
kedro run --pipeline data_science --tags clustering
```

**Qué hace:**
- Prepara datos para clustering
- Estandariza features
- Entrena 4 algoritmos: K-Means, DBSCAN, Agglomerative, GMM
- Evalúa y guarda métricas de cada algoritmo

#### 3. Pipeline de Clasificación

```powershell
kedro run --pipeline data_science --tags classification
```

**Qué hace:**
- Prepara datos para clasificación (predecir VENTA_CLASE)
- Divide en entrenamiento y prueba
- Entrena 5 modelos: LogisticRegression, RandomForest, GradientBoosting, SVC, KNN
- Evalúa y guarda métricas

#### 4. Pipeline de Regresión

```powershell
kedro run --pipeline data_science --tags regression
```

**Qué hace:**
- Prepara datos para regresión (predecir CANTIDAD)
- Divide en entrenamiento y prueba
- Entrena 5 modelos: LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting
- Evalúa y guarda métricas

#### 5. Pipeline de Reporting

```powershell
kedro run --pipeline reporting
```

**Qué hace:**
- Genera reportes estructurados de todos los pipelines
- Crea archivos pickle con análisis completos:
  - `analisis_pipeline_data_processing.pickle`
  - `analisis_pipeline_clustering.pickle`
  - `analisis_pipeline_clasificacion.pickle`
  - `analisis_pipeline_regresion.pickle`

---

## 📊 Visualización de Resultados

### Opción 1: Ejecutar Notebooks en Jupyter

#### Paso 1: Iniciar Jupyter Notebook

```powershell
cd "C:\Users\raulr\OneDrive\Escritorio\Proyecto definitivo\proyecto-kedro"
.\venv\Scripts\python.exe -m jupyter notebook notebooks
```

O usar el script creado:
```powershell
.\iniciar_jupyter.bat
```

#### Paso 2: Abrir y Ejecutar los Notebooks

1. Se abrirá tu navegador con la interfaz de Jupyter
2. Selecciona el kernel: **"Python 3 (ipykernel)"**
3. Ejecuta los notebooks en este orden:
   - `01_Reporte_Data_Processing.ipynb`
   - `02_Reporte_Clustering.ipynb`
   - `03_Reporte_Clasificacion.ipynb`
   - `04_Reporte_Regresion.ipynb`
   - `05_Resumen_General.ipynb`

**Para ejecutar cada celda:**
- `Shift + Enter`: Ejecutar celda y avanzar
- `Ctrl + Enter`: Ejecutar celda sin avanzar
- `Alt + Enter`: Ejecutar celda y crear nueva

### Opción 2: Ver Datos Directamente desde Python

Puedes cargar y explorar los datos directamente:

```powershell
python
```

```python
from pathlib import Path
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import pandas as pd

# Inicializar proyecto
project_path = Path.cwd()
bootstrap_project(project_path)
session = KedroSession.create(project_path=project_path)
catalog = session.load_context().catalog

# Cargar datos
ventas = catalog.load('ventas_preprocesadas')
print(ventas.head())
print(ventas.shape)

# Cargar reportes
analisis_dp = catalog.load('analisis_pipeline_data_processing')
print(analisis_dp.keys())

session.close()
```

---

## 🛠️ Comandos Útiles

### Ver Estructura del Proyecto

```powershell
tree /F /A
```

### Ver Catálogo de Datos Disponibles

```powershell
kedro catalog list
```

### Ver Información de un Dataset Específico

```powershell
kedro catalog describe ventas_preprocesadas
```

### Limpiar Caché de Kedro

Si encuentras problemas con datos desactualizados:

```powershell
kedro catalog clear
```

### Ver Parámetros del Proyecto

```powershell
kedro pipeline list
```

### Ejecutar un Nodo Específico

```powershell
kedro run --node limpiar_productos_node
```

### Ver Logs Detallados

```powershell
kedro run --verbose
```

### Verificar Instalación de Dependencias

```powershell
pip list
```

### Actualizar Dependencias

```powershell
pip install --upgrade -r requirements.txt
```

---

## 📁 Estructura de Archivos Generados

Después de ejecutar los pipelines, encontrarás:

```
proyecto-kedro/
├── data/
│   ├── 01_raw/
│   │   └── matriz-venta.csv
│   ├── 02_intermediate/
│   │   ├── productos_limpios.parquet
│   │   ├── productos_con_peso.parquet
│   │   ├── productos_normalizados.parquet
│   │   └── datos_normalizados.parquet
│   ├── 03_primary/
│   │   └── ventas_preprocesadas.parquet
│   ├── 04_feature/
│   │   ├── X_clf.parquet
│   │   ├── X_reg.parquet
│   │   ├── X_clustering.parquet
│   │   └── ...
│   ├── 05_model_input/
│   │   ├── X_train_clf.parquet
│   │   ├── X_test_clf.parquet
│   │   └── ...
│   ├── 06_models/
│   │   ├── modelo_kmeans.pickle
│   │   ├── modelo_dbscan.pickle
│   │   ├── resultados_clf.pickle
│   │   └── ...
│   ├── 07_model_output/
│   │   ├── metricas_clf.parquet
│   │   ├── metricas_reg.parquet
│   │   └── ...
│   └── 08_reporting/
│       ├── analisis_pipeline_data_processing.pickle
│       ├── analisis_pipeline_clustering.pickle
│       ├── analisis_pipeline_clasificacion.pickle
│       └── analisis_pipeline_regresion.pickle
└── notebooks/
    ├── 01_Reporte_Data_Processing.ipynb
    ├── 02_Reporte_Clustering.ipynb
    ├── 03_Reporte_Clasificacion.ipynb
    ├── 04_Reporte_Regresion.ipynb
    └── 05_Resumen_General.ipynb
```

---

## 🔍 Verificación de Ejecución Exitosa

### Verificar que los Pipelines se Ejecutaron Correctamente

```powershell
# Verificar que existen los archivos de reporte
Test-Path "data\08_reporting\analisis_pipeline_data_processing.pickle"
Test-Path "data\08_reporting\analisis_pipeline_clustering.pickle"
Test-Path "data\08_reporting\analisis_pipeline_clasificacion.pickle"
Test-Path "data\08_reporting\analisis_pipeline_regresion.pickle"
```

Todos deberían retornar `True`.

### Verificar Modelos Entrenados

```powershell
# Verificar modelos de clustering
Test-Path "data\06_models\modelo_kmeans.pickle"
Test-Path "data\06_models\modelo_dbscan.pickle"
Test-Path "data\06_models\modelo_agglomerative.pickle"
Test-Path "data\06_models\modelo_gmm.pickle"

# Verificar resultados de clasificación y regresión
Test-Path "data\06_models\resultados_clf.pickle"
Test-Path "data\06_models\resultados_reg.pickle"
```

---

## ⚠️ Solución de Problemas Comunes

### Problema 1: "No module named 'kedro'"

**Solución:**
```powershell
pip install kedro
```

### Problema 2: "Dataset not found in catalog"

**Solución:**
```powershell
kedro catalog clear
kedro run
```

### Problema 3: "Permission denied" al guardar archivos

**Solución:**
- Cierra cualquier programa que pueda estar usando los archivos
- Ejecuta PowerShell como administrador
- O simplemente vuelve a ejecutar el pipeline

### Problema 4: Error al ejecutar notebooks

**Solución:**
1. Verifica que el kernel sea "Python 3 (ipykernel)"
2. Asegúrate de que el entorno virtual esté activo
3. Reinstala jupyter si es necesario:
   ```powershell
   pip install jupyter notebook
   ```

### Problema 5: Errores de importación en notebooks

**Solución:**
Verifica que el path del proyecto sea correcto. En los notebooks, el código usa:
```python
project_path = Path.cwd().parent  # Subir un nivel desde notebooks/
```

Si ejecutas desde la raíz del proyecto, cambia a:
```python
project_path = Path.cwd()
```

---

## 📝 Flujo de Trabajo Recomendado

### Primera Ejecución Completa

1. **Activar entorno virtual:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Ejecutar todo el proyecto:**
   ```powershell
   kedro run
   ```

3. **Verificar que todo se ejecutó:**
   ```powershell
   kedro catalog list
   ```

4. **Abrir Jupyter para visualizar:**
   ```powershell
   .\venv\Scripts\python.exe -m jupyter notebook notebooks
   ```

### Ejecuciones Posteriores

Si solo quieres actualizar un pipeline específico:

```powershell
# Solo actualizar data processing
kedro run --pipeline data_processing

# Solo actualizar clustering
kedro run --pipeline data_science --tags clustering

# Solo actualizar reportes
kedro run --pipeline reporting
```

---

## 🎯 Resumen de Comandos Esenciales

```powershell
# 1. Activar entorno
.\venv\Scripts\Activate.ps1

# 2. Ejecutar proyecto completo
kedro run

# 3. Abrir Jupyter
.\venv\Scripts\python.exe -m jupyter notebook notebooks

# 4. Ver catálogo
kedro catalog list

# 5. Limpiar caché (si hay problemas)
kedro catalog clear
```

---

## ✅ Checklist de Verificación

Antes de considerar el proyecto completamente ejecutado, verifica:

- [ ] Entorno virtual creado y activado
- [ ] Todas las dependencias instaladas
- [ ] Pipeline `data_processing` ejecutado exitosamente
- [ ] Pipeline `data_science` (clustering) ejecutado exitosamente
- [ ] Pipeline `data_science` (clasificación) ejecutado exitosamente
- [ ] Pipeline `data_science` (regresión) ejecutado exitosamente
- [ ] Pipeline `reporting` ejecutado exitosamente
- [ ] Todos los archivos de reporte generados en `data/08_reporting/`
- [ ] Notebooks ejecutados y visualizaciones generadas

---

## 📞 Notas Adicionales

- **Tiempo de ejecución:** El proyecto completo tarda aproximadamente 1-2 minutos
- **Espacio en disco:** Asegúrate de tener al menos 500 MB libres
- **Memoria:** El proyecto usa muestreo automático para datasets grandes (máximo 10,000 muestras)

---

¡Listo! Con esta guía podrás ejecutar el proyecto completo desde la terminal. 🚀

