# 📊 Análisis de la Fase de Clustering - Pipeline Data Science

## 🎯 Objetivo General
La fase de clustering agrupa los datos de ventas en clusters (grupos) similares sin necesidad de etiquetas previas. Esto permite descubrir patrones ocultos en los datos y segmentar productos/comunas por comportamiento de ventas.

---

## 📥 Entrada del Pipeline de Clustering

### Dataset de Entrada: `ventas_preprocesadas`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/03_primary/ventas_preprocesadas.parquet`
- **Origen**: Output del pipeline de `data_processing`
- **Columnas**: FECHA, PRODUCTO, COMUNA, CANTIDAD, MES, PRODUCTO_ID, COMUNA_ID, VENTA_MES_ANTERIOR, AUMENTA, VENTA_CLASE

---

## 🔄 Flujo del Pipeline de Clustering (15 Nodos)

### **NODO 1: `prepare_clustering_data_node`**
**Función**: `prepare_clustering_data()`

#### ¿Qué hace?
- **Prepara los datos** para algoritmos de clustering
- **Selecciona features numéricas** automáticamente
- **Muestrea los datos** si son muy grandes (máximo 10,000 muestras por defecto)

#### Transformaciones específicas:
1. **Muestreo** (si `len(data) > max_samples`):
   - Toma una muestra aleatoria de `max_samples` filas
   - Usa `random_state=42` para reproducibilidad
   - **Propósito**: Reducir tiempo de cómputo en datasets grandes

2. **Selección de features**:
   - Si `feature_columns` es `None`, selecciona automáticamente todas las columnas numéricas
   - Filtra solo columnas de tipo numérico (`np.number`)
   - **Resultado**: DataFrame solo con features numéricas

#### Output: `X_clustering`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/04_feature/X_clustering.parquet`
- **Estado**: DataFrame con solo features numéricas, listo para clustering

---

### **NODO 2: `scale_features_node`**
**Función**: `scale_features()`

#### ¿Qué hace?
- **Estandariza las features** usando `StandardScaler`
- **Normaliza** todas las variables a media 0 y desviación estándar 1
- **Guarda el scaler** para poder transformar nuevos datos

#### Transformaciones específicas:
1. **Estandarización**:
   - Aplica `StandardScaler.fit_transform()` a todas las features
   - Fórmula: `(x - media) / desviación_estándar`
   - **Propósito**: Todas las features tienen la misma escala (importante para clustering)

2. **Preservación de estructura**:
   - Mantiene nombres de columnas originales
   - Mantiene índices originales

#### Outputs:
- **`X_clustering_scaled`**: Features estandarizadas (ParquetDataset)
- **`scaler_clustering`**: Objeto StandardScaler guardado (PickleDataset)
  - **Ubicación**: `data/06_models/scaler_clustering.pickle`
  - **Uso**: Para estandarizar nuevos datos en producción

---

### **NODOS 3-5: K-Means Clustering**

#### **NODO 3: `train_kmeans_node`**
**Función**: `train_kmeans()`

##### ¿Qué hace?
- **Entrena un modelo K-Means** con 3 clusters por defecto
- **Agrupa los datos** en k grupos basándose en distancia euclidiana

##### Parámetros:
- `n_clusters`: 3 (por defecto)
- `random_state`: 42 (reproducibilidad)
- `n_init`: 10 (intentos de inicialización)

##### Outputs:
- **`modelo_kmeans`**: Modelo KMeans entrenado (PickleDataset)
- **`labels_kmeans`**: Etiquetas de cluster asignadas a cada muestra (PickleDataset)

#### **NODO 4: `evaluate_kmeans_node`**
**Función**: `evaluate_clustering()`

##### ¿Qué hace?
- **Evalúa la calidad** del clustering K-Means
- **Calcula métricas** de evaluación

##### Métricas calculadas:
1. **Silhouette Score**: Mide qué tan bien separados están los clusters
   - Rango: -1 a 1 (más alto = mejor)
   - Mide cohesión interna y separación entre clusters

2. **Davies-Bouldin Score**: Mide la separación entre clusters
   - Rango: 0 a ∞ (más bajo = mejor)
   - Considera la distancia entre clusters y su tamaño

3. **Calinski-Harabasz Score**: Ratio de varianza entre clusters vs dentro de clusters
   - Rango: 0 a ∞ (más alto = mejor)
   - También conocido como "Variance Ratio Criterion"

4. **Información adicional**:
   - `n_clusters`: Número de clusters encontrados
   - `n_noise`: Puntos de ruido (solo para DBSCAN)

##### Output: `metricas_kmeans`
- **Tipo**: `pickle.PickleDataset`
- **Ubicación**: `data/08_reporting/metricas_kmeans.pickle`
- **Contenido**: Diccionario con todas las métricas

#### **NODO 5: `add_clusters_kmeans_node`**
**Función**: `add_cluster_labels_to_data()`

##### ¿Qué hace?
- **Agrega las etiquetas de cluster** al DataFrame original
- **Crea una nueva columna** con el número de cluster asignado

##### Transformaciones:
- Agrega columna `cluster` (o nombre personalizado) con los labels
- Asegura que los tamaños coincidan (trunca si es necesario)

##### Output: `datos_con_clusters_kmeans`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/05_model_input/datos_con_clusters_kmeans.parquet`
- **Estado**: Datos originales + columna de cluster K-Means

---

### **NODOS 6-8: DBSCAN Clustering**

#### **NODO 6: `train_dbscan_node`**
**Función**: `train_dbscan()`

##### ¿Qué hace?
- **Entrena un modelo DBSCAN** (Density-Based Spatial Clustering)
- **Encuentra clusters** basándose en densidad de puntos
- **Identifica puntos de ruido** (outliers)

##### Parámetros:
- `eps`: 0.5 (distancia máxima entre puntos del mismo cluster)
- `min_samples`: 5 (mínimo de puntos para formar un cluster)

##### Características especiales:
- **No requiere especificar número de clusters** (lo encuentra automáticamente)
- **Puede identificar puntos de ruido** (label = -1)
- **Útil para encontrar outliers** y clusters de forma irregular

##### Outputs:
- **`modelo_dbscan`**: Modelo DBSCAN entrenado (PickleDataset)
- **`labels_dbscan`**: Etiquetas de cluster (incluye -1 para ruido) (PickleDataset)

#### **NODO 7: `evaluate_dbscan_node`**
**Función**: `evaluate_clustering()`

##### ¿Qué hace?
- **Evalúa DBSCAN** con las mismas métricas que K-Means
- **Filtra puntos de ruido** antes de calcular métricas
- **Reporta número de clusters y puntos de ruido**

##### Output: `metricas_dbscan`
- **Tipo**: `pickle.PickleDataset`
- **Ubicación**: `data/08_reporting/metricas_dbscan.pickle`

#### **NODO 8: `add_clusters_dbscan_node`**
**Función**: `add_cluster_labels_to_data()`

##### Output: `datos_con_clusters_dbscan`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/05_model_input/datos_con_clusters_dbscan.parquet`
- **Nota**: Puede contener puntos con label -1 (ruido/outliers)

---

### **NODOS 9-11: Agglomerative Clustering**

#### **NODO 9: `train_agglomerative_node`**
**Función**: `train_agglomerative_with_fallback()`

##### ¿Qué hace?
- **Entrena Agglomerative Clustering** (clustering jerárquico)
- **Usa PCA** para reducir dimensiones y optimizar memoria
- **Tiene fallback** si n_clusters es inválido

##### Características especiales:
1. **Reducción de dimensiones**:
   - Aplica PCA para reducir a máximo 10 componentes
   - **Propósito**: Optimizar uso de memoria (Agglomerative es costoso)

2. **Fallback**:
   - Si `n_clusters` es None o inválido, usa `fallback_n_clusters=3`
   - **Propósito**: Robustez ante parámetros incorrectos

##### Parámetros:
- `n_clusters`: 3 (por defecto, con fallback)
- `linkage`: "ward" (método de enlace jerárquico)

##### Outputs:
- **`modelo_agglomerative`**: Modelo AgglomerativeClustering (PickleDataset)
- **`labels_agglomerative`**: Etiquetas de cluster (PickleDataset)

#### **NODO 10: `evaluate_agglomerative_node`**
**Función**: `evaluate_clustering()`

##### Output: `metricas_agglomerative`
- **Tipo**: `pickle.PickleDataset`
- **Ubicación**: `data/08_reporting/metricas_agglomerative.pickle`

#### **NODO 11: `add_clusters_agglomerative_node`**
**Función**: `add_cluster_labels_to_data()`

##### Output: `datos_con_clusters_agglomerative`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/05_model_input/datos_con_clusters_agglomerative.parquet`

---

### **NODOS 12-14: Gaussian Mixture Model (GMM)**

#### **NODO 12: `train_gmm_node`**
**Función**: `train_gmm_with_fallback()`

##### ¿Qué hace?
- **Entrena un modelo GMM** (Gaussian Mixture Model)
- **Modela los datos** como una mezcla de distribuciones gaussianas
- **Tiene fallback** si n_components es inválido

##### Características:
- **Modelo probabilístico**: Asigna probabilidades de pertenencia a cada cluster
- **Más flexible** que K-Means (puede modelar clusters elípticos)
- **Fallback**: Usa `fallback_n_clusters=3` si n_components es inválido

##### Parámetros:
- `n_components`: 3 (número de distribuciones gaussianas)
- `random_state`: 42
- `n_init`: 10 (intentos de inicialización)

##### Outputs:
- **`modelo_gmm`**: Modelo GaussianMixture (PickleDataset)
- **`labels_gmm`**: Etiquetas de cluster (PickleDataset)

#### **NODO 13: `evaluate_gmm_node`**
**Función**: `evaluate_clustering()`

##### Output: `metricas_gmm`
- **Tipo**: `pickle.PickleDataset`
- **Ubicación**: `data/08_reporting/metricas_gmm.pickle`

#### **NODO 14: `add_clusters_gmm_node`**
**Función**: `add_cluster_labels_to_data()`

##### Output: `datos_con_clusters_gmm`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/05_model_input/datos_con_clusters_gmm.parquet`

---

## 📊 Resumen de Algoritmos de Clustering

### Comparación de Algoritmos:

| Algoritmo | Tipo | Número de Clusters | Ventajas | Desventajas |
|-----------|------|-------------------|----------|-------------|
| **K-Means** | Particional | Fijo (3) | Rápido, simple, escalable | Requiere especificar k, asume clusters esféricos |
| **DBSCAN** | Basado en densidad | Automático | Encuentra outliers, clusters irregulares | Sensible a parámetros eps y min_samples |
| **Agglomerative** | Jerárquico | Fijo (3) | Crea dendrograma, flexible | Costoso en memoria, usa PCA para optimizar |
| **GMM** | Probabilístico | Fijo (3) | Modela clusters elípticos, probabilidades | Más lento que K-Means, más parámetros |

---

## 🎯 Propósito de Cada Algoritmo en el Contexto de Ventas:

1. **K-Means**: Segmentación básica de productos/comunas por volumen de ventas
2. **DBSCAN**: Identificar outliers y grupos de comportamiento anómalo
3. **Agglomerative**: Entender jerarquías y relaciones entre segmentos
4. **GMM**: Modelar distribuciones complejas de ventas con probabilidades

---

## 📝 Observaciones Técnicas:

1. **Estandarización crítica**: Todos los algoritmos usan datos estandarizados (importante para que todas las features tengan el mismo peso)

2. **Muestreo**: Si hay más de 10,000 muestras, se muestrea para optimizar tiempo de cómputo

3. **Manejo de ruido**: DBSCAN puede identificar puntos de ruido (label = -1), que se filtran en las métricas

4. **Optimización de memoria**: Agglomerative usa PCA para reducir dimensiones antes de clustering

5. **Reproducibilidad**: Todos los algoritmos usan `random_state=42` para resultados consistentes

---

## 🔍 Flujo Visual del Pipeline de Clustering:

```
ventas_preprocesadas
    ↓
[prepare_clustering_data] → X_clustering
    ↓
[scale_features] → X_clustering_scaled + scaler_clustering
    ↓
    ├─→ [train_kmeans] → modelo_kmeans + labels_kmeans
    │       ↓
    │   [evaluate_clustering] → metricas_kmeans
    │       ↓
    │   [add_cluster_labels] → datos_con_clusters_kmeans
    │
    ├─→ [train_dbscan] → modelo_dbscan + labels_dbscan
    │       ↓
    │   [evaluate_clustering] → metricas_dbscan
    │       ↓
    │   [add_cluster_labels] → datos_con_clusters_dbscan
    │
    ├─→ [train_agglomerative] → modelo_agglomerative + labels_agglomerative
    │       ↓
    │   [evaluate_clustering] → metricas_agglomerative
    │       ↓
    │   [add_cluster_labels] → datos_con_clusters_agglomerative
    │
    └─→ [train_gmm] → modelo_gmm + labels_gmm
            ↓
        [evaluate_clustering] → metricas_gmm
            ↓
        [add_cluster_labels] → datos_con_clusters_gmm
```

---

## ✅ Outputs Finales del Pipeline de Clustering:

1. **4 Modelos entrenados** (PickleDataset):
   - `modelo_kmeans.pickle`
   - `modelo_dbscan.pickle`
   - `modelo_agglomerative.pickle`
   - `modelo_gmm.pickle`

2. **4 Sets de labels** (PickleDataset):
   - `labels_kmeans.pickle`
   - `labels_dbscan.pickle`
   - `labels_agglomerative.pickle`
   - `labels_gmm.pickle`

3. **4 Sets de métricas** (PickleDataset):
   - `metricas_kmeans.pickle`
   - `metricas_dbscan.pickle`
   - `metricas_agglomerative.pickle`
   - `metricas_gmm.pickle`

4. **4 Datasets con clusters agregados** (ParquetDataset):
   - `datos_con_clusters_kmeans.parquet`
   - `datos_con_clusters_dbscan.parquet`
   - `datos_con_clusters_agglomerative.parquet`
   - `datos_con_clusters_gmm.parquet`

5. **Scaler guardado** (PickleDataset):
   - `scaler_clustering.pickle` (para estandarizar nuevos datos)

