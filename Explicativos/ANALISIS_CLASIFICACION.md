# 📊 Análisis de la Fase de Clasificación - Pipeline Data Science

## 🎯 Objetivo General
La fase de clasificación entrena modelos de machine learning para predecir la clase de venta (baja/media/alta) basándose en features históricas y características de productos/comunas. Se evalúan múltiples algoritmos para encontrar el mejor modelo.

---

## 📥 Entrada del Pipeline de Clasificación

### Dataset de Entrada: `ventas_preprocesadas`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/03_primary/ventas_preprocesadas.parquet`
- **Origen**: Output del pipeline de `data_processing`
- **Columnas**: FECHA, PRODUCTO, COMUNA, CANTIDAD, MES, PRODUCTO_ID, COMUNA_ID, VENTA_MES_ANTERIOR, AUMENTA, VENTA_CLASE

---

## 🔄 Flujo del Pipeline de Clasificación (5 Nodos)

### **NODO 1: `preparar_datos_clasificacion_node`**
**Función**: `preparar_datos_clasificacion()`

#### ¿Qué hace?
- **Prepara los datos** para modelos de clasificación
- **Crea features históricas** adicionales (promedios móviles, delta)
- **Crea la variable objetivo** (VENTA_CLASE) basada en rangos de cantidad
- **Muestrea los datos** si son muy grandes (máximo 10,000 muestras por defecto)

#### Transformaciones específicas:

1. **Muestreo** (si `len(df) > max_samples`):
   - Toma una muestra aleatoria de `max_samples` filas
   - Usa `random_state=42` para reproducibilidad
   - **Propósito**: Reducir tiempo de cómputo

2. **Limpieza básica**:
   - Normaliza nombres de columnas (elimina espacios)
   - Convierte FECHA a datetime
   - Elimina filas con fechas inválidas

3. **Variables base**:
   - `MES`: Mes extraído de la fecha (1-12)
   - `PRODUCTO_ID`: Código numérico del producto
   - `COMUNA_ID`: Código numérico de la comuna

4. **Variables históricas agrupadas**:
   - `VENTA_MES_ANTERIOR`: Cantidad vendida el mes anterior (usando `shift(1)`)
   - `PROM_3_MESES`: Promedio móvil de 3 meses
   - `PROM_6_MESES`: Promedio móvil de 6 meses
   - `DELTA_MES`: Cambio porcentual respecto al mes anterior
     - Fórmula: `(CANTIDAD - VENTA_MES_ANTERIOR) / VENTA_MES_ANTERIOR`
     - Solo calculado cuando `VENTA_MES_ANTERIOR != 0`

5. **Variable objetivo (VENTA_CLASE)**:
   - Clasificación multiclase basada en rangos de CANTIDAD:
     - **Clase 0 (baja)**: CANTIDAD < 10
     - **Clase 1 (media)**: 10 ≤ CANTIDAD < 50
     - **Clase 2 (alta)**: CANTIDAD ≥ 50
   - Usa `pd.cut()` con bins: `[0, 10, 50, max(CANTIDAD)]`

6. **Limpieza final**:
   - Elimina filas con nulos en `VENTA_MES_ANTERIOR` o `VENTA_CLASE`
   - Convierte todas las features a enteros (int32)
   - Rellena nulos con 0 antes de convertir

7. **Separación X e y**:
   - **X**: Features numéricas (MES, PRODUCTO_ID, COMUNA_ID, VENTA_MES_ANTERIOR, PROM_3_MESES, PROM_6_MESES, DELTA_MES)
   - **y**: Variable objetivo (VENTA_CLASE codificada como int8: 0, 1, 2)

#### Outputs:
- **`X_clf`**: Features para clasificación (ParquetDataset)
- **`y_clf`**: Variable objetivo (ParquetDataset)

---

### **NODO 2: `pre_proceso_clf_node`**
**Función**: `pre_proceso_clf()`

#### ¿Qué hace?
- **Ajusta el tamaño** de X e y a un valor fijo (56884 filas)
- **Propósito**: Asegurar consistencia en el tamaño de los datos

#### ⚠️ Observación:
- **Hardcode de 56884 filas**: Esto parece ser un valor específico de un dataset anterior
- **Riesgo**: Si los datos tienen menos de 56884 filas, puede causar errores
- **Recomendación**: Debería ser dinámico o removerse si no es necesario

#### Transformaciones:
- Trunca X e y a las primeras 56884 filas
- Resetea índices

#### Outputs:
- **`X_clf_proc`**: Features preprocesadas (ParquetDataset)
- **`y_clf_proc`**: Variable objetivo preprocesada (ParquetDataset)

---

### **NODO 3: `dividir_datos_clf_node`**
**Función**: `dividir_datos_clf()`

#### ¿Qué hace?
- **Divide los datos** en conjuntos de entrenamiento y prueba
- **Usa train_test_split** de sklearn

#### Parámetros:
- `test_size`: 0.2 (20% para prueba, 80% para entrenamiento)
- `random_state`: 42 (reproducibilidad)

#### Outputs:
- **`X_train_clf`**: Features de entrenamiento (ParquetDataset)
- **`X_test_clf`**: Features de prueba (ParquetDataset)
- **`y_train_clf`**: Variable objetivo de entrenamiento (ParquetDataset)
- **`y_test_clf`**: Variable objetivo de prueba (ParquetDataset)

---

### **NODO 4: `entrenar_modelos_clasificacion_node`**
**Función**: `entrenar_modelos_clasificacion_cv()`

#### ¿Qué hace?
- **Entrena 5 modelos de clasificación** diferentes
- **Usa GridSearchCV** para optimizar hiperparámetros
- **Usa validación cruzada** (StratifiedKFold con 5 folds)
- **Guarda los modelos** entrenados en archivos .pkl

#### Modelos entrenados:

1. **LogisticRegression**:
   - Parámetros optimizados: `C` [0.01, 0.1, 1, 10]
   - `max_iter`: 1000

2. **RandomForestClassifier**:
   - Parámetros optimizados: `n_estimators` [50, 100, 200]
   - `random_state`: 42

3. **GradientBoostingClassifier**:
   - Parámetros optimizados: `n_estimators` [50, 100]
   - `random_state`: 42

4. **SVC (Support Vector Classifier)**:
   - Parámetros optimizados: `C` [0.1, 1, 10]
   - `probability=True` (para obtener probabilidades)

5. **KNeighborsClassifier (KNN)**:
   - Parámetros optimizados: `n_neighbors` [3, 5, 7]

#### Proceso:
- Para cada modelo:
  1. Crea GridSearchCV con el modelo y parámetros
  2. Entrena con validación cruzada (5 folds estratificados)
  3. Selecciona el mejor modelo según el score CV
  4. Guarda el mejor modelo en `models_clf/{nombre}.pkl`
  5. Almacena: mejor modelo, mejores parámetros, score CV

#### Output: `resultados_clf`
- **Tipo**: `pickle.PickleDataset`
- **Ubicación**: `data/06_models/resultados_clf.pickle`
- **Contenido**: Diccionario con 5 modelos entrenados, cada uno con:
  - `modelo`: Mejor estimador encontrado
  - `mejor_params`: Mejores hiperparámetros
  - `score_cv`: Score promedio de validación cruzada

---

### **NODO 5: `evaluar_modelos_clasificacion_node`**
**Función**: `evaluar_modelos_clasificacion()`

#### ¿Qué hace?
- **Evalúa cada modelo** en el conjunto de prueba
- **Calcula métricas** de clasificación multiclase
- **Compara el rendimiento** de todos los modelos

#### Métricas calculadas (para cada modelo):

1. **Accuracy (Precisión)**:
   - Proporción de predicciones correctas
   - Rango: 0 a 1 (más alto = mejor)

2. **Precision (Precisión)**:
   - Promedio ponderado de precisión por clase
   - Mide qué tan precisas son las predicciones positivas
   - Rango: 0 a 1 (más alto = mejor)

3. **Recall (Sensibilidad)**:
   - Promedio ponderado de recall por clase
   - Mide qué tan bien encuentra las clases positivas
   - Rango: 0 a 1 (más alto = mejor)

4. **F1 Score**:
   - Media armónica de Precision y Recall
   - Balance entre precisión y sensibilidad
   - Rango: 0 a 1 (más alto = mejor)

#### Proceso:
- Para cada modelo en `resultados_clf`:
  1. Hace predicciones en `X_test`
  2. Calcula las 4 métricas usando `y_test` y `y_pred`
  3. Almacena las métricas en un diccionario

#### Output: `metricas_clf`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/08_reporting/metricas_clf.parquet`
- **Formato**: DataFrame con filas = modelos, columnas = métricas
- **Contenido**:
  ```
  Modelo              | Accuracy | Precision | Recall | F1
  --------------------|----------|-----------|--------|----
  LogisticRegression  |   0.XX   |   0.XX    |  0.XX  | 0.XX
  RandomForest        |   0.XX   |   0.XX    |  0.XX  | 0.XX
  GradientBoosting    |   0.XX   |   0.XX    |  0.XX  | 0.XX
  SVC                 |   0.XX   |   0.XX    |  0.XX  | 0.XX
  KNN                 |   0.XX   |   0.XX    |  0.XX  | 0.XX
  ```

---

## 📊 Resumen del Pipeline de Clasificación

### Flujo Visual:
```
ventas_preprocesadas
    ↓
[preparar_datos_clasificacion] → X_clf, y_clf
    ↓
[pre_proceso_clf] → X_clf_proc, y_clf_proc (ajusta a 56884 filas)
    ↓
[dividir_datos_clf] → X_train_clf, X_test_clf, y_train_clf, y_test_clf
    ↓
[entrenar_modelos_clasificacion_cv] → resultados_clf (5 modelos entrenados)
    ↓
[evaluar_modelos_clasificacion] → metricas_clf (DataFrame con métricas)
```

### Features Utilizadas (7 features):
1. `MES`: Mes de la venta (1-12)
2. `PRODUCTO_ID`: Código numérico del producto
3. `COMUNA_ID`: Código numérico de la comuna
4. `VENTA_MES_ANTERIOR`: Cantidad vendida el mes anterior
5. `PROM_3_MESES`: Promedio móvil de 3 meses
6. `PROM_6_MESES`: Promedio móvil de 6 meses
7. `DELTA_MES`: Cambio porcentual respecto al mes anterior

### Variable Objetivo:
- **VENTA_CLASE**: Clasificación multiclase (3 clases)
  - **Clase 0 (baja)**: CANTIDAD < 10
  - **Clase 1 (media)**: 10 ≤ CANTIDAD < 50
  - **Clase 2 (alta)**: CANTIDAD ≥ 50

### Modelos Evaluados (5):
1. **LogisticRegression**: Regresión logística (lineal)
2. **RandomForest**: Bosque aleatorio (ensemble)
3. **GradientBoosting**: Boosting con gradiente (ensemble)
4. **SVC**: Máquinas de vectores de soporte (kernel)
5. **KNN**: K-Vecinos más cercanos (basado en instancias)

### Técnicas Utilizadas:
- **GridSearchCV**: Búsqueda exhaustiva de hiperparámetros
- **StratifiedKFold**: Validación cruzada estratificada (5 folds)
- **Métricas multiclase**: Accuracy, Precision, Recall, F1 (weighted average)

---

## ⚠️ Observaciones y Mejoras Potenciales

### Problemas Identificados:

1. **Hardcode de 56884 filas en `pre_proceso_clf`**:
   - **Problema**: Valor fijo que puede no ser apropiado para todos los datasets
   - **Riesgo**: Si hay menos filas, puede causar errores o pérdida de datos
   - **Recomendación**: Hacer dinámico o eliminar si no es necesario

2. **Guardado de modelos en ruta relativa**:
   - **Problema**: `joblib.dump(gs.best_estimator_, f"{output_path}/{nombre}.pkl")` usa ruta relativa
   - **Riesgo**: Puede no guardarse en la ubicación esperada
   - **Recomendación**: Usar el catálogo de Kedro para guardar modelos

3. **Muestreo a 10,000 filas**:
   - **Problema**: Puede perder información valiosa
   - **Recomendación**: Considerar aumentar o hacer configurable

### Mejoras Sugeridas:

1. **Balanceo de clases**: Las clases pueden estar desbalanceadas (baja tiene más registros)
2. **Feature engineering adicional**: Podrían agregarse más features relevantes
3. **Métricas adicionales**: Matriz de confusión, reporte de clasificación por clase
4. **Validación de datos**: Verificar que todas las features estén presentes

---

## 📝 Outputs Finales del Pipeline de Clasificación:

1. **Features y targets**:
   - `X_clf.parquet`: Features preparadas
   - `y_clf.parquet`: Variable objetivo
   - `X_clf_proc.parquet`: Features preprocesadas
   - `y_clf_proc.parquet`: Target preprocesado

2. **Datos divididos**:
   - `X_train_clf.parquet`: Features de entrenamiento
   - `X_test_clf.parquet`: Features de prueba
   - `y_train_clf.parquet`: Target de entrenamiento
   - `y_test_clf.parquet`: Target de prueba

3. **Modelos entrenados**:
   - `resultados_clf.pickle`: Diccionario con 5 modelos entrenados
   - Modelos guardados en `models_clf/` (si se usa ruta relativa)

4. **Métricas de evaluación**:
   - `metricas_clf.parquet`: DataFrame con métricas de todos los modelos

---

## 🎯 Propósito de la Clasificación:

La clasificación permite:
- **Predecir el nivel de ventas** (baja/media/alta) basándose en características históricas
- **Identificar patrones** que influyen en el volumen de ventas
- **Tomar decisiones** sobre inventario, promociones, o distribución basadas en predicciones
- **Comparar modelos** para seleccionar el mejor algoritmo para este problema específico

