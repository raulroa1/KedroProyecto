# 📊 Análisis de la Fase de Regresión - Pipeline Data Science

## 🎯 Objetivo General
La fase de regresión entrena modelos de machine learning para predecir la cantidad de ventas (valor continuo) basándose en features históricas y características de productos/comunas. Se evalúan múltiples algoritmos de regresión para encontrar el mejor modelo.

---

## 📥 Entrada del Pipeline de Regresión

### Dataset de Entrada: `ventas_preprocesadas`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/03_primary/ventas_preprocesadas.parquet`
- **Origen**: Output del pipeline de `data_processing`
- **Columnas**: FECHA, PRODUCTO, COMUNA, CANTIDAD, MES, PRODUCTO_ID, COMUNA_ID, VENTA_MES_ANTERIOR, AUMENTA, VENTA_CLASE

---

## 🔄 Flujo del Pipeline de Regresión (5 Nodos)

### **NODO 1: `preparar_datos_regresion_node`**
**Función**: `preparar_datos_regresion()`

#### ¿Qué hace?
- **Prepara los datos** para modelos de regresión
- **Crea features históricas** adicionales (promedios móviles, delta)
- **Define la variable objetivo** como CANTIDAD (valor continuo)
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

5. **Limpieza final**:
   - Elimina filas con nulos en `VENTA_MES_ANTERIOR`

6. **Separación X e y**:
   - **X**: Features numéricas (MES, PRODUCTO_ID, COMUNA_ID, VENTA_MES_ANTERIOR, PROM_3_MESES, PROM_6_MESES, DELTA_MES)
   - **y**: Variable objetivo (CANTIDAD - valor continuo)

#### Outputs:
- **`X_reg`**: Features para regresión (ParquetDataset)
- **`y_reg`**: Variable objetivo (ParquetDataset)

---

### **NODO 2: `pre_proceso_rg_node`**
**Función**: `pre_proceso_rg()`

#### ¿Qué hace?
- **Ajusta el tamaño** de X e y a un valor fijo (56884 filas por defecto)
- **Propósito**: Asegurar consistencia en el tamaño de los datos

#### ⚠️ Observación:
- **Hardcode de 56884 filas**: Esto parece ser un valor específico de un dataset anterior
- **Riesgo**: Si los datos tienen menos de 56884 filas, puede causar errores o pérdida de datos
- **Recomendación**: Debería ser dinámico o removerse si no es necesario

#### Transformaciones:
- Trunca X e y a las primeras n_filas filas
- Resetea índices

#### Outputs:
- **`X_reg_proc`**: Features preprocesadas (ParquetDataset)
- **`y_reg_proc`**: Variable objetivo preprocesada (ParquetDataset)

---

### **NODO 3: `dividir_datos_reg_node`**
**Función**: `dividir_datos_reg()`

#### ¿Qué hace?
- **Divide los datos** en conjuntos de entrenamiento y prueba
- **Usa train_test_split** de sklearn

#### Parámetros:
- `test_size`: 0.2 (20% para prueba, 80% para entrenamiento)
- `random_state`: 42 (reproducibilidad)

#### Outputs:
- **`X_train_reg`**: Features de entrenamiento (ParquetDataset)
- **`X_test_reg`**: Features de prueba (ParquetDataset)
- **`y_train_reg`**: Variable objetivo de entrenamiento (ParquetDataset)
- **`y_test_reg`**: Variable objetivo de prueba (ParquetDataset)

---

### **NODO 4: `entrenar_modelos_regresion_node`**
**Función**: `entrenar_modelos_regresion_cv()`

#### ¿Qué hace?
- **Entrena 5 modelos de regresión** diferentes
- **Usa GridSearchCV** para optimizar hiperparámetros (cuando aplica)
- **Usa validación cruzada** (KFold con 5 folds)
- **Guarda los modelos** entrenados en archivos .pkl

#### Modelos entrenados:

1. **LinearRegression**:
   - Modelo lineal básico
   - Sin hiperparámetros a optimizar

2. **Ridge**:
   - Regresión con regularización L2
   - Parámetros optimizados: `alpha` [0.1, 1.0, 10.0]

3. **Lasso**:
   - Regresión con regularización L1
   - Parámetros optimizados: `alpha` [0.01, 0.1, 1.0]

4. **RandomForestRegressor**:
   - Bosque aleatorio para regresión
   - Parámetros optimizados: `n_estimators` [50, 100]
   - `random_state`: 42, `n_jobs`: -1

5. **GradientBoostingRegressor**:
   - Boosting con gradiente para regresión
   - Parámetros optimizados: `n_estimators` [50, 100]
   - `random_state`: 42

#### Proceso:
- Para cada modelo:
  1. Si tiene parámetros, crea GridSearchCV con validación cruzada (5 folds)
  2. Si no tiene parámetros, entrena directamente
  3. Selecciona el mejor modelo según el score CV (si aplica)
  4. Guarda el mejor modelo en `models_reg/{nombre}.pkl`
  5. Almacena: mejor modelo, mejores parámetros, score CV

#### Output: `resultados_reg`
- **Tipo**: `pickle.PickleDataset`
- **Ubicación**: `data/06_models/resultados_reg.pickle`
- **Contenido**: Diccionario con 5 modelos entrenados, cada uno con:
  - `modelo`: Mejor estimador encontrado
  - `mejor_params`: Mejores hiperparámetros
  - `score_cv`: Score promedio de validación cruzada (si aplica)

---

### **NODO 5: `evaluar_modelos_regresion_node`**
**Función**: `evaluar_modelos_regresion()`

#### ¿Qué hace?
- **Evalúa cada modelo** en el conjunto de prueba
- **Calcula métricas** de regresión
- **Compara el rendimiento** de todos los modelos

#### Métricas calculadas (para cada modelo):

1. **RMSE (Root Mean Squared Error)**:
   - Raíz cuadrada del error cuadrático medio
   - Mide la desviación promedio de las predicciones
   - Rango: 0 a ∞ (más bajo = mejor)
   - Unidades: Mismas que la variable objetivo

2. **MAE (Mean Absolute Error)**:
   - Error absoluto medio
   - Mide el error promedio en valor absoluto
   - Rango: 0 a ∞ (más bajo = mejor)
   - Unidades: Mismas que la variable objetivo
   - Menos sensible a outliers que RMSE

3. **R² (Coefficient of Determination)**:
   - Coeficiente de determinación
   - Mide qué tan bien el modelo explica la varianza
   - Rango: -∞ a 1 (más alto = mejor)
   - 1 = predicción perfecta, 0 = modelo no mejor que la media, <0 = peor que la media

#### Proceso:
- Para cada modelo en `resultados_reg`:
  1. Hace predicciones en `X_test`
  2. Calcula las 3 métricas usando `y_test` y `y_pred`
  3. Almacena las métricas en un diccionario

#### Output: `metricas_reg`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/08_reporting/metricas_reg.parquet`
- **Formato**: DataFrame con filas = modelos, columnas = métricas
- **Contenido**:
  ```
  Modelo              | RMSE  | MAE   | R2
  --------------------|-------|-------|-------
  LinearRegression    |  X.XX |  X.XX | 0.XX
  Ridge               |  X.XX |  X.XX | 0.XX
  Lasso               |  X.XX |  X.XX | 0.XX
  RandomForest        |  X.XX |  X.XX | 0.XX
  GradientBoosting    |  X.XX |  X.XX | 0.XX
  ```

---

## 📊 Resumen del Pipeline de Regresión

### Flujo Visual:
```
ventas_preprocesadas
    ↓
[preparar_datos_regresion] → X_reg, y_reg
    ↓
[pre_proceso_rg] → X_reg_proc, y_reg_proc (ajusta a n_filas filas)
    ↓
[dividir_datos_reg] → X_train_reg, X_test_reg, y_train_reg, y_test_reg
    ↓
[entrenar_modelos_regresion_cv] → resultados_reg (5 modelos entrenados)
    ↓
[evaluar_modelos_regresion] → metricas_reg (DataFrame con métricas)
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
- **CANTIDAD**: Valor continuo (número de unidades vendidas)
- **Tipo**: Regresión (predicción de valores numéricos)

### Modelos Evaluados (5):
1. **LinearRegression**: Regresión lineal (sin regularización)
2. **Ridge**: Regresión con regularización L2
3. **Lasso**: Regresión con regularización L1
4. **RandomForestRegressor**: Bosque aleatorio (ensemble)
5. **GradientBoostingRegressor**: Boosting con gradiente (ensemble)

### Técnicas Utilizadas:
- **GridSearchCV**: Búsqueda exhaustiva de hiperparámetros
- **KFold**: Validación cruzada (5 folds)
- **Métricas de regresión**: RMSE, MAE, R²

---

## ⚠️ Observaciones y Mejoras Potenciales

### Problemas Identificados:

1. **Hardcode de 56884 filas en `pre_proceso_rg`**:
   - **Problema**: Valor fijo que puede no ser apropiado para todos los datasets
   - **Riesgo**: Si hay menos filas, puede causar errores o pérdida de datos
   - **Recomendación**: Hacer dinámico o eliminar si no es necesario

2. **Guardado de modelos en ruta relativa**:
   - **Problema**: `joblib.dump(best_model, f"{output_path}/{nombre}.pkl")` usa ruta relativa
   - **Riesgo**: Puede no guardarse en la ubicación esperada
   - **Recomendación**: Usar el catálogo de Kedro para guardar modelos

3. **Muestreo a 10,000 filas**:
   - **Problema**: Puede perder información valiosa
   - **Recomendación**: Considerar aumentar o hacer configurable

4. **y_reg como Series**:
   - **Problema**: `y_reg` se retorna como Series, pero el catálogo espera DataFrame para Parquet
   - **Recomendación**: Convertir a DataFrame antes de retornar

### Mejoras Sugeridas:

1. **Feature engineering adicional**: Podrían agregarse más features relevantes
2. **Escalado de features**: Considerar estandarizar/normalizar features antes de entrenar
3. **Métricas adicionales**: MAPE (Mean Absolute Percentage Error), gráficos de residuos
4. **Validación de datos**: Verificar que todas las features estén presentes

---

## 📝 Outputs Finales del Pipeline de Regresión:

1. **Features y targets**:
   - `X_reg.parquet`: Features preparadas
   - `y_reg.parquet`: Variable objetivo
   - `X_reg_proc.parquet`: Features preprocesadas
   - `y_reg_proc.parquet`: Target preprocesado

2. **Datos divididos**:
   - `X_train_reg.parquet`: Features de entrenamiento
   - `X_test_reg.parquet`: Features de prueba
   - `y_train_reg.parquet`: Target de entrenamiento
   - `y_test_reg.parquet`: Target de prueba

3. **Modelos entrenados**:
   - `resultados_reg.pickle`: Diccionario con 5 modelos entrenados
   - Modelos guardados en `models_reg/` (si se usa ruta relativa)

4. **Métricas de evaluación**:
   - `metricas_reg.parquet`: DataFrame con métricas de todos los modelos

---

## 🎯 Propósito de la Regresión:

La regresión permite:
- **Predecir la cantidad exacta de ventas** basándose en características históricas
- **Identificar patrones** que influyen en el volumen de ventas
- **Tomar decisiones** sobre inventario, producción, o distribución basadas en predicciones numéricas
- **Comparar modelos** para seleccionar el mejor algoritmo para este problema específico
- **Entender relaciones** entre features y la variable objetivo mediante modelos interpretables (Linear, Ridge, Lasso)

