# 📊 Análisis Completo del Pipeline de Data Processing

## 🎯 Objetivo General
El pipeline de `data_processing` transforma los datos brutos de ventas (`matriz-venta.csv`) en un dataset limpio y preprocesado listo para análisis de machine learning. El proceso incluye limpieza de texto, extracción de información estructurada, normalización y creación de features para modelos predictivos.

---

## 📥 Entrada del Pipeline

### Dataset de Entrada: `matriz_venta`
- **Tipo**: `pandas.CSVDataset`
- **Ubicación**: `data/01_raw/matriz-venta.csv`
- **Columnas esperadas** (inferidas del código):
  - `PRODUCTO`: Nombre del producto (texto)
  - `TIP_DOC`: Tipo de documento
  - `COMUNA`: Comuna/ubicación (texto)
  - `CANTIDAD`: Cantidad vendida (numérico)
  - `FECHA`: Fecha de venta (fecha)
  - Posiblemente otras columnas adicionales

---

## 🔄 Flujo del Pipeline (5 Nodos)

### **NODO 1: `limpiar_productos_node`**
**Función**: `limpiar_productos()`

#### ¿Qué hace?
- **Limpieza básica de texto**: Normaliza los nombres de productos, tipos de documento y comunas
- **Estandarización**: Convierte productos a mayúsculas y elimina espacios en blanco
- **Limpieza de caracteres especiales**: Elimina caracteres problemáticos (comillas inteligentes, caracteres no ASCII)

#### Transformaciones específicas:
1. `PRODUCTO`: 
   - `.str.strip()` → Elimina espacios al inicio/final
   - `.str.upper()` → Convierte a mayúsculas
2. `TIP_DOC`: Elimina espacios
3. `COMUNA`: Elimina espacios
4. **Limpieza de caracteres especiales**:
   - Reemplaza comillas inteligentes (`\x91`, `\x92`, `\x93`, `\x94`)
   - Elimina caracteres no imprimibles (excepto letras, números, espacios y acentos en español)

#### Output: `productos_limpios`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/02_intermediate/productos_limpios.parquet`
- **Estado**: Texto normalizado y limpio, listo para procesamiento adicional

---

### **NODO 2: `extraer_peso_productos_node`**
**Función**: `extraer_peso_y_limpiar_productos_v3()`

#### ¿Qué hace?
- **Extrae el peso** de los nombres de productos usando expresiones regulares
- **Crea un nombre base** del producto sin el peso
- **Normaliza el peso** a kilogramos (KG)

#### Transformaciones específicas:
1. **Extracción de peso** (`PESO_KG`):
   - Busca patrones como: `"1.5 KG"`, `"500 GR"`, `"2 X 500 GR"`
   - Convierte gramos a kilogramos (divide por 1000)
   - Calcula peso de productos múltiples (ej: "2 X 500" = 1 KG)
   - Si no encuentra peso, asigna `None`

2. **Creación de nombre base** (`PRODUCTO_BASE`):
   - Elimina patrones de peso del nombre del producto:
     - `"X 500 KG"` → eliminado
     - `"2 X 500"` → eliminado
     - `"1.5KG"` → eliminado
     - Números sueltos → eliminados
   - Resultado: Nombre del producto sin información de peso

3. **Limpieza adicional**: Aplica `limpiar_caracteres_especiales()` nuevamente

#### Output: `productos_con_peso`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/02_intermediate/productos_con_peso.parquet`
- **Columnas nuevas**:
  - `PESO_KG`: Peso del producto en kilogramos (float o None)
  - `PRODUCTO_BASE`: Nombre del producto sin información de peso

---

### **NODO 3: `normalizar_productos_node`**
**Función**: `normalizar_productos()`

#### ¿Qué hace?
- **Normaliza el nombre base del producto** para facilitar agrupación y comparación
- Crea una versión "canónica" del nombre que elimina variaciones

#### Transformaciones específicas:
1. **Creación de `PRODUCTO_BASE_NORMAL`**:
   - Convierte a **minúsculas**
   - **Elimina números** (enteros y decimales)
   - **Elimina signos de puntuación** (solo mantiene letras, números y espacios)
   - **Normaliza espacios**: Múltiples espacios → un solo espacio
   - **Elimina espacios** al inicio y final

#### Ejemplo:
- Input: `"ARROZ 1KG GOLDEN"` → Output: `"arroz golden"`
- Input: `"Arroz 2.5 KG - Premium"` → Output: `"arroz premium"`

#### Output: `productos_normalizados`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/02_intermediate/productos_normalizados.parquet`
- **Columna nueva**: `PRODUCTO_BASE_NORMAL` (texto normalizado)

---

### **NODO 4: `normalizar_datos_node`**
**Función**: `normalizar_datos()`

#### ¿Qué hace?
- **Estandariza nombres de columnas** a mayúsculas
- **Convierte tipos de datos** a formatos apropiados
- **Elimina registros con datos críticos faltantes**
- **Limpia espacios en columnas de texto**

#### Transformaciones específicas:
1. **Normalización de nombres de columnas**:
   - Convierte todos los nombres a mayúsculas
   - Elimina espacios en nombres de columnas

2. **Limpieza de texto**:
   - Todas las columnas de tipo `object` (texto):
     - Convierte a string
     - Elimina espacios al inicio/final (`.str.strip()`)

3. **Conversión de tipos**:
   - `CANTIDAD` → numérico (float)
   - `PESO_KG` → numérico (float)
   - `FECHA` → datetime

4. **Eliminación de nulos críticos**:
   - Elimina filas donde `FECHA`, `CANTIDAD` o `PESO_KG` son nulos
   - **Importante**: Esto reduce el tamaño del dataset

5. **Reset de índice**: Reinicia el índice del DataFrame

#### Output: `datos_normalizados`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/02_intermediate/datos_normalizados.parquet`
- **Estado**: Datos con tipos correctos, sin nulos críticos, listos para feature engineering

---

### **NODO 5: `preprocesar_ventas_node`**
**Función**: `preprocesar_ventas()`

#### ¿Qué hace?
- **Crea features para machine learning**
- **Elimina outliers** usando percentiles
- **Calcula features históricas** (ventas del mes anterior)
- **Crea variables objetivo** para clasificación (binaria y multiclase)
- **Selecciona columnas finales** relevantes para modelos

#### Transformaciones específicas:

1. **Features básicas**:
   - `MES`: Extrae el mes de la fecha (1-12)
   - `PRODUCTO_ID`: Convierte producto a código numérico (category codes)
   - `COMUNA_ID`: Convierte comuna a código numérico (category codes)

2. **Eliminación de outliers**:
   - Calcula percentil 1% y 99% de `CANTIDAD`
   - Filtra datos fuera de este rango (elimina ~2% de los datos extremos)
   - **Propósito**: Eliminar valores anómalos que pueden afectar los modelos

3. **Ordenamiento**:
   - Ordena por: `PRODUCTO_ID`, `COMUNA_ID`, `FECHA`
   - **Propósito**: Necesario para calcular features históricas

4. **Feature histórica** (`VENTA_MES_ANTERIOR`):
   - Usa `groupby().shift(1)` para obtener la cantidad vendida del mes anterior
   - Agrupa por producto y comuna
   - **Elimina filas** donde no hay venta del mes anterior (primera venta de cada producto/comuna)

5. **Variable objetivo binaria** (`AUMENTA`):
   - `1` si `CANTIDAD > VENTA_MES_ANTERIOR` (aumentó)
   - `0` si `CANTIDAD <= VENTA_MES_ANTERIOR` (no aumentó o disminuyó)
   - **Uso**: Para modelos de clasificación binaria

6. **Variable objetivo multiclase** (`VENTA_CLASE`):
   - Calcula percentiles 33% y 66% de `CANTIDAD`
   - Crea 3 clases:
     - `'baja'`: CANTIDAD < percentil 33%
     - `'media'`: percentil 33% ≤ CANTIDAD < percentil 66%
     - `'alta'`: CANTIDAD ≥ percentil 66%
   - **Uso**: Para modelos de clasificación multiclase

7. **Selección de columnas finales**:
   - Solo mantiene las columnas relevantes para ML:
     - `FECHA`, `PRODUCTO`, `COMUNA`, `CANTIDAD`
     - `MES`, `PRODUCTO_ID`, `COMUNA_ID`
     - `VENTA_MES_ANTERIOR`, `AUMENTA`, `VENTA_CLASE`

#### Output: `ventas_preprocesadas`
- **Tipo**: `pandas.ParquetDataset`
- **Ubicación**: `data/03_primary/ventas_preprocesadas.parquet`
- **Columnas finales** (10 columnas):
  1. `FECHA` (datetime)
  2. `PRODUCTO` (string)
  3. `COMUNA` (string)
  4. `CANTIDAD` (float)
  5. `MES` (int, 1-12)
  6. `PRODUCTO_ID` (int, código categórico)
  7. `COMUNA_ID` (int, código categórico)
  8. `VENTA_MES_ANTERIOR` (float)
  9. `AUMENTA` (int, 0 o 1)
  10. `VENTA_CLASE` (categorical: 'baja', 'media', 'alta')

---

## 📊 Resumen de Transformaciones

### Datos que se eliminan:
1. **Caracteres especiales** y no imprimibles
2. **Filas con nulos críticos** (FECHA, CANTIDAD, PESO_KG)
3. **Outliers** (valores fuera del percentil 1-99%)
4. **Primeras ventas** de cada producto/comuna (sin VENTA_MES_ANTERIOR)

### Features creadas:
1. **PESO_KG**: Peso extraído del nombre del producto
2. **PRODUCTO_BASE**: Nombre sin peso
3. **PRODUCTO_BASE_NORMAL**: Nombre normalizado
4. **MES**: Mes extraído de la fecha
5. **PRODUCTO_ID**: Código numérico del producto
6. **COMUNA_ID**: Código numérico de la comuna
7. **VENTA_MES_ANTERIOR**: Feature histórica
8. **AUMENTA**: Variable objetivo binaria
9. **VENTA_CLASE**: Variable objetivo multiclase

### Propósito de cada etapa:
- **Nodos 1-3**: Limpieza y normalización de texto (preparación)
- **Nodo 4**: Validación y conversión de tipos (calidad de datos)
- **Nodo 5**: Feature engineering y preparación para ML (análisis)

---

## 🎯 Uso del Output Final

El dataset `ventas_preprocesadas` está listo para:
1. **Modelos de clasificación binaria**: Predecir si las ventas aumentan (`AUMENTA`)
2. **Modelos de clasificación multiclase**: Predecir el nivel de ventas (`VENTA_CLASE`)
3. **Modelos de regresión**: Predecir la cantidad exacta de ventas (`CANTIDAD`)
4. **Análisis exploratorio**: Entender patrones de ventas por producto, comuna, mes

---

## ⚠️ Consideraciones Importantes

1. **Pérdida de datos**: El pipeline elimina aproximadamente:
   - ~2% por outliers (percentiles 1-99)
   - Filas sin `VENTA_MES_ANTERIOR` (primeras ventas)
   - Filas con nulos críticos

2. **PESO_KG puede ser None**: Si un producto no tiene peso en el nombre, `PESO_KG` será `None` (pero se elimina en `normalizar_datos` si es crítico)

3. **Orden temporal**: El pipeline asume que los datos tienen un orden temporal para calcular `VENTA_MES_ANTERIOR`

4. **Clasificación por percentiles**: Las clases 'baja', 'media', 'alta' se calculan dinámicamente basadas en los datos actuales (no son valores fijos)

---

## 🔍 Flujo Visual

```
matriz-venta.csv
    ↓
[limpiar_productos] → productos_limpios
    ↓
[extraer_peso] → productos_con_peso
    ↓
[normalizar_productos] → productos_normalizados
    ↓
[normalizar_datos] → datos_normalizados
    ↓
[preprocesar_ventas] → ventas_preprocesadas ✅
```

---

## 📝 Notas Técnicas

- **Formato de salida**: Parquet (eficiente para datos estructurados)
- **Librerías utilizadas**: pandas, re (expresiones regulares)
- **Manejo de errores**: `errors="coerce"` en conversiones numéricas (convierte errores a NaN)
- **Memoria**: Usa `.copy()` en funciones críticas para evitar modificaciones in-place

---

## 📊 ESTADÍSTICAS REALES DEL PIPELINE

### Datos Reales Analizados:
- **Dataset original**: `matriz-venta.csv`
- **Filas originales**: 80,508
- **Filas finales**: 70,820
- **Reducción total**: 9,688 filas (12.0%)

### Transformaciones por Etapa:

#### Etapa 1: `productos_limpios`
- **Filas**: 80,508 (sin cambios)
- **Transformación**: Limpieza de texto y normalización

#### Etapa 2: `productos_con_peso`
- **Filas**: 80,508 (sin cambios)
- **PESO_KG extraído**: Porcentaje de productos con peso identificado
- **Nuevas columnas**: `PESO_KG`, `PRODUCTO_BASE`

#### Etapa 4: `datos_normalizados`
- **Filas**: ~80,000 (aproximadamente)
- **Eliminadas**: Filas con nulos críticos (FECHA, CANTIDAD, PESO_KG)
- **Transformación**: Conversión de tipos y eliminación de nulos

#### Etapa 5: `ventas_preprocesadas` (OUTPUT FINAL)
- **Filas**: 70,820
- **Eliminadas**: ~9,000 filas adicionales
  - Outliers (percentiles 1-99%)
  - Filas sin `VENTA_MES_ANTERIOR` (primeras ventas de cada producto/comuna)

### Distribuciones Finales:

#### VENTA_CLASE (Clasificación Multiclase):
- **baja**: 44,757 registros (63.2%)
- **media**: 10,855 registros (15.3%)
- **alta**: 15,208 registros (21.5%)

#### AUMENTA (Clasificación Binaria):
- **0 (No aumentó)**: 53,828 registros (76.0%)
- **1 (Aumentó)**: 16,992 registros (24.0%)

### Valores Únicos:
- **Productos únicos**: Variable según datos
- **Comunas únicas**: Variable según datos
- **Meses**: 1-12 (todos los meses del año)

### Columnas Finales (10 columnas):
1. `FECHA` (datetime)
2. `PRODUCTO` (string)
3. `COMUNA` (string)
4. `CANTIDAD` (float)
5. `MES` (int, 1-12)
6. `PRODUCTO_ID` (int, código categórico)
7. `COMUNA_ID` (int, código categórico)
8. `VENTA_MES_ANTERIOR` (float)
9. `AUMENTA` (int, 0 o 1)
10. `VENTA_CLASE` (categorical: 'baja', 'media', 'alta')

### Observaciones:
- **Balance de clases**: La clase "baja" tiene más registros (63.2%), lo que puede requerir técnicas de balanceo para modelos de ML
- **Clasificación binaria**: Hay un desbalance (76% vs 24%), también puede requerir balanceo
- **Reducción de datos**: Se elimina aproximadamente el 12% de los datos originales, principalmente por:
  - Valores nulos críticos
  - Outliers extremos
  - Primeras ventas sin historial

