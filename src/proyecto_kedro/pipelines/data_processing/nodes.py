import pandas as pd
import re
from typing import Tuple, List

def limpiar_caracteres_especiales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza caracteres problemáticos (como comillas ' ') por simples o los elimina.
    """
    def clean_text(s):
        if isinstance(s, str):
            # Reemplaza comillas inteligentes y otros caracteres no ASCII
            s = s.replace('\x91', "'").replace('\x92', "'")
            s = s.replace('\x93', '"').replace('\x94', '"')
            # Elimina otros caracteres no imprimibles
            s = re.sub(r'[^\x20-\x7EÁÉÍÓÚÑáéíóúñ]', '', s)
            return s.strip()
        return s

    return df.applymap(clean_text)

def limpiar_productos(df: pd.DataFrame) -> pd.DataFrame:
    df["PRODUCTO"] = df["PRODUCTO"].str.strip().str.upper()
    df["TIP_DOC"] = df["TIP_DOC"].str.strip()
    df["COMUNA"] = df["COMUNA"].str.strip()
    # Limpiar caracteres especiales
    df = limpiar_caracteres_especiales(df)
    return df

def extraer_peso_y_limpiar_productos_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los nombres de productos, extrae el peso en KG y crea un nombre base,
    además de limpiar caracteres especiales.
    """
    def extraer_peso(nombre: str) -> float:
        nombre = nombre.upper()
        match_kg = re.search(r'(\d+[.,]?\d*)\s*KG', nombre)
        if match_kg:
            return float(match_kg.group(1).replace(',', '.'))
        match_gr = re.search(r'(\d+[.,]?\d*)\s*GR', nombre)
        if match_gr:
            return float(match_gr.group(1).replace(',', '.')) / 1000
        match_mult = re.search(r'(\d+)\s*X\s*(\d+)', nombre)
        if match_mult:
            return float(match_mult.group(1)) * float(match_mult.group(2)) / 1000
        return None

    df["PRODUCTO"] = df["PRODUCTO"].str.strip().str.upper()
    df["PESO_KG"] = df["PRODUCTO"].apply(extraer_peso)
    df["PRODUCTO_BASE"] = df["PRODUCTO"]\
        .str.replace(r'X\s*\d+(\s*(KG|GR))?', '', regex=True)\
        .str.replace(r'\d+\s*X\s*\d+', '', regex=True)\
        .str.replace(r'\d+[.,]?\d*KG', '', regex=True)\
        .str.replace(r'\d+', '', regex=True)\
        .str.strip()

    # Limpiar caracteres especiales en todo el dataframe antes de guardar
    df = limpiar_caracteres_especiales(df)
    return df

def normalizar_productos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega una columna PRODUCTO_BASE_NORMAL al dataframe, normalizando el nombre del producto.
    """
    def normalizar(nombre):
        nombre = nombre.lower()                      # pasar a minúsculas
        nombre = re.sub(r'\d+(\.\d+)?', '', nombre) # quitar números
        nombre = re.sub(r'[^\w\s]', '', nombre)     # quitar signos de puntuación
        nombre = re.sub(r'\s+', ' ', nombre)        # espacios extra
        return nombre.strip()                        # quitar espacios al inicio y fin

    df['PRODUCTO_BASE_NORMAL'] = df['PRODUCTO_BASE'].apply(normalizar)
    return df

def normalizar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas, tipos de datos y formatos para un DataFrame de ventas o compras."""
    
    # Asegurar mayúsculas en los nombres de columnas
    df.columns = [col.strip().upper() for col in df.columns]

    # Limpiar espacios en texto
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # Convertir tipos
    df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="coerce")
    df["PESO_KG"] = pd.to_numeric(df["PESO_KG"], errors="coerce")
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

    # Eliminar nulos críticos
    df = df.dropna(subset=["FECHA", "CANTIDAD", "PESO_KG"])

    # Resetear índice
    df = df.reset_index(drop=True)

    return df

def preprocesar_ventas(df):
    df = df.copy()
    
    # --- Features básicas ---
    df['MES'] = df['FECHA'].dt.month
    df['PRODUCTO_ID'] = df['PRODUCTO'].astype('category').cat.codes
    df['COMUNA_ID'] = df['COMUNA'].astype('category').cat.codes
    
    # --- Filtrado por percentiles para eliminar outliers ---
    q_low = df['CANTIDAD'].quantile(0.01)   # 1er percentil
    q_high = df['CANTIDAD'].quantile(0.99)  # 99º percentil
    df = df[(df['CANTIDAD'] >= q_low) & (df['CANTIDAD'] <= q_high)]
    
    # --- Ordenar para calcular ventas anteriores ---
    df = df.sort_values(['PRODUCTO_ID','COMUNA_ID','FECHA'])
    
    # --- Feature histórica para clasificación binaria ---
    df['VENTA_MES_ANTERIOR'] = df.groupby(['PRODUCTO_ID','COMUNA_ID'])['CANTIDAD'].shift(1)
    df = df.dropna(subset=['VENTA_MES_ANTERIOR'])
    df['AUMENTA'] = (df['CANTIDAD'] > df['VENTA_MES_ANTERIOR']).astype(int)
    
    # --- Clasificación multiclase usando percentiles ---
    # Calcular percentiles para dividir en 3 rangos: baja, media, alta
    p33 = df['CANTIDAD'].quantile(0.33)
    p66 = df['CANTIDAD'].quantile(0.66)
    
    # Crear clases basadas en percentiles
    df['VENTA_CLASE'] = pd.cut(
        df['CANTIDAD'],
        bins=[-float('inf'), p33, p66, float('inf')],
        labels=['baja', 'media', 'alta'],
        include_lowest=True
    )
    
    # --- Selección final de columnas ---
    columnas_finales = [
        'FECHA','PRODUCTO','COMUNA','CANTIDAD','MES','PRODUCTO_ID','COMUNA_ID',
        'VENTA_MES_ANTERIOR','AUMENTA','VENTA_CLASE'
    ]
    
    df = df[columnas_finales]
    
    return df
