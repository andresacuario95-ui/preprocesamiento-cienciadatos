import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def cargar_datos(ruta_archivo):
    """
    Cargar dataset desde un archivo CSV
    """
    try:
        df = pd.read_csv(ruta_archivo)
        print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def explorar_datos(df):
    """
    Exploración básica del dataset
    """
    print("=== EXPLORACIÓN DE DATOS ===")
    print(f"Dimensiones: {df.shape}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    return df.info()

def manejar_valores_nulos(df, estrategia='media'):
    """
    Manejar valores nulos en el dataset
    """
    print("=== MANEJO DE VALORES NULOS ===")
    
    # Identificar columnas con valores nulos
    columnas_nulos = df.columns[df.isnull().any()].tolist()
    
    if not columnas_nulos:
        print("No hay valores nulos en el dataset")
        return df
    
    print(f"Columnas con valores nulos: {columnas_nulos}")
    
    for columna in columnas_nulos:
        if df[columna].dtype in ['object', 'bool']:
            # Para variables categóricas
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            # Para variables numéricas
            if estrategia == 'media':
                imputer = SimpleImputer(strategy='mean')
            elif estrategia == 'mediana':
                imputer = SimpleImputer(strategy='median')
            else:
                imputer = SimpleImputer(strategy='constant', fill_value=0)
        
        df[columna] = imputer.fit_transform(df[[columna]])
        print(f"Valores nulos en {columna}: IMPUTADOS")
    
    return df

def eliminar_duplicados(df):
    """
    Eliminar filas duplicadas
    """
    print("=== ELIMINACIÓN DE DUPLICADOS ===")
    filas_antes = df.shape[0]
    df = df.drop_duplicates()
    filas_despues = df.shape[0]
    duplicados_eliminados = filas_antes - filas_despues
    
    print(f"Filas antes: {filas_antes}")
    print(f"Filas después: {filas_despues}")
    print(f"Duplicados eliminados: {duplicados_eliminados}")
    
    return df

def codificar_variables_categoricas(df):
    """
    Codificar variables categóricas usando one-hot encoding
    """
    print("=== CODIFICACIÓN DE VARIABLES CATEGÓRICAS ===")
    
    # Identificar columnas categóricas
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not columnas_categoricas:
        print("No hay variables categóricas para codificar")
        return df
    
    print(f"Variables categóricas encontradas: {columnas_categoricas}")
    
    # Aplicar one-hot encoding
    df_encoded = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)
    
    print(f"Dimensiones después de codificación: {df_encoded.shape}")
    
    return df_encoded

def normalizar_datos(df, metodo='standard'):
    """
    Normalizar variables numéricas
    """
    print("=== NORMALIZACIÓN DE DATOS ===")
    
    # Identificar columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columnas_numericas:
        print("No hay variables numéricas para normalizar")
        return df
    
    print(f"Variables numéricas a normalizar: {columnas_numericas}")
    
    if metodo == 'standard':
        scaler = StandardScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    else:
        print("Método no reconocido. Usando StandardScaler por defecto")
        scaler = StandardScaler()
    
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])
    print("Normalización completada")
    
    return df

def pipeline_preprocesamiento_completo(ruta_archivo):
    """
    Pipeline completo de preprocesamiento
    """
    print("🚀 INICIANDO PIPELINE DE PREPROCESAMIENTO COMPLETO")
    print("=" * 50)
    
    # 1. Cargar datos
    df = cargar_datos(ruta_archivo)
    if df is None:
        return None
    
    # 2. Exploración inicial
    explorar_datos(df)
    
    # 3. Manejar valores nulos
    df = manejar_valores_nulos(df)
    
    # 4. Eliminar duplicados
    df = eliminar_duplicados(df)
    
    # 5. Codificar variables categóricas
    df = codificar_variables_categoricas(df)
    
    # 6. Normalizar datos
    df = normalizar_datos(df)
    
    print("=" * 50)
    print("✅ PIPELINE DE PREPROCESAMIENTO COMPLETADO")
    print(f"Dataset final: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de cómo usar el pipeline
    # dataset = pipeline_preprocesamiento_completo('data/raw/dataset.csv')
    pass