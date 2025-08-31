import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocesar_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(base_dir, "database.csv")
    raw_data = pd.read_csv(ruta_csv, low_memory=False)

    # Se revisa el tamaño del dataset original
    mostrar_tamanio(raw_data)

    # Se eliminan filas vacías y se verifica el nuevo tamaño del dataset sin esas filas vacías
    df_sin_vacios = eliminar_filas_vacias(raw_data)
    mostrar_tamanio(df_sin_vacios)

    # Se revisa información de las variables: tipo de dato que contienen y cuántos registros no nulos contienen
    mostrar_info_variables(df_sin_vacios)

    # Se eliminan las columnas que no se consideran de interés para el modelo
    data_elegida = df_sin_vacios.drop(columns=['author', 'review_date', 'customer_review', 'route', 'date_flown'])

    # Estandarizamos la columna 'aircraft'
    data_elegida = estandarizar_aircraft(data_elegida)

    # Conversión de etiqueta `recommended` a binaria
    data_elegida['recommended'] = data_elegida['recommended'].map({'yes': 1, 'no': 0})

    # Imputar valores faltantes en la etiqueta con la moda
    moda_recommended = data_elegida['recommended'].mode()[0]
    data_elegida['recommended'] = data_elegida['recommended'].fillna(moda_recommended)

    # Separar X e y
    X = data_elegida.drop(columns=['recommended'])
    y = data_elegida['recommended']

    # División en entrenamiento, validación y prueba
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=11)

    # Dividimos las columnas por tipos de variables
    columnas_numericas = ['overall', 'seat_comfort', 'cabin_service', 'food_bev', 'entertainment', 'ground_service', 'value_for_money']
    columnas_nominales = ['airline', 'traveller_type', 'aircraft']
    columna_ordinal = ['cabin']

    # Se crean los pipelines para codificar y escalar las variables
    preprocessor = crear_pipelines(columnas_numericas, columnas_nominales, columna_ordinal)

    # Ajustar el preprocesador con el conjunto de entrenamiento
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test


def mostrar_tamanio(df):
    print(f'Tamaño actual del dataset: {df.shape}')
    print('='*60 + '\n')

def eliminar_filas_vacias(df):
    df = df.dropna(how='all')
    df = df.reset_index(drop=True)
    return df

def mostrar_info_variables(df):
    print('Información sobre las variables del dataset:')
    print(df.info())
    print('='*60 + '\n')
    print('Estadística descriptiva del dataset:')
    print(df.describe(include="all"))
    print('='*60 + '\n')

def estandarizar_aircraft(df):
    # Pasar a minúsculas y quitar espacios extra
    df['aircraft'] = df['aircraft'].str.lower().str.strip()

    # Reemplazos datos que no tienen sentido
    replacements = {
        'unknow': 'unknown',
        "don't know!": 'unknown',
        'dont know': 'unknown',
        'don t know': 'unknown',
        'i dont know': 'unknown',
        '?': 'unknown',
        '=': 'unknown',
        'no': 'unknown',
        'plane': 'unknown',
        'economy': 'unknown',
        'various': 'multiple',
        'several': 'multiple',
        'multiple': 'multiple'
    }
    df['aircraft'] = df['aircraft'].replace(replacements)

    # Limpieza basada en patrones (palabras clave)
    df['aircraft'] = df['aircraft'].apply(regex_aircraft)

    # Agrupar aeronaves poco frecuentes (menos de 50 ocurrencias) como "other"
    counts = df['aircraft'].value_counts()
    rare_aircraft = counts[counts < 50].index
    df['aircraft'] = df['aircraft'].replace(rare_aircraft, 'other')

    return df

def regex_aircraft(x):
    if not isinstance(x, str):
        return 'unknown'
    if re.search(r'boeing\s*737', x): return 'boeing 737'
    if re.search(r'boeing\s*747', x): return 'boeing 747'
    if re.search(r'boeing\s*777', x): return 'boeing 777'
    if re.search(r'boeing\s*787', x): return 'boeing 787'
    if re.search(r'boeing', x): return 'boeing other'
    if re.search(r'airbus\s*a320', x): return 'airbus a320'
    if re.search(r'airbus\s*a321', x): return 'airbus a321'
    if re.search(r'airbus\s*a330', x): return 'airbus a330'
    if re.search(r'airbus\s*a350', x): return 'airbus a350'
    if re.search(r'airbus', x): return 'airbus other'
    if re.search(r'unknown', x): return 'unknown'
    if re.search(r'multiple', x): return 'multiple'
    return 'other'

def crear_pipelines(columnas_numericas, columnas_nominales, columna_ordinal):
    # Pipeline para variables numéricas
    pipeline_numericas = Pipeline([
        # Se imputan los valores faltantes de las variables numéricas con la media
        ('imputer', SimpleImputer(strategy='mean')),
        # Escalamos los datos numéricos porque la escala de la variable overall es diferente a las demás
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables nominales
    pipeline_nominales = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Pipeline para columna cabin (ordinal)
    orden_cabin = [['Economy Class', 'Premium Economy', 'Business Class', 'First Class']]
    pipeline_cabin = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=orden_cabin))
    ])

    # Combinar todo con ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', pipeline_numericas, columnas_numericas),
        ('nom', pipeline_nominales, columnas_nominales),
        ('cabin', pipeline_cabin, columna_ordinal)
    ])

    return preprocessor