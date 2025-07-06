import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_fraud_data(df, target_col='IsFraud', test_size=0.2, val_size=0.25, random_state=42):
    """
    Preprocesamiento completo para datos de fraude:
    1. Manejo de fechas
    2. Extracción de características temporales
    3. Eliminación de columnas no necesarias
    4. One-Hot Encoding
    5. División estratificada
    6. Transformaciones cíclicas
    7. Estandarización
    
    Parámetros:
    -----------
    df : DataFrame
        Datos originales con columna TransactionDate
    target_col : str
        Nombre de la columna objetivo
    test_size, val_size : float
        Tamaños para test y validation sets
    random_state : int
        Semilla para reproducibilidad
        
    Retorna:
    --------
    dict
        Diccionario con X_train, X_val, X_test, y_train, y_val, y_test
    """
    #Copia del DataFrame para no modificar el original
    df_processed = df.copy()
    
    #Procesamiento de fechas
    df_processed['TransactionDate'] = pd.to_datetime(df_processed['TransactionDate'])
    df_processed['TransactionHour'] = df_processed['TransactionDate'].dt.hour
    df_processed['TransactionDay'] = df_processed['TransactionDate'].dt.day
    df_processed['TransactionMonth'] = df_processed['TransactionDate'].dt.month
    
    #Eliminar columnas no necesarias
    features_to_drop = ['TransactionID', 'TransactionDate', 'MerchantID']
    df_processed = df_processed.drop(columns=features_to_drop)
    
    #One-Hot Encoding para variables categóricas
    categorical_cols = ['TransactionType', 'Location']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    #Separar features y target
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    
    #División estratificada train/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    #División train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full)
    
    #Transformaciones cíclicas
    cyclic_features = [
        ('TransactionHour', 24),
        ('TransactionDay', 31),
        ('TransactionMonth', 12)
    ]
    
    for col, max_val in cyclic_features:
        for dataset in [X_train, X_val, X_test]:
            dataset[f'{col}_sin'] = np.sin(2 * np.pi * dataset[col]/max_val)
            dataset[f'{col}_cos'] = np.cos(2 * np.pi * dataset[col]/max_val)
            dataset.drop(col, axis=1, inplace=True)
    
    #Estandarización de Amount
    scaler = StandardScaler()
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_val['Amount'] = scaler.transform(X_val[['Amount']])
    X_test['Amount'] = scaler.transform(X_test[['Amount']])
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler
    }