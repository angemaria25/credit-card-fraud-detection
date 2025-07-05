import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_col='IsFraud', test_size=0.2, val_size=0.25, random_state=42):
    """
    Preprocesamiento completo para el dataset.
    División train/val/test, transformaciones cíclicas y escalado.
    """
    #Copia del dataframe para no modificar el original
    df = df.copy()
    
    #Separar features y target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
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
    
    #Función interna para transformaciones cíclicas
    def _add_cyclic_features(df, col_name, max_val):
        df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name]/max_val)
        df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name]/max_val)
        return df.drop(col_name, axis=1)
    
    #Aplicar a cada conjunto
    for col, max_val in cyclic_features:
        X_train = _add_cyclic_features(X_train, col, max_val)
        X_val = _add_cyclic_features(X_val, col, max_val)
        X_test = _add_cyclic_features(X_test, col, max_val)
    
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
        'scaler': scaler  #Por si necesitas el scaler después
    }