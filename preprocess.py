import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_y_preprocesar_datos(ruta_csv):
    """
    Carga el dataset de fraude, lo preprocesa completamente y lo divide
    en conjuntos de entrenamiento, validación y prueba.
    """
    data = pd.read_csv(ruta_csv)

    data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
    data['Hour'] = data['TransactionDate'].dt.hour
    data['DayOfWeek'] = data['TransactionDate'].dt.dayofweek
    data['Day'] = data['TransactionDate'].dt.day
    data['Month'] = data['TransactionDate'].dt.month

    features_to_drop = ['TransactionID', 'TransactionDate', 'MerchantID']
    df_processed = data.drop(columns=features_to_drop)

    df_processed = pd.get_dummies(df_processed, columns=['TransactionType', 'Location'], drop_first=True)

    X = df_processed.drop('IsFraud', axis=1)
    y = df_processed['IsFraud']
    
    #stratify=y para mantener proporción de fraudes
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)

    scaler = StandardScaler()
    
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_val['Amount'] = scaler.transform(X_val[['Amount']])
    X_test['Amount'] = scaler.transform(X_test[['Amount']])
    
    return X_train, X_val, X_test, y_train, y_val, y_test