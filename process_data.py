import pandas as pd
from sklearn.utils import shuffle

def balance_dataset(input_path, output_path, target_col='IsFraud', n_samples=10000, random_state=42):
    """
    Balancea un dataset desbalanceado manteniendo todos los casos positivos y muestreando la clase negativa.
    
    Parámetros:
    -----------
    input_path : str
        Ruta al archivo CSV original
    output_path : str
        Ruta donde guardar el dataset balanceado
    target_col : str
        Nombre de la columna objetivo (default 'IsFraud')
    n_samples : int
        Número de muestras a conservar de la clase mayoritaria (default 10000)
    random_state : int
        Semilla para reproducibilidad (default 42)
    
    Retorna:
    --------
    tuple
        (DataFrame balanceado, conteo de clases)
    """
    df = pd.read_csv(input_path)
    
    fraudes = df[df[target_col] == 1]
    no_fraudes = df[df[target_col] == 0]
    
    no_fraudes_reducidos = no_fraudes.sample(n=n_samples, random_state=random_state)
    
    df_balanceado = shuffle(pd.concat([fraudes, no_fraudes_reducidos]), random_state=random_state)
    
    df_balanceado.to_csv(output_path, index=False)
    
    class_counts = df_balanceado[target_col].value_counts().to_dict()
    
    print(f"Proporción final: {class_counts[0]} no fraudes vs {class_counts[1]} fraudes")
    print(f"Nuevo balance: {len(df_balanceado)} registros en total")
    
    return df_balanceado, class_counts