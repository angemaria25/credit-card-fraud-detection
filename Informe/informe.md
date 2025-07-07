El proyecto aborda la detección de fraude en transacciones de trajetas de crédito, el objetivo es clasificar cada transacción como legítima (clase 0) o fraudulenta (clase 1).

Se utilizó el dataset `Credit Card Fraud Detection de Kaggle`, que contiene 100,000 transacciones simuladas con las siguientes características principales:

- **TransactionID**:  Un identificador único para cada transacción, que garantiza la trazabilidad.
    - **TransactionDate**: La fecha y hora en que se produjo la transacción, lo que permite el análisis temporal.
    - **Amount**: El valor monetario de la transacción, que puede ayudar a identificar transacciones inusualmente grandes que pueden indicar fraude.
    - **MerchantID**: Un identificador para el comerciante involucrado en la transacción, útil para evaluar patrones de fraude relacionados con el comerciante.
    - **TransactionType**: Indica si la transacción fue una compra o un reembolso, proporcionando contexto para la actividad.
    - **Location**: La ubicación geográfica de la transacción, lo que facilita el análisis de las tendencias de fraude por región.
    - **IsFraud**: Una variable objetivo binaria que indica si la transacción es fraudulenta (1) o legítima (0), esencial para los modelos de aprendizaje supervisado

**El análisis inicial de los datos reveló varios puntos:**

1. Desbalanceo de la Variable Objetivo (IsFraud): El dataset está extremadamente desbalanceado. Solo el 1% de las transacciones (1000) son fraudulentas, mientras que el 99% (99000) son legítimas. Este desbalance es el mayor desafío. Un modelo entrenado con estos datos tenderá a predecir siempre `no fraude`, logrando una alta exactitud (accuracy) pero siendo inútil en la práctica. Se determinó que era necesario balancear el conjunto de datos para un entrenamiento efectivo.

2. Análisis Temporal y de Montos: Las transacciones fraudulentas muestran un pico de incidencia durante las horas nocturnas y de madrugada, esto sugiere que los patrones temporales son una característica predictiva importante. Aunque la distribución general de los montos es similar para ambas clases, las transacciones fraudulentas tienden a tener montos ligeramente más altos en promedio.

3. Análisis Geográfico y por Tipo de Transacción: Ciertas ciudades, como New York, presentan una tasa de fraude significativamente mayor en comparación con otras. 

4. Calidad de los Datos: Se verificó que el dataset no contenía valores nulos ni registros duplicados, lo que simplificó la fase de limpieza.

**Proceso de Preprocesamiento de Datos:**

Para preparar los datos se siguieron los siguientes pasos:

- Balanceo del Dataset: Para manejar el desbalance extremo, se realizó un submuestreo (undersampling) de la clase mayoritaria. Se conservaron todas las 1000 transacciones de fraude y se seleccionó una muestra aleatoria de 10000 registros de transacciones legítimas, luego se creó un nuevo dataset balanceado (dataset_balanceado.csv) con 11000 transacciones.

- Ingeniería y Eliminación de Características: Se descartaron las columnas `TransactionID`, `MerchantID` y `TransactionDate`, ya que son identificadores sin valor predictivo (TransactionID) y su información fue extraída en nuevas características más útiles (TransactionDate). De TransactionDate se extrajeron TransactionHour, TransactionDay y TransactionMonth para capturar los patrones temporales identificados.

- Codificación y Transformación de Variables:
    - Variables Categóricas: TransactionType y Location se convirtieron a formato numérico mediante One-Hot Encoding, creando columnas binarias para cada categoría. Esto permite que el modelo interprete estas características correctamente.
    - Escalado de Amount: El monto de la transacción se estandarizó utilizando StandardScaler para que tuviera una media de 0 y una desviación estándar de 1, lo cual es crucial para modelos sensibles a la escala de las variables.
    - Transformación Cíclica de Tiempo: Las variables TransactionHour, TransactionDay y TransactionMonth se transformaron utilizando funciones seno y coseno, se utilizó esta técnica para capturar la naturaleza cíclica del tiempo (ej. la hora 23 está cerca de la hora 0), permitiendo al modelo entender patrones estacionales y diarios de forma más efectiva.

- División de Datos: El dataset procesado se dividió en conjuntos de entrenamiento (60%), validación (20%) y prueba (20%). Se utilizó una división estratificada para asegurar que la proporción de fraudes y no fraudes se mantuviera constante en cada conjunto.

Para establecer una línea base y evaluar el rendimiento de un modelo lineal se entrenó un modelo de Regresión Logística. El análisis se centró en abordar el desbalanceo de clases, que fue identificado como el principal desafío en la etapa de análisis exploratorio.

Al entrenar un modelo de Regresión Logística estándar sobre los datos preprocesados, el modelo base fue incapaz de identificar ninguna transacción fraudulenta, logrando un Recall de 0.0 para la clase de fraude, aunque la exactitud (accuracy) fue alta (90.91%), esta métrica resultó engañosa, ya que el modelo simplemente predecía la clase mayoritaria (no fraude) para todas las instancias, este resultado confirmó que el desbalanceo de clases impedía que el modelo aprendiera los patrones de la clase minoritaria, haciéndolo inútil para la detección de fraude.

Aplicación de Técnicas de Muestreo: Se exploraron varias técnicas para mitigar el desbalanceo de clases, comparando sus resultados en el conjunto de validación, con un enfoque en las métricas de Recall (capacidad para detectar fraudes) y F1-Score (equilibrio entre precisión y recall).

Random Undersampling (Submuestreo): Se redujo el número de muestras de la clase legítima para igualar a la clase de fraude, esta técnica fue la más efectiva para aumentar la detección, el modelo logró el Recall más alto (57%) y el F1-Score más alto (0.167), sin embargo, su precisión fue baja, generando muchas falsas alarmas.

Random Oversampling y SMOTE (Sobremuestreo): Se aumentó el número de muestras de la clase de fraude, ya sea duplicándolas (Oversampling) o creando nuevas muestras sintéticas (SMOTE), aunque ambos enfoques permitieron al modelo detectar fraudes (Recall de 48.5% y 32% respectivamente), su rendimiento fue inferior al del undersampling. La precisión siguió siendo muy baja, y SMOTE, en particular, pareció introducir ruido que confundió al modelo lineal.

Parámetro class_weight='balanced': Se ajustó la función de pérdida del modelo para penalizar más los errores en la clase de fraude, esta técnica ofreció un buen equilibrio, logrando un Recall del 50%, cercano al de undersampling, pero sin la desventaja de descartar datos potencialmente valiosos.

Ajuste de Hiperparámetros (GridSearchCV): Se utilizó GridSearchCV para encontrar el valor óptimo del hiperparámetro C (inverso de la fuerza de regularización), maximizando el F1-Score. El valor óptimo encontrado fue C=0.01.

Se observó que el umbral por defecto (0.5) no ofrecía el mejor equilibrio entre precisión y recall por lo cual se calculó la curva Precisión-Recall y se identificó el umbral óptimo de 0.479 que maximizaba el F1-Score en el conjunto de validación, aplicando este umbral al conjunto de prueba, el modelo optimizado de Regresión Logística logró un Recall del 66%, aunque esto significa que detectó dos tercios de los fraudes, la precisión se mantuvo muy baja.

La Regresión Logística, al ser un modelo lineal, demostró tener limitaciones para capturar patrones complejos y no lineales que probablemente caracterizan a las transacciones fraudulentas. A pesar de aplicar técnicas para el desbalanceo y optimizar los hiperparámetros, no se consiguió una precisión aceptable. El alto volumen de falsas alarmas (transacciones legítimas marcadas como fraude) hace que este modelo, aunque mejorado, sea inviable para una implementación en un entorno de producción real.


