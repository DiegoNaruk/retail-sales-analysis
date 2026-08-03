# Importamos la librería pandas, que es ampliamente utilizada para el análisis y manipulación de datos estructurados en Python
import pandas as pd
import numpy as np


# Cargamos el archivo CSV que contiene los datos de entrenamiento en un DataFrame de pandas. El archivo 'train.csv' 
# debe estar en el mismo directorio que este script o proporcionar la ruta completa al archivo.
df = pd.read_csv('train.csv')
df
# Cargamos el archivo CSV que contiene los datos de prueba en un DataFrame de pandas. El archivo 'test.csv' 
# debe estar en el mismo directorio que este script o proporcionar la ruta completa al archivo.
df_test = pd.read_csv('test.csv')
df_test

# Al no tener la columna target se decició usar solamente el dataset de train:

# Exploracion inicial

# Mostramos información general del dataset con el método info()
# Esto incluye el número de entradas, cantidad de columnas, nombres de columnas, tipos de datos y valores nulos
df.info()

# Mostramos estadísticas descriptivas de las columnas numéricas del dataset con el método describe()
# Incluye métricas como la media, desviación estándar, valores mínimo y máximo, así como los percentiles
# Es útil para entender la distribución y el rango de los datos en cada columna
df.describe()

# Verificamos la cantidad de valores nulos presentes en cada columna del DataFrame
# isnull() devuelve una máscara booleana donde True indica valores faltantes
# sum() cuenta cuántos valores nulos hay por cada columna
df.isnull().sum()


# Verificamos si existen filas duplicadas en el DataFrame
# El método duplicated() devuelve una serie booleana donde cada fila es marcada como True si ya apareció antes
# Luego usamos sum() para contar cuántas filas duplicadas hay en total en el conjunto de datos
# Esto es útil para garantizar que no haya entradas repetidas que puedan distorsionar el análisis
print("Number of duplicated rows:", df.duplicated().sum())
#No existen valores duplicados en nuestro dataset. En caso de tener valores duplicados se hartia lo siguiente


#Eliminación de Duplicados

# Eliminamos las filas duplicadas del DataFrame en caso de que existan
# Utilizamos el método drop_duplicates(), que por defecto conserva la primera aparición y elimina las siguientes
# Asignamos el resultado de nuevo a 'df' para actualizar el DataFrame sin duplicados
# Esto ayuda a mantener la integridad de los datos y evita sesgos por registros repetidos
df = df.drop_duplicates()

#Tipos de Datos
# Verificamos los tipos de datos de cada columna del DataFrame
# El atributo dtypes nos muestra si las columnas están clasificadas como enteros, flotantes, objetos (texto), etc.
# Es importante asegurarse de que cada columna tenga el tipo de dato adecuado para evitar errores en análisis posteriores
# Por ejemplo, fechas o categorías mal tipadas podrían requerir conversión antes de usarse correctamente
display(df.dtypes)

# Exploración de Datos

# Importamos la biblioteca Matplotlib para crear gráficos y visualizaciones en 2D
# También importamos Seaborn, una biblioteca basada en Matplotlib que proporciona una interfaz más amigable y estilos más estéticos
import matplotlib.pyplot as plt
import seaborn as sns

# Creamos una nueva figura con un tamaño personalizado para mejorar la visibilidad del gráfico
plt.figure(figsize=(10, 6))

# Usamos un histograma con Seaborn para visualizar la distribución de la variable 'SalePrice'
# El parámetro bins=30 divide el rango de edades en 30 intervalos o "cajones"
# El parámetro kde=True añade una curva de densidad suavizada sobre el histograma, lo que ayuda a visualizar la forma de la distribución
sns.histplot(df['SalePrice'], bins=30, kde=True)

# Asignamos un título descriptivo al gráfico
plt.title('Distribución del precio de las casas')

# Etiquetamos el eje X con el nombre de la variable representada
plt.xlabel('SalePrice')

# Etiquetamos el eje Y para indicar que se representa la frecuencia de ocurrencia
plt.ylabel('Frecuencia')

# Mostramos el gráfico final en pantalla
plt.show()

# Podemos observar que gran cantidad de los precios de las casas estan en un rango de entre 100 mil y 250 mil dolares.
# Ademas se tiene casas con valores sobresalientes de mas de 500 mil doalres

# Visualización Bivariada 

# Creamos una nueva figura con un tamaño personalizado para mejorar la visibilidad del gráfico
plt.figure(figsize=(10, 6))

# Utilizamos un gráfico de dispersión (scatter plot) para analizar la relación entre dos variables numéricas:
# 'SalePrice' en el eje X y 'YearBuilt' en el eje Y
# Además, usamos el parámetro 'hue' para colorear los puntos según la variable categórica 'target'
sns.scatterplot(x='SalePrice', y='YearBuilt', data=df)

# Asignamos un título descriptivo al gráfico para contextualizar la información mostrada
plt.title('Precio de casa vs anio de construccion ')

# Etiquetamos el eje X para indicar que representa la edad de los pacientes
plt.xlabel('Precio de casa ')

# Etiquetamos el eje Y para indicar que representa la frecuencia cardíaca máxima alcanzada
plt.ylabel('anio de construccion')

# Mostramos el gráfico final en pantalla
plt.show()

#De acuerdo a nuestro grafico gran cantidad de las casas mas baratas fueron construidas antes de los años 2000, 
# mientras que las mas caras fueron construidad despues de los años 2000.


# Matriz de Correlación

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

# Handle remaining missing values by filling with the mean of each column
df_processed = df_encoded.fillna(df_encoded.mean())
missing_values_after_imputation = df_processed.isnull().sum()
print("Missing values after handling:")
display(missing_values_after_imputation[missing_values_after_imputation > 0])

# Crear el mapa de calor de correlación
plt.figure(figsize=(12, 10))
# Creamos una figura de tamaño 12x10 pulgadas para el gráfico de calor.

correlation_matrix = df_processed.corr()
# Calculamos la matriz de correlación entre las variables numéricas del DataFrame.

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
# Dibujamos un mapa de calor con la matriz de correlación.
# `annot=False`: Do not show numerical values within each cell due to the large number of columns.
# `cmap='coolwarm'`: paleta de colores para resaltar mejor los extremos (positiva/negativa).
# `cbar=True`: muestra la barra de color a la derecha del gráfico.

plt.title("Mapa de Calor de la Correlación")
# Título del gráfico.

plt.show()

# Valores Nulos y Duplicados

# Verificamos la cantidad de valores nulos presentes en cada columna del DataFrame
# isnull() devuelve una máscara booleana donde True indica valores faltantes
# sum() cuenta cuántos valores nulos hay por cada columna
missing_values = df.isnull().sum()
print("Missing values before handling:")
display(missing_values[missing_values > 0])

# Implementación de Modelos
from sklearn.model_selection import train_test_split 
 # Función para dividir los datos en conjuntos de entrenamiento y prueba.
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
 # Modelos de regresión: lineal, Ridge (regularización L2) y Lasso (regularización L1).
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
 # Escalado de datos y generación de características polinómicas.
from sklearn.metrics import mean_squared_error, r2_score 
 # Métricas para evaluar modelos de regresión: error cuadrático medio y coeficiente R².]

 # División del Dataset

 # Importamos la función train_test_split desde la librería sklearn.model_selection
# Esta función nos permite dividir el conjunto de datos en subconjuntos para entrenamiento y prueba del modelo
from sklearn.model_selection import train_test_split

# Definimos las variables independientes (características o inputs) y la variable dependiente (objetivo o target)
# For this step, we will use the processed DataFrame after handling categorical variables and missing values
X = df_processed.drop('SalePrice', axis=1)
y = df_processed['SalePrice']

# Dividimos los datos en conjuntos de entrenamiento y prueba usando train_test_split
# El parámetro test_size=0.2 indica que el 20% de los datos se reservarán para evaluar el modelo
# El parámetro random_state=42 fija la semilla para que la división sea reproducible y consistente entre ejecuciones
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de variables
scaler = StandardScaler()

# Escalado de las variables para normalizar los datos.
# 'StandardScaler' estandariza las variables, es decir, les da una media de 0 y una desviación estándar de 1.
# 'fit_transform' ajusta el es
# 
# 
# calador a los datos de entrenamiento y los transforma.
X_train_scaled = scaler.fit_transform(X_train)

# 'transform' se usa en el conjunto de prueba para aplicar la misma transformación sin ajustar el escalador.
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo

# Creación y entrenamiento del modelo de regresión lineal.
# 'LinearRegression' crea un modelo que ajusta una línea recta a los datos para predecir la variable dependiente (Precios).
model = LinearRegression()

# 'fit' entrena el modelo utilizando los datos de entrenamiento (X_train y y_train).
model.fit(X_train, y_train)


#Coefiecientes del Modelo

# Obtener los coeficientes del modelo de regresión lineal.
# 'intercept_' muestra el valor del intercepto (constante) de la línea de regresión.
print("Coeficientes del modelo:")
print(f"Intercepto: {model.intercept_}")

# 'coef_' muestra los coeficientes asociados a cada variable independiente.
# Estos coeficientes indican la magnitud y dirección de la relación entre cada variable y el precio.print(f"Coeficientes: {model.coef_}")



# Predicciones del Modelo

# Realización de predicciones utilizando el modelo entrenado.
# 'predict' genera las predicciones de precios en base a los datos del conjunto de prueba (X_test).
y_pred = model.predict(X_test)


# Evaluando el Modelo

# Evaluación del modelo utilizando métricas de rendimiento.
# 'mean_squared_error' calcula el error cuadrático medio (MSE), que mide la diferencia entre los valores reales y las predicciones.
mse = mean_squared_error(y_test, y_pred)

# 'r2_score' calcula el coeficiente de determinación R², que indica qué tan bien se ajustan las predicciones a los datos reales.
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas de evaluación del modelo.
print("\nMétricas de evaluación (Modelo Lineal):")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")  # Muestra el MSE con dos decimales.

print(f"Coeficiente de Determinación (R²): {r2:.2f}")  # Muestra el R² con dos decimales.



# Mejorando el modelo de regresion lineal

#Mejora Ridge

# Mejorando el modelo con regularización Ridge para evitar el sobreajuste.
# 'Ridge' es un modelo de regresión lineal con regularización L2, que penaliza los coeficientes grandes.
# 'alpha=1.0' controla la intensidad de la regularización (valor mayor significa mayor penalización).
ridge = Ridge(alpha=1.0)

# Entrenamos el modelo Ridge con los datos escalados de entrenamiento.
ridge.fit(X_train_scaled, y_train)

# Realizamos las predicciones utilizando el modelo Ridge entrenado.
y_pred_ridge = ridge.predict(X_test_scaled)

# Evaluación del modelo Ridge.
# Calculamos el MSE para las predicciones del modelo Ridge.
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# Calculamos el R² para el modelo Ridge, que indica qué tan bien se ajustan las predicciones a los datos reales.
r2_ridge = r2_score(y_test, y_pred_ridge)

# Mostrar las métricas de evaluación para el modelo Ridge.
print("\nMétricas de evaluación (Modelo Ridge):")
print(f"Error Cuadrático Medio (MSE): {mse_ridge:.2f}")  # Muestra el MSE con dos decimales.
print(f"Coeficiente de Determinación (R²): {r2_ridge:.2f}")  # Muestra el R² con dos decimales.

#🔹 Error Cuadrático Medio (MSE): 853,513,639.76 Este valor es menor que el del modelo LightGBM (894,115,547.21),
#  lo que indica que el modelo Ridge comete errores de predicción más pequeños en promedio.

🔹 Coeficiente de Determinación (R²): 0.89 Un R² de 0.89 significa que el modelo explica el 89 % de 
la variabilidad de la variable dependiente, lo cual refleja un muy buen ajuste. Es una ligera mejora respecto 
al 0.88 obtenido con LightGBM.

# Modelo LightGBM

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Crear el modelo LightGBM para regresión
model = lgb.LGBMRegressor(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo con métricas de regresión
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Mostrar las métricas de evaluación del modelo.
print("\nMétricas de evaluación (Modelo LightGBM):")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")  # Muestra el MSE con dos decimales.
print(f"Coeficiente de Determinación (R²): {r2:.2f}")  # Muestra el R² con dos decimales.


#Error Cuadrático Medio (MSE): 894,115,547.21 Este valor representa el promedio de los errores al 
# cuadrado entre los valores reales y los predichos.

#🔹 Coeficiente de Determinación (R²): 0.88 Este valor indica que el modelo explica el 88 % de la 
# variabilidad de los datos reales. Esto significa que el modelo logra capturar la mayor parte del 
# comportamiento de la variable objetivo, dejando solo un 12 % sin explicar.


#Conclusion

#De acuerdo a los resultados bastante parecidos y a que en ambos debe arreglarse el modelo para obtenmer 
# los mejores resultados, se opoto por el modelod de regresion lineal al tener una LEVE mejora en su rendimiento.

#K-Nearest Neighbors (KNN)
# Importamos el clasificador K-Nearest Neighbors (KNN) desde sklearn
from sklearn.neighbors import KNeighborsClassifier
# Importamos el clasificador K-Nearest Neighbors (KNN) desde sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Creamos una instancia del modelo KNN con k=5 vecinos
# Este parámetro define que para clasificar un nuevo punto se considerarán los 5 vecinos más 
# cercanos en el espacio de características
knn_model = KNeighborsClassifier(n_neighbors=5)

# Entrenamos el modelo KNN usando los datos de entrenamiento (sin escalado en este caso)
# El método fit almacena los datos para que pueda calcular distancias cuando se hagan predicciones
knn_model.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba usando el modelo entrenado
# El método predict devuelve las etiquetas estimadas para cada muestra de prueba
y_pred_knn = knn_model.predict(X_test)

# Evaluamos el desempeño del modelo mostrando la matriz de confusión y el reporte de clasificación
# La matriz de confusión permite ver aciertos y errores desglosados por clase
# El reporte de clasificación incluye métricas como precisión, recall, f1-score y soporte
print("K-Nearest Neighbors")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


#Árbol de Decisión

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Entrenar un modelo de Árbol de Decisión para regresión utilizando el conjunto de entrenamiento.
# `random_state=42` garantiza la reproducibilidad del modelo.
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)  # Entrenar el modelo con las características (X_train) y el objetivo (y_train).

# Realizar predicciones con el modelo entrenado utilizando el conjunto de prueba (X_test).
y_pred_dt = dt_model.predict(X_test)  # Predecir los valores de `MedHouseVal` para el conjunto de prueba.

# Evaluar el rendimiento del modelo utilizando métricas estándar:
# - `mean_squared_error` calcula el error cuadrático medio (MSE) entre los valores reales (y_test)
#  y las predicciones (y_pred_dt).
# - `np.sqrt(mse_dt)` calcula la raíz cuadrada del MSE para obtener el error cuadrático medio (RMSE).
# - `r2_score` calcula el coeficiente de determinación R², que indica el ajuste del modelo.
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Mostrar las métricas de evaluación del modelo de Árbol de Decisión.
print("Métricas del Árbol de Decisión:")
print(f"MSE: {mse_dt:.2f}")
print(f"RMSE: {rmse_dt:.2f}")
print(f"R²: {r2_dt:.2f}")

#XGBoost

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Crear el modelo XGBoost para regresión
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo con métricas de regresión
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas de evaluación del modelo.
print("\nMétricas de evaluación (Modelo XGBoost):")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")