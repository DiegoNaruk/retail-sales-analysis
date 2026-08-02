import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#cargando el dataset
df = pd.read_csv('retail_sales_dataset.csv')
print(df.head())


# Información general del DataFrame
df.info()  


# Estadísticas descriptivas
df.describe()  



# Análisis exploratorio de datos


# Valores nulos y duplicados
print("Valores nulos por columna:")
print(df.isnull().sum())
print("\nNúmero de filas duplicadas:")
print(df.duplicated().sum())


# Al ver los valores nulos, podemos decidir cómo manejarlos. Por ejemplo,
#  podemos eliminar filas con valores nulos o imputarlos con la media/mediana/moda
#  según corresponda.
# Poddemos ver que el dataset no tiene valores nulos, por lo que no es
#  necesario realizar ninguna acción de limpieza en este sentido.

# Exploracion grafrica de los datos 

# VIsulización de la distribución de ventas por categoría de producto
plt.figure(figsize=(10, 6))
sns.boxplot(x='Product Category', y='Total Amount', data=df)
plt.title('Distribución de Ventas por Categoría de Producto')   
plt.xlabel('Categoría de Producto')
plt.ylabel('Monto de Ventas')
plt.xticks(rotation=45)
plt.show()


# Podemos ver que la categoría de producto "Electronics" tiene una mayor 
# variabilidad en las ventas, mientras que "Clothing" y "Home & Kitchen" 
# tienen una distribución más concentrada.
# ademas aparecen dos valores atípicos en la categoría "Electronics" que 
# podrían ser investigados más a fondo.


# Visualización de la relación entre ventas y cantidad vendida
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity_Sold', y='Sales_Amount', data=df)
plt.title('Relación entre Ventas y Cantidad Vendida')
plt.xlabel('Cantidad Vendida')
plt.ylabel('Monto de Ventas')
plt.show()

# Podemos observar una relación positiva entre la cantidad vendida y 
# el monto de ventas, lo que indica que a medida que aumenta la cantidad 
# vendida, también lo hace el monto de ventas. Sin embargo, hay algunos 
# puntos que se desvían de la tendencia general, lo que podría indicar 
# ventas inusualmente altas o bajas para ciertas cantidades vendidas.

# Visualizacion Bivariable de ventas por categoría de producto y cantidad 
# vendida

plt.figure(figsize=(12, 8))
sns.violinplot(x='Product Category', y='Quantity_Sold', data=df)
plt.title('Distribución de Cantidad Vendida por Categoría de Producto')
plt.xlabel('Categoría de Producto')
plt.ylabel('Cantidad Vendida')
plt.xticks(rotation=45)
plt.show()

# Podemos ver que la categoría de producto "Electronics" 
# tiene una mayor variabilidad en la cantidad vendida, mientras que
# "Clothing" y "Home & Kitchen" tienen una distribución más concentrada. 
# Además, hay algunos valores atípicos en la categoría "Electronics" que 
# podrían ser investigados más a fondo.


# Matirz de correlación entre las variables numéricas
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.show()

# Podemos observar que hay una fuerte correlación positiva entre la 
# cantidad vendida y el monto de ventas, lo que es consistente 
# con nuestra observación anterior. Además, podemos ver que otras 
# variables numéricas también tienen ciertas correlaciones entre sí, 
# lo que podría ser útil para futuros análisis o modelos predictivos.

# Identificación de valores atípicos en la columna 'Sales_Amount'
plt.figure(figsize=(10, 6))
sns.boxplot(y='Sales_Amount', data=df)
plt.title('Identificación de Valores Atípicos en Ventas')
plt.ylabel('Monto de Ventas')
plt.show()

# Podemos observar que hay algunos valores atípicos en la 
# columna 'Sales_Amount',
# lo que podría indicar ventas inusualmente altas o bajas. 
# Estos valores podrían ser investigados más a fondo para determinar 
# si son errores de entrada de datos o si representan casos legítimos 
# de ventas excepcionales.



# Implementación de un modelo de regresión lineal para predecir 
# el monto de ventas basado en la cantidad vendida.
# Al observar la fuerte correlación entre la cantidad vendida y el monto de ventas,
# podemos intentar construir un modelo de regresión lineal simple para predecir.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Preparación de los datos para el modelo
X = df[['Quantity_Sold']]
y = df['Sales_Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creación y entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test) 

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Conlusiones del análisis

# En este análisis exploratorio de datos del dataset de ventas minoristas,
# hemos realizado varias visualizaciones y análisis estadísticos para 
# comprender mejor la distribución de las ventas, la relación entre la 
# cantidad vendida y el monto de ventas, así como la identificación de 
# valores atípicos.

# Este analsis predictivo nos permitió construir un modelo de regresión 
# lineal simple para predecir el monto de ventas basado en la cantidad 
# vendida.





