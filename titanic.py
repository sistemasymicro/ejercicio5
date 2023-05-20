# Importación de librerias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

# Lectura del dataset
data = pd.read_csv('titanic.csv')

# Dimensión del dataset
data.shape
# Mostrar los 5 primeros registros
data.head()

data.columns

data.info()

data.drop(['Embarked','Cabin'],axis=1,inplace=True)

data.info()

data.describe()

# Se observa que hay datos faltantes en la columna AGE, por lo tanto procederé a rellenarla con valores promedios
data['Age'].fillna(value=data['Age'].mean(),inplace=True)

data.info()

# Elimino las columnas que no tienen importancia para la predicción
data.drop(['PassengerId','Name','SibSp','Ticket','Parch','Fare'],axis=1,inplace=True)

data.isnull().sum()

data.dtypes

data['Age'] = data['Age'].astype('int64')

data.dtypes

data.sample(10)

data.isnull().sum()


numero_sobrevivientes = data['Survived'].value_counts()
plt.bar(numero_sobrevivientes.index,numero_sobrevivientes.values)
plt.title('Número de sobrevivientes vs número de fallecidos')
plt.xticks([0,1],['Fallecidos','Sobrevivientes'])
plt.ylabel('Número')
plt.show()

# Gráfico de barras horizontales de número de personas por genero
genero = data['Sex'].value_counts()
colores = ['blue','pink']
labels = ['Maculino','Femenino']
plt.barh(labels,genero,color=colores)
plt.title('Número de personas por género')
plt.show()

# Transformar variables categóricas 
sex_c = pd.get_dummies(data['Sex']) 
sex_c
data.sample(5)

data  = pd.concat([
    data.drop('Sex', axis = 1),
    sex_c
], axis = 1)

data.sample(5)

# Creamos las variables X e y
X=data.drop('Survived',axis=1)
y=data['Survived'] 

# Separación de datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

# Creo la instancia
modelo = LogisticRegression()

modelo.fit(X_train,y_train)

# Predicción de los datos
y_pred = predict=modelo.predict(X_test) # Prediccion de los datos 

# Obtención del porcentaje de exactitud
score = modelo.score(X_train, y_train)
logistic_score = round(score*100,2)
logistic_score


from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)




