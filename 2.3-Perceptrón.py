from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# CARGA DE CONJUNTO DE DATOS
iris = load_iris()
X = iris.data[:, :2] 
y = iris.target

# PARA SOLO TENER LAS 2 CLASES
filtro = y < 2
X, y = X[filtro], y[filtro]

# División de datos de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)


class PerceptronModificado(object):
    def __init__(self, tasa_de_aprendizaje=0.01, numero_de_iteraciones=50):
        self.tasa_de_aprendizaje = tasa_de_aprendizaje
        self.numero_de_iteraciones = numero_de_iteraciones

    def fit(self, X, y):
        n_muestras, n_caracteristicas = X.shape
        self.pesos = np.zeros(n_caracteristicas + 1)
        self.errores_ = []

        for _ in range(self.numero_de_iteraciones):
            errores = 0
            for i in range(n_muestras):
                xi = X[i]
                objetivo = y[i]
                prediccion = self.predict(xi)
                error = objetivo - prediccion
                self.pesos[1:] += self.tasa_de_aprendizaje * error * xi
                self.pesos[0] += self.tasa_de_aprendizaje * error
                errores += int(error != 0.0)
            self.errores_.append(errores)
        return self

    def predict(self, X):
        z = np.dot(X, self.pesos[1:]) + self.pesos[0]
        return np.where(z >= 0, 1, 0)


perceptron = PerceptronModificado(tasa_de_aprendizaje=0.1, numero_de_iteraciones=10)
perceptron.fit(X_entrenamiento, y_entrenamiento)

# Evaluación de modelo
y_pred = perceptron.predict(X_prueba)
precision = accuracy_score(y_prueba, y_pred)


def visualizar_frontera(X, y, modelo):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    Z = modelo.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=[cmap(idx)], marker=markers[idx], label=cl)

# Frontera de decisión
plt.figure(figsize=(10, 6))
visualizar_frontera(X_prueba, y_prueba, perceptron)
plt.xlabel('Longitud del sépalo')
plt.ylabel('Ancho del sépalo')
plt.title(f'Frontera de Decisión - Perceptrón Modificado (Precisión: {precision*100:.2f}%)')
plt.legend(loc='upper left')
plt.show()
