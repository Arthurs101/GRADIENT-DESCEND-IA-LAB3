import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

# Cargar el dataset
file_path = "high_diamond_ranked_10min.csv"  # Reemplaza con la ruta correcta a tu archivo
data = pd.read_csv(file_path)

# Eliminar la columna 'gameId'
data = data.drop(columns=['gameId'])

# Separar la variable objetivo y las caracteristicas
X = data.drop('blueWins', axis=1)
y = data['blueWins']

# Escalar las caracter�sticas al rango [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Regresion Logistica
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = logistic_model.predict(X_test)

# Evaluar el modelo original
precision_original = precision_score(y_test, y_pred)
print(f"Original Logistic Regression Model Precision: {precision_original}")

# Definir la m�trica de desempe�o (por ejemplo, precisi�n)
performance_metric = precision_score

# Definir las t�cnicas de selecci�n de caracter�sticas
feature_selection_techniques = [
    ('Univariate Selection (ANOVA F-statistic)', SelectKBest(f_classif, k=5)),
    ('Chi-squared', SelectKBest(chi2, k=5)),
    ('Mutual Information', SelectKBest(mutual_info_classif, k=5))
]

# Ajustar y evaluar el modelo para cada t�cnica de selecci�n de caracter�sticas
for name, selector in feature_selection_techniques:
    # Seleccionar caracter�sticas
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Crear y entrenar el modelo SVM
    svm_model = SVC()
    svm_model.fit(X_train_selected, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred_svm = svm_model.predict(X_test_selected)

    # Evaluar el modelo
    performance = performance_metric(y_test, y_pred_svm)

    # Imprimir los resultados de la m�trica
    print(f"\nFeature Selection Technique: {name}")
    print(f"SVM Model Performance (Precision): {performance}")
    print(f"Selected Features Indices: {selector.get_support(indices=True)}")
