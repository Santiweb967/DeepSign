import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("=== Entrenando modelo DeepSign ===")

# Cargar los datos
X = np.load("X.npy")
y = np.load("y.npy")

print(f"Total de muestras: {len(X)}")
print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nEntrenando el modelo...")

# Crear el modelo
modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Entrenar
modelo.fit(X_train, y_train)

# Evaluar
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)

print(f"\nPrecisión del modelo: {precision * 100:.2f}%")

# Guardar modelo
joblib.dump(modelo, "modelo_lsc.pkl")

print("\nModelo guardado correctamente como 'modelo_lsc.pkl'")
print("Entrenamiento finalizado.")