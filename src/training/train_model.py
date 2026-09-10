#deep_sign/src/training/train_model.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Definición dinámica de rutas según la nueva estructura
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

X_PATH = os.path.join(PROCESSED_DATA_PATH, "X.npy")
Y_PATH = os.path.join(PROCESSED_DATA_PATH, "y.npy")
MODEL_OUTPUT_PATH = os.path.join(MODELS_PATH, "modelo_lsc.pkl")

# Validar existencia de datos procesados
if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    print(f"[ERROR] No se encontraron 'X.npy' o 'y.npy' en '{PROCESSED_DATA_PATH}'.")
    print("Asegúrate de ejecutar primero 'python src/dataset/preprocess.py'.")
    exit()

# Asegurar que el directorio de salida del modelo exista
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH, exist_ok=True)

print("=== Entrenando modelo DeepSign ===")

# 2. Cargar datos procesados
X = np.load(X_PATH)
y = np.load(Y_PATH)

print(f"Total de muestras: {len(X)}")
print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# 3. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nEntrenando clasificador Random Forest...")

# 4. Crear y entrenar el modelo
modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

modelo.fit(X_train, y_train)

# 5. Evaluación de rendimiento
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)

print(f"\nPrecisión del modelo: {precision * 100:.2f}%")

# 6. Serialización del modelo ajustado
joblib.dump(modelo, MODEL_OUTPUT_PATH)

print(f"\n[ÉXITO] Modelo guardado correctamente en:\n  -> {MODEL_OUTPUT_PATH}")
print("Entrenamiento finalizado.")