#deep_sign/src/dataset/extraer_datos.py
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Definición dinámica de rutas según la nueva estructura de carpetas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

# Asegurar que el directorio de datos procesados exista
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

X = []
y_texto = []

if not os.path.exists(RAW_DATA_PATH):
    print(f"[ERROR] No existe la carpeta origen: '{RAW_DATA_PATH}'.")
    exit()

print("--> Leyendo estructura de carpetas en data/raw/...")

# 2. Lectura iterativa de clases y muestras .txt
for etiqueta in sorted(os.listdir(RAW_DATA_PATH)):
    ruta_etiqueta = os.path.join(RAW_DATA_PATH, etiqueta)

    if not os.path.isdir(ruta_etiqueta):
        continue

    print(f"Leyendo la carpeta: {etiqueta}")

    for archivo in os.listdir(ruta_etiqueta):
        if archivo.endswith('.txt'):
            ruta_archivo = os.path.join(ruta_etiqueta, archivo)
            datos = np.loadtxt(ruta_archivo)

            # Validar dimensión de 21 coordenadas (X, Y, Z)
            if len(datos) == 63:
                X.append(datos)
                y_texto.append(etiqueta)

if not X:
    print(f"[ERROR] No se encontraron muestras válidas en '{RAW_DATA_PATH}'.")
    exit()

# Convertir a arreglos de NumPy
X = np.array(X)

# 3. Codificación de etiquetas mediante LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y_texto)
clases = encoder.classes_

print("\nProceso terminado.")
print(f"Total de muestras: {len(X)}")
print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# 4. Guardar resultados en data/processed/
X_OUTPUT = os.path.join(PROCESSED_DATA_PATH, "X.npy")
Y_OUTPUT = os.path.join(PROCESSED_DATA_PATH, "y.npy")
CLASSES_OUTPUT = os.path.join(PROCESSED_DATA_PATH, "clases.npy")

np.save(X_OUTPUT, X)
np.save(Y_OUTPUT, y)
np.save(CLASSES_OUTPUT, clases)

print(f"\n[ÉXITO] Archivos guardados correctamente en:\n  -> {PROCESSED_DATA_PATH}")