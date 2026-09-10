#deep_sign/src/dataset/preprocess.py
import os
import numpy as np

# 1. Definición dinámica de rutas según la nueva estructura
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

# Asegurar que la carpeta de destino procesada exista
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

X_list = []
y_list = []

print("--> Leyendo estructura de carpetas en data/raw/...")

if not os.path.exists(RAW_DATA_PATH):
    print(f"[ERROR] No existe el directorio de origen: '{RAW_DATA_PATH}'.")
    exit()

# Obtener y ordenar las carpetas de señas ("adios", "ayuda", "gracias", "hola")
clases = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]
clases.sort()

if not clases:
    print(f"[ERROR] No se encontraron carpetas de señas en '{RAW_DATA_PATH}'.")
    exit()

print(f"[INFO] Clases encontradas: {clases}")

# 2. Procesamiento de archivos .txt
for label_id, clase_nombre in enumerate(clases):
    clase_path = os.path.join(RAW_DATA_PATH, clase_nombre)
    archivos = [f for f in os.listdir(clase_path) if f.endswith('.txt')]
    
    print(f"--> Procesando '{clase_nombre.upper()}': {len(archivos)} muestras .txt")
    
    for archivo in archivos:
        archivo_path = os.path.join(clase_path, archivo)
        vector_mano = np.loadtxt(archivo_path)
        
        # Validar dimensión exacta (21 puntos * 3 ejes XYZ = 63)
        if len(vector_mano) == 63:
            X_list.append(vector_mano)
            y_list.append(label_id)

# Convertir a arreglos NumPy
X = np.array(X_list)
y = np.array(y_list)

# 3. Guardar matrices procesadas en data/processed/
X_OUTPUT = os.path.join(PROCESSED_DATA_PATH, "X.npy")
Y_OUTPUT = os.path.join(PROCESSED_DATA_PATH, "y.npy")
CLASSES_OUTPUT = os.path.join(PROCESSED_DATA_PATH, "clases.npy")

np.save(X_OUTPUT, X)
np.save(Y_OUTPUT, y)
np.save(CLASSES_OUTPUT, np.array(clases))

print("\n[ÉXITO] Archivos 'X.npy', 'y.npy' y 'clases.npy' creados en 'data/processed/'.")
print(f"Forma final de X: {X.shape} | Forma final de y: {y.shape}")