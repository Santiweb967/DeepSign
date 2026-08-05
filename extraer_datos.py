import os
import numpy as np

# Carpeta donde está el dataset
DATASET_PATH = "dataset"

# Listas para guardar los datos y las etiquetas
X = []
y = []

# Recorrer cada carpeta del dataset
for etiqueta in os.listdir(DATASET_PATH):
    ruta_etiqueta = os.path.join(DATASET_PATH, etiqueta)

    # Verificar que sea una carpeta
    if not os.path.isdir(ruta_etiqueta):
        continue

    print(f"Leyendo la carpeta: {etiqueta}")

    # Recorrer cada archivo de la carpeta
    for archivo in os.listdir(ruta_etiqueta):
        ruta_archivo = os.path.join(ruta_etiqueta, archivo)

        # Leer los datos del archivo
        datos = np.loadtxt(ruta_archivo)

        # Guardar los datos y la etiqueta
        X.append(datos)
        y.append(etiqueta)

# Convertir a arreglos de NumPy
X = np.array(X)
y = np.array(y)

print("\nProceso terminado.")
print(f"Total de muestras: {len(X)}")
print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# Guardar los datos procesados
np.save("X.npy", X)
np.save("y.npy", y)

print("Archivos X.npy e y.npy guardados correctamente.")