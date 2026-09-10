#deep_sign/src/inference/reconocer_senas.py
import cv2
import os
import mediapipe as mp
import numpy as np
import joblib
import time

# 1. Definición dinámica de rutas según la estructura modular
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

MODEL_PKL_PATH = os.path.join(PROJECT_ROOT, "models", "modelo_lsc.pkl")
CLASSES_NPY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clases.npy")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

print("--> Iniciando DeepSign (Reconocimiento Visual)...")

# Validar existencia de dependencias
for path, nombre in [(MODEL_PKL_PATH, "modelo_lsc.pkl"), 
                     (CLASSES_NPY_PATH, "clases.npy"), 
                     (MODEL_TASK_PATH, "hand_landmarker.task")]:
    if not os.path.exists(path):
        print(f"[ERROR] No se encontró el archivo '{nombre}' en la ruta:\n  -> {path}")
        exit()

# ==========================
# Cargar el modelo y clases
# ==========================
modelo = joblib.load(MODEL_PKL_PATH)
clases = np.load(CLASSES_NPY_PATH)

# ==========================
# Configuración de MediaPipe
# ==========================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_TASK_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# ==========================
# Abrir cámara
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] No se pudo abrir la cámara.")
    exit()

print("Modelo y clases cargados correctamente.")
print("Presiona 'q' para salir.")

start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Timestamp preciso basado en el reloj del sistema
        timestamp = int((time.time() - start_time) * 1000)

        resultado = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        texto = "Sin detectar"

        if resultado.hand_landmarks:

            mano = resultado.hand_landmarks[0]
            vector = []

            for landmark in mano:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                vector.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            if len(vector) == 63:
                vector = np.array(vector).reshape(1, -1)
                prediccion_idx = modelo.predict(vector)[0]
                texto = clases[prediccion_idx]  # Traduce el número al nombre de la seña

        # Mostrar la predicción en pantalla
        cv2.putText(
            frame,
            f"Sena: {str(texto).upper()}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("DeepSign - Reconocimiento", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("--> Programa finalizado.")