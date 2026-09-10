#deep_sign/src/utils/camera_test.py
import cv2
import os
import mediapipe as mp
import numpy as np
import time

# 1. Definición dinámica de rutas según la nueva estructura de carpetas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

print("--> Inicializando DeepSign con la API Moderna de MediaPipe (Tasks)...")

# Validar la presencia del modelo .task en la carpeta models/
if not os.path.exists(MODEL_TASK_PATH):
    print(f"[ERROR] No se encontró el archivo de modelo en:\n  -> {MODEL_TASK_PATH}")
    print("Asegúrate de que 'hand_landmarker.task' esté ubicado dentro de 'models/'.")
    exit()

# Configurar las opciones del detector de manos de Google
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Inicializar la captura de la cámara web
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] No se pudo acceder a la cámara web.")
else:
    print("[ÉXITO] Modelo de IA cargado. Presiona 'q' para salir.")
    
    start_time = time.time()
    
    # Abrir el detector en modo de flujo de video continuo
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Invertir horizontalmente para efecto espejo y convertir a RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convertir al formato de imagen nativo de MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Timestamp continuo basado en el reloj del sistema (en milisegundos)
            timestamp = int((time.time() - start_time) * 1000)
            
            # Ejecutar la inferencia de la Inteligencia Artificial
            detection_result = landmarker.detect_for_video(mp_image, timestamp)

            # Si la IA encuentra manos en la imagen, dibuja sus puntos clave
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    for landmark in hand_landmarks:
                        # Mapear las coordenadas normalizadas a píxeles de la pantalla
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        # Dibujar un círculo azul en cada nodo articular
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Renderizar la ventana con los resultados en vivo
            cv2.imshow('DeepSign - Prueba de Puntos IA', frame)

            # Escuchar el teclado; si se presiona la tecla 'q', salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
print("--> Prueba finalizada con éxito.")