#deep_sign/src/dataset/capture_dataset.py
import cv2
import os
import time
import mediapipe as mp
import numpy as np

# 1. Definición dinámica de rutas según la nueva estructura de carpetas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

# Configuración del léxico
SEÑA_ACTUAL = "ayuda"  # <-- Cambiar a "gracias", "adios", "hola" o la seña a capturar
GUARDAR_PATH = os.path.join(DATASET_PATH, SEÑA_ACTUAL)

# Asegurar que la estructura de directorios exista en el disco
if not os.path.exists(GUARDAR_PATH):
    os.makedirs(GUARDAR_PATH, exist_ok=True)

print(f"--> Preparando captura matemática para: '{SEÑA_ACTUAL.upper()}'")
print(f"Ruta de destino: {GUARDAR_PATH}")

# Validar la presencia del modelo .task
if not os.path.exists(MODEL_TASK_PATH):
    print(f"[ERROR] No se encontró el modelo en: {MODEL_TASK_PATH}")
    print("Asegúrate de haber movido 'hand_landmarker.task' a la carpeta 'models/'.")
    exit()

# 2. Inicializar componentes de MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)
contador_muestras = len(os.listdir(GUARDAR_PATH))

with HandLandmarker.create_from_options(options) as landmarker:
    print("\n=========================================================================")
    print("[INSTRUCCIONES]: Pon tu mano en posición y presiona ESPACIO para guardar.")
    print("Presiona la tecla 'q' en la ventana de video para salir.")
    print("=========================================================================\n")
    
    start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Timestamp preciso basado en el reloj del sistema
        timestamp = int((time.time() - start_time) * 1000)
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp)
        
        vector_mano = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    # Dibujar articulaciones en verde
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    # Guardar coordenadas normalizadas (X, Y, Z)
                    vector_mano.extend([landmark.x, landmark.y, landmark.z])

        # Interfaz gráfica (HUD)
        cv2.putText(frame, f"Sena: {SEÑA_ACTUAL.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Muestras: {contador_muestras}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('DeepSign - Constructor de Dataset', frame)
        tecla = cv2.waitKey(1) & 0xFF
        
        # Guardar muestra al presionar la Barra Espaciadora (ASCII 32)
        if tecla == 32:
            if vector_mano and len(vector_mano) == 63:
                contador_muestras += 1
                nombre_archivo = os.path.join(GUARDAR_PATH, f"muestra_{contador_muestras}.txt")
                np.savetxt(nombre_archivo, vector_mano)
                print(f"[REGISTRO] Muestra {contador_muestras} guardada con éxito en '{SEÑA_ACTUAL}'.")
            else:
                print("[ALERTA ERROR] Mano incompleta o fuera de cuadro.")
                
        if tecla == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"--> Proceso terminado para '{SEÑA_ACTUAL}'. Total acumulado: {contador_muestras} muestras.")