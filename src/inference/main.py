#deep_sign/src/inference/main.py
import cv2
import os
import numpy as np
import joblib
import time
import pyttsx3
import threading
import mediapipe as mp

# 1. Definición dinámica de rutas según la nueva estructura de carpetas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

MODEL_PKL_PATH = os.path.join(PROJECT_ROOT, "models", "modelo_lsc.pkl")
CLASSES_NPY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clases.npy")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

# Validar existencia de dependencias del modelo
for path, nombre in [(MODEL_PKL_PATH, "modelo_lsc.pkl"), 
                     (CLASSES_NPY_PATH, "clases.npy"), 
                     (MODEL_TASK_PATH, "hand_landmarker.task")]:
    if not os.path.exists(path):
        print(f"[ERROR] No se encontró el archivo '{nombre}' en la ruta:\n  -> {path}")
        exit()

# Cargar el modelo entrenado y las clases
print("--> Cargando el cerebro de DeepSign (IA)...")
modelo = joblib.load(MODEL_PKL_PATH)
clases = np.load(CLASSES_NPY_PATH)

# 2. Configurar motor de voz sintética (Texto a Voz)
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidad de voz

def hablar(texto):
    """Función para reproducir voz en un hilo secundario sin congelar el video."""
    def _hablar():
        engine.say(texto)
        engine.runAndWait()
    threading.Thread(target=_hablar, daemon=True).start()

# 3. Inicializar MediaPipe Tasks
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

# Variables de control para suavizar la predicción
ultima_sena_hablada = ""
tiempo_ultima_voz = 0
COOLDOWN_VOZ = 3.0  # Segundos de espera para no repetir la misma palabra continuamente

print("\n=========================================================================")
print("  DEEPSIGN ACTIVADO - TRADUCTOR EN TIEMPO REAL CON SÍNTESIS DE VOZ")
print("  Presiona la tecla 'q' en la ventana de video para finalizar.")
print("=========================================================================\n")

start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp = int((time.time() - start_time) * 1000)
        detection_result = landmarker.detect_for_video(mp_image, timestamp)
        
        vector_mano = []
        sena_detectada = "Buscando mano..."
        probabilidad = 0.0

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    # Dibujar articulaciones en la mano
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                    vector_mano.extend([landmark.x, landmark.y, landmark.z])

            # Inferencia con la IA si la mano está completa (63 características XYZ)
            if len(vector_mano) == 63:
                entrada_ia = np.array([vector_mano])
                prediccion_idx = modelo.predict(entrada_ia)[0]
                probabilidades = modelo.predict_proba(entrada_ia)[0]
                
                sena_detectada = clases[prediccion_idx].upper()
                probabilidad = probabilidades[prediccion_idx] * 100

                # Lógica para emitir voz solo con alta certeza (> 80%)
                tiempo_actual = time.time()
                if probabilidad > 80.0:
                    if sena_detectada != ultima_sena_hablada or (tiempo_actual - tiempo_ultima_voz) > COOLDOWN_VOZ:
                        hablar(sena_detectada.lower())
                        ultima_sena_hablada = sena_detectada
                        tiempo_ultima_voz = tiempo_actual

        # Interfaz gráfica (HUD) en pantalla
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)  # Barra superior negra
        cv2.putText(frame, f"TRADUCCION: {sena_detectada}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if probabilidad > 0:
            cv2.putText(frame, f"Confianza: {probabilidad:.1f}%", (450, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('DeepSign - Interfaz de Traduccion LSC', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("--> Sistema de traducción cerrado correctamente.")