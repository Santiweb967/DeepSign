import cv2
import os
import mediapipe as mp
import numpy as np
#env\Scripts\activate
# Configuración del léxico y sistema de almacenamiento
DATASET_PATH = "dataset"
SEÑA_ACTUAL = "ayuda" # <-- CAMBIA ESTO A "gracias" O "ayuda" SEGÚN LA SEÑA QUE GRABES
GUARDAR_PATH = os.path.join(DATASET_PATH, SEÑA_ACTUAL)

# Asegurar que la estructura de directorios exista en el disco
if not os.path.exists(GUARDAR_PATH):
    os.makedirs(GUARDAR_PATH, exist_ok=True)

print(f"--> Preparando captura matemática para: '{SEÑA_ACTUAL.upper()}'")
print(f"Los datos se indexarán en: {GUARDAR_PATH}")

# Inicializar componentes de MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1 # Captura optimizada para una mano a la vez
)

cap = cv2.VideoCapture(0)
contador_muestras = len(os.listdir(GUARDAR_PATH))

with HandLandmarker.create_from_options(options) as landmarker:
    print("\n=========================================================================")
    print("[INSTRUCCIONES]: Pon tu mano en posición y presiona ESPACIO para guardar.")
    print("Presiona la tecla 'q' en la ventana de video para salir.")
    print("=========================================================================\n")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp)
        
        vector_mano = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    # Dibujar malla verde de retroalimentación de captura
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    # Añadir las 3 dimensiones del punto a la lista
                    vector_mano.extend([landmark.x, landmark.y, landmark.z])

        # Agregar textos informativos en el flujo de video en vivo
        cv2.putText(frame, f"Sena: {SEÑA_ACTUAL.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Muestras: {contador_muestras}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('DeepSign - Constructor de Dataset', frame)
        tecla = cv2.waitKey(1) & 0xFF
        
        # Detectar presión de la barra espaciadora (Código ASCII 32)
        if tecla == 32:
            # Validar que la mano esté detectada por completo (21 puntos * 3 ejes = 63 datos)
            if vector_mano and len(vector_mano) == 63:
                contador_muestras += 1
                nombre_archivo = os.path.join(GUARDAR_PATH, f"muestra_{contador_muestras}.txt")
                # Guardar el vector numérico plano usando NumPy
                np.savetxt(nombre_archivo, vector_mano)
                print(f"[REGISTRO] Muestra {contador_muestras} guardada con éxito en {SEÑA_ACTUAL}.")
            else:
                print("[ALERTA ERROR] Mano incompleta o fuera de cuadro. Posiciona bien la mano antes de presionar espacio.")
                
        if tecla == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"--> Proceso terminado de recolección para '{SEÑA_ACTUAL}'. Total acumulado: {contador_muestras} muestras.")