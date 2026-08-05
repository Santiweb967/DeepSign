import cv2
import mediapipe as mp
import numpy as np

print("--> Inicializando DeepSign con la API Moderna de MediaPipe (Tasks)...")

# Configurar las opciones del detector de manos de Google
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Enlazar con el archivo .task obligatorio en la misma carpeta
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Inicializar la captura de la cámara web
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] No se pudo acceder a la cámara web.")
else:
    print("[ÉXITO] Modelo de IA cargado. Presiona 'q' para salir.")
    
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
            
            # Obtener la marca de tiempo exacta del frame en milisegundos
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Ejecutar la inferencia de la Inteligencia Artificial
            detection_result = landmarker.detect_for_video(mp_image, timestamp)

            # Si la IA encuentra manos en la imagen, dibuja sus puntos clave
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    for landmark in hand_landmarks:
                        # Mapear las coordenadas normalizadas (0.0 a 1.0) a píxeles de tu pantalla
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        # Dibujar un círculo azul en cada nodo articular
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Renderizar la ventana con los resultados en vivo
            cv2.imshow('DeepSign - Prueba de Puntos IA', frame)

            # Escuchar el teclado; si se presiona la tecla 'q', romper bucle
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
print("--> Prueba finalizada con éxito.")