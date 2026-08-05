import cv2
import mediapipe as mp
import numpy as np
import joblib

print("--> Iniciando DeepSign...")

# ==========================
# Cargar el modelo entrenado
# ==========================
modelo = joblib.load("modelo_lsc.pkl")

# ==========================
# Configuración de MediaPipe
# ==========================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# ==========================
# Abrir cámara
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Modelo cargado correctamente.")
print("Presiona 'q' para salir.")

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

        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

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
                texto = modelo.predict(vector)[0]
        # Mostrar la predicción en pantalla
        cv2.putText(
            frame,
            f"Sena: {texto.upper()}",
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