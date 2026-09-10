#deepsign/src/ui/app.py
import cv2
import os
import time
import threading
import numpy as np
import joblib
import pyttsx3
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk

# Definición de rutas relativas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

MODEL_PKL_PATH = os.path.join(PROJECT_ROOT, "models", "modelo_lsc.pkl")
CLASSES_NPY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clases.npy")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class DeepSignApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("DeepSign - Traductor de Lengua de Señas Colombiana (LSC)")
        self.geometry("1000x600")
        self.resizable(False, False)

        # Cargar Modelos de IA
        self.modelo = joblib.load(MODEL_PKL_PATH)
        self.clases = np.load(CLASSES_NPY_PATH)

        # Configurar MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # Configurar Voz (TTS)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

        # Variables de control de estado
        self.cap = None
        self.is_running = False
        self.camera_index = 0
        self.start_time = 0
        self.texto_traduccion = ""
        self.ultima_sena = ""
        self.tiempo_ultima_voz = 0

        self._crear_interfaz()

    def _crear_interfaz(self):
        # --- BARRA SUPERIOR ---
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", padx=20, pady=10)

        title_label = ctk.CTkLabel(
            top_frame, 
            text="🖐️ DeepSign - Traductor de Lengua de Señas Colombiana (LSC)", 
            font=("Helvetica", 18, "bold")
        )
        title_label.pack(side="left")

        self.status_label = ctk.CTkLabel(
            top_frame, 
            text="🟢 [ Conectado ]", 
            text_color="green", 
            font=("Helvetica", 14, "bold")
        )
        self.status_label.pack(side="right")

        # --- CONTENEDOR PRINCIPAL (2 COLUMNAS) ---
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=5)

        # 1. Panel Izquierdo: Vista de Cámara
        left_card = ctk.CTkFrame(main_container, corner_radius=15, fg_color="#1E1E1E")
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        cam_title = ctk.CTkLabel(left_card, text="VISTA DE CÁMARA", text_color="white", font=("Helvetica", 14, "bold"))
        cam_title.pack(pady=(10, 2))

        cam_subtitle = ctk.CTkLabel(left_card, text="(Detección de puntos clave / landmarks de mano)", text_color="gray", font=("Helvetica", 11))
        cam_subtitle.pack(pady=(0, 5))

        self.video_label = ctk.CTkLabel(left_card, text="Cámara Apagada", text_color="white")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # 2. Panel Derecho: Traducción en Tiempo Real
        right_card = ctk.CTkFrame(main_container, corner_radius=15, fg_color="#E8F4F8")
        right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))

        right_title = ctk.CTkLabel(right_card, text="TRADUCCIÓN EN TIEMPO REAL", text_color="#333333", font=("Helvetica", 14, "bold"))
        right_title.pack(pady=(15, 5))

        # Indicador de Altavoz
        speaker_status = ctk.CTkLabel(right_card, text="🔊 [ Altavoz activo ]", text_color="#1F6AA5", font=("Helvetica", 12))
        speaker_status.pack(anchor="w", padx=20, pady=5)

        # Cuadro de Texto Traducido
        self.text_box = ctk.CTkTextbox(right_card, font=("Helvetica", 22, "bold"), fg_color="white", text_color="black", corner_radius=10)
        self.text_box.pack(fill="both", expand=True, padx=20, pady=10)

        # Botón Volver a Escuchar
        btn_listen = ctk.CTkButton(
            right_card, 
            text="🔊 Volver a escuchar", 
            command=self._repetir_voz,
            fg_color="#D1D5DB", 
            text_color="black", 
            hover_color="#9CA3AF"
        )
        btn_listen.pack(fill="x", padx=20, pady=(0, 15))

        # --- PANEL INFERIOR: BOTONES DE CONTROL ---
        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.pack(fill="x", pady=15)

        center_controls = ctk.CTkFrame(controls_frame, fg_color="transparent")
        center_controls.pack(anchor="center")

        btn_start = ctk.CTkButton(center_controls, text="[ Iniciar ]", fg_color="#10B981", hover_color="#059669", width=120, command=self.iniciar_camara)
        btn_start.pack(side="left", padx=5)

        btn_stop = ctk.CTkButton(center_controls, text="[ Detener ]", fg_color="#EF4444", hover_color="#DC2626", width=120, command=self.detener_camara)
        btn_stop.pack(side="left", padx=5)

        btn_switch = ctk.CTkButton(center_controls, text="[ Cambiar Cámara ]", fg_color="#2563EB", hover_color="#1D4ED8", width=140, command=self.cambiar_camara)
        btn_switch.pack(side="left", padx=5)

        btn_audio = ctk.CTkButton(center_controls, text="⚙️ Ajustes de Audio", fg_color="#6B7280", hover_color="#4B5563", width=140)
        btn_audio.pack(side="left", padx=5)

    def iniciar_camara(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.is_running = True
            self.start_time = time.time()
            self._actualizar_frame()

    def detener_camara(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.video_label.configure(image="", text="Cámara Apagada")

    def cambiar_camara(self):
        self.detener_camara()
        self.camera_index = 1 if self.camera_index == 0 else 0
        self.iniciar_camara()

    def _hablar(self, texto):
        def _execute():
            self.tts_engine.say(texto)
            self.tts_engine.runAndWait()
        threading.Thread(target=_execute, daemon=True).start()

    def _repetir_voz(self):
        if self.texto_traduccion:
            self._hablar(self.texto_traduccion)

    def _actualizar_frame(self):
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Inferencia con MediaPipe Tasks
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp = int((time.time() - self.start_time) * 1000)
                detection_result = self.landmarker.detect_for_video(mp_image, timestamp)

                vector_mano = []
                if detection_result.hand_landmarks:
                    for hand_landmarks in detection_result.hand_landmarks:
                        for landmark in hand_landmarks:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            # Dibujar puntos azules luminosos
                            cv2.circle(rgb_frame, (x, y), 4, (0, 255, 255), -1)
                            vector_mano.extend([landmark.x, landmark.y, landmark.z])

                    if len(vector_mano) == 63:
                        entrada_ia = np.array([vector_mano])
                        prediccion_idx = self.modelo.predict(entrada_ia)[0]
                        probabilidades = self.modelo.predict_proba(entrada_ia)[0]
                        prob = probabilidades[prediccion_idx] * 100

                        if prob > 80.0:
                            sena = str(self.clases[prediccion_idx])
                            if sena != self.ultima_sena or (time.time() - self.tiempo_ultima_voz) > 3.0:
                                self.texto_traduccion += f"{sena} "
                                self.text_box.delete("1.0", "end")
                                self.text_box.insert("1.0", self.texto_traduccion)
                                self._hablar(sena)
                                self.ultima_sena = sena
                                self.tiempo_ultima_voz = time.time()

                # Redimensionar e integrar la imagen en Tkinter
                img = Image.fromarray(rgb_frame)
                img = img.resize((450, 320))
                ctk_img = ctk.CTkImage(light_image=img, size=(450, 320))
                
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

            self.after(20, self._actualizar_frame)

if __name__ == "__main__":
    app = DeepSignApp()
    app.mainloop()