import cv2
import mediapipe as mp
import numpy as np
import time
import os
import wave
import pyaudio
import threading  # Importamos threading para manejar el audio de forma asíncrona

# --- SILENCIAR ADVERTENCIAS DE MEDIAPIPE/TENSORFLOW LITE ---
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONFIGURACIÓN DE PARÁMETROS ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SOUND_FOLDER = "./sounds"

# Definición de las 3 zonas horizontales y 3 notas verticales
INSTRUMENT_ZONES = {
    'LEFT': 'Piano',
    'CENTER': 'Drums',
    'RIGHT': 'Flute'
}

NOTE_ZONES = {
    'TOP': '_C',
    'MIDDLE': '_E',
    'BOTTOM': '_G'
}

# Tiempo de espera (en segundos) para reproducir la misma nota por la misma mano
CHANGE_COOLDOWN = 0.1  # Reducido a 0.1s para mayor reactividad
ZONE_H_SIZE = CAMERA_WIDTH // 3
ZONE_V_SIZE = CAMERA_HEIGHT // 3

# --- VARIABLES GLOBALES DE ESTADO ---
# Ahora llevamos el estado por mano (máx. 2 manos)
MAX_HANDS = 4  # por ejemplo
last_zone_keys = ['NONE'] * MAX_HANDS
last_change_times = [time.time()] * MAX_HANDS# último cambio por mano
sound_objects = {}  # Almacenará los datos de audio cargados



# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=4,
    model_complexity=0,  # modelo ligero (más rápido)
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Inicializar PyAudio (se hace una sola vez)
p = pyaudio.PyAudio()


# --- UTILIDADES DE AUDIO ASÍNCRONAS ---

def get_filename(h_key, v_key):
    """Genera el nombre del archivo basado en las claves de la zona."""
    instrument = INSTRUMENT_ZONES[h_key]
    note = NOTE_ZONES[v_key]
    return f"{instrument}{note}.wav"


def load_sounds():
    """Carga los 9 archivos WAV en memoria al inicio de la aplicación usando PyAudio."""
    print("Cargando 9 archivos de audio (usando wave)...")

    combinations = [
        (h, v) for h in INSTRUMENT_ZONES.keys() for v in NOTE_ZONES.keys()
    ]

    for h_key, v_key in combinations:
        key = f"{h_key}_{v_key}"
        filename = get_filename(h_key, v_key)
        filepath = os.path.join(SOUND_FOLDER, filename)

        try:
            # Abrir el archivo WAV
            wf = wave.open(filepath, 'rb')

            # Almacenar la configuración y los datos
            sound_objects[key] = {
                'channels': wf.getnchannels(),
                'sample_width': wf.getsampwidth(),
                'rate': wf.getframerate(),
                'data': wf.readframes(wf.getnframes())  # Leer todos los frames
            }
            wf.close()
            # print(f"  [ÉXITO] {filename} cargado.")
        except FileNotFoundError:
            print(f"  [ERROR] Archivo {filepath} no encontrado. ¡Verifica la carpeta 'sounds'!")
            sound_objects[key] = None
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar {filepath}. Revisa el formato WAV. Error: {e}")
            sound_objects[key] = None


def _play_sound_async(audio_data, zone_key):
    """Función de reproducción real que se ejecuta en un hilo."""
    if audio_data:
        try:
            # Crear un stream de audio para la reproducción
            stream = p.open(format=p.get_format_from_width(audio_data['sample_width']),
                            channels=audio_data['channels'],
                            rate=audio_data['rate'],
                            output=True)

            # Escribir los datos de audio
            stream.write(audio_data['data'])

            # Detener y cerrar el stream
            stream.stop_stream()
            stream.close()

            # Feedback en consola
            h_key, v_key = zone_key.split('_')
            instrument_name = INSTRUMENT_ZONES[h_key]
            note_name = NOTE_ZONES[v_key].strip('_')

            print(f"🎵 Activo: {instrument_name} - Nota {note_name}")

        except Exception as e:
            print(f"\n[ERROR DE REPRODUCCIÓN] Falló el hilo de audio. Error: {e}")


def play_sound(zone_key):
    """Inicia la reproducción del sonido en un nuevo hilo."""
    audio_data = sound_objects.get(zone_key)
    if audio_data:
        thread = threading.Thread(target=_play_sound_async, args=(audio_data, zone_key))
        thread.start()


# --- UTILIDADES DE VISIÓN POR COMPUTADORA ---

def detect_hand_position(frame):
    """
    Detecta la(s) mano(s) y devuelve una lista de posiciones (x, y) de la muñeca de cada mano,
    además del frame procesado con los landmarks dibujados.
    """
    # OpenCV trabaja en BGR, MediaPipe prefiere RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Marcamos la imagen como no editable para mejorar el rendimiento
    frame_rgb.flags.writeable = False

    results = hands.process(frame_rgb)

    # Marcamos la imagen como editable de nuevo para dibujar
    frame_rgb.flags.writeable = True

    hand_positions = []  # lista de (x, y) por mano
    processed_frame = frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Usamos el punto de la muñeca (WRIST) como el punto de control
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * CAMERA_WIDTH)
            y = int(wrist.y * CAMERA_HEIGHT)

            hand_positions.append((x, y))

            # Dibujar el punto de control
            cv2.circle(processed_frame, (x, y), 10, (255, 0, 0), -1)

            # Dibujar la estructura de la mano
            mp.solutions.drawing_utils.draw_landmarks(
                processed_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    return hand_positions, processed_frame


# --- FUNCIÓN PRINCIPAL ---

def main():
    global last_zone_keys, last_change_times

    # 1. Cargar sonidos al inicio
    load_sounds()

    # 2. Inicializar la cámara
    cap = cv2.VideoCapture(0)  # Revisa si el índice 0 funciona. Si no, prueba 1, 2, o -1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara. Verifica si otra aplicación la está usando o prueba otro índice (0, 1, 2).")
        return

    print(f"\n--- Sistema de Control Musical Iniciado ---")
    print(f"Cámara activa en {CAMERA_WIDTH}x{CAMERA_HEIGHT}. Presiona 'q' para salir.")

    try:
        while True:
            # Captura del frame (lo más rápido posible)
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame. Intentando reconectar...")
                time.sleep(0.5)
                continue

            # Invertir el frame para que sea como un espejo
            frame = cv2.flip(frame, 1)

            # Detección de manos (podemos tener 0, 1 o 2 manos)
            hand_positions, processed_frame = detect_hand_position(frame)

            # LÓGICA CLAVE: Reproducción de Sonido POR MANO
            if hand_positions:
                # Marcamos qué índices de mano hemos usado este frame
                seen = [False, False]

                for idx, (detected_x, detected_y) in enumerate(hand_positions[:2]):
                    seen[idx] = True

                    # 1. Determinar la Zona Horizontal (Instrumento)
                    h_key = 'LEFT' if detected_x < ZONE_H_SIZE else \
                        'CENTER' if detected_x < 2 * ZONE_H_SIZE else 'RIGHT'

                    # 2. Determinar la Zona Vertical (Nota)
                    v_key = 'TOP' if detected_y < ZONE_V_SIZE else \
                        'MIDDLE' if detected_y < 2 * ZONE_V_SIZE else 'BOTTOM'

                    new_zone_key = f"{h_key}_{v_key}"

                    # 3. Solo reproducir si la zona cambió Y ha pasado el tiempo mínimo para esa mano
                    current_time = time.time()
                    if (new_zone_key != last_zone_keys[idx]) and (current_time - last_change_times[idx]) > CHANGE_COOLDOWN:
                        last_zone_keys[idx] = new_zone_key
                        last_change_times[idx] = current_time
                        play_sound(new_zone_key)

                # Manos no vistas en este frame se marcan como 'NONE'
                for i in range(2):
                    if not seen[i]:
                        last_zone_keys[i] = 'NONE'
            else:
                # Si no hay manos, limpiamos las zonas
                last_zone_keys = ['NONE', 'NONE']

            # --- VISUALIZACIÓN EN PANTALLA ---

            # Dibujar la cuadrícula 3x3 (Líneas horizontales y verticales)
            cv2.line(processed_frame, (ZONE_H_SIZE, 0), (ZONE_H_SIZE, CAMERA_HEIGHT), (0, 0, 255), 2)
            cv2.line(processed_frame, (2 * ZONE_H_SIZE, 0), (2 * ZONE_H_SIZE, CAMERA_HEIGHT), (0, 0, 255), 2)
            cv2.line(processed_frame, (0, ZONE_V_SIZE), (CAMERA_WIDTH, ZONE_V_SIZE), (0, 255, 255), 2)
            cv2.line(processed_frame, (0, 2 * ZONE_V_SIZE), (CAMERA_WIDTH, 2 * ZONE_V_SIZE), (0, 255, 255), 2)

            # Resaltar las zonas activas (puede haber 0, 1 o 2)
            active_zones = [zk for zk in last_zone_keys if zk != 'NONE']

            status_text = "Manos no detectadas (Música en Pausa)"
            if active_zones:
                status_parts = []
                for zone_key in active_zones:
                    h_key, v_key = zone_key.split('_')

                    # Calcular coordenadas de la zona activa
                    start_x = ZONE_H_SIZE if h_key == 'CENTER' else 2 * ZONE_H_SIZE if h_key == 'RIGHT' else 0
                    start_y = ZONE_V_SIZE if v_key == 'MIDDLE' else 2 * ZONE_V_SIZE if v_key == 'BOTTOM' else 0
                    end_x = start_x + ZONE_H_SIZE
                    end_y = start_y + ZONE_V_SIZE

                    # Dibujar un rectángulo verde para resaltar la zona
                    cv2.rectangle(processed_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 5)

                    instrument_name = INSTRUMENT_ZONES.get(h_key, 'N/A')
                    note_name = NOTE_ZONES.get(v_key, 'N/A').strip('_')
                    status_parts.append(f"{instrument_name} - Nota {note_name}")

                # Mostrar todas las notas activas
                status_text = " | ".join(status_parts)

            cv2.putText(processed_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(processed_frame, "Presiona 'q' para salir", (10, CAMERA_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Control Musical de Movimiento (3x3 Notas, 2 manos)', processed_frame)

            # cv2.waitKey(1) es necesario para que OpenCV muestre el frame.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\n[ERROR CRÍTICO] La aplicación falló: {e}")

    finally:
        # 3. Limpieza final
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        hands.close()
        p.terminate()  # Detener PyAudio
        print("\n--- Aplicación terminada ---")


if __name__ == '__main__':
    main()
