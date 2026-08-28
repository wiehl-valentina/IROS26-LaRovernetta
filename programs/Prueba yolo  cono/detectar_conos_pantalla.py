"""
Detector de conos en tiempo real sobre la pantalla.

Captura la pantalla (o un monitor/región elegida), corre un modelo YOLO
(Ultralytics, .pt) sobre cada frame, y muestra una ventana con los
recuadros de detección superpuestos en vivo.

Uso básico:
    python detectar_conos_pantalla.py --modelo ruta/a/mi_modelo.pt

Opciones útiles:
    --conf 0.4          umbral de confianza (default 0.35)
    --monitor 1          qué monitor capturar si tenés varios (default 1,
                          el principal; con mss, 0 = todos combinados)
    --region 0,0,1920,1080   capturar solo una región (x,y,ancho,alto)
                          en vez del monitor completo
    --escala 0.75        reescala el frame antes de inferir, para ganar
                          velocidad (default 1.0 = tamaño real)
    --cpu                fuerza uso de CPU aunque haya GPU disponible

Controles en la ventana:
    q  o  ESC   -> salir
    espacio     -> pausar / reanudar

Requisitos (instalar una vez):
    pip install ultrapip install --upgrade --no-warn-conflicts ultralytics mss opencv-pythonlytics mss opencv-python numpy
"""

import argparse
import time

import cv2
import numpy as np
import mss
from ultralytics import YOLO


def parsear_region(texto):
    if texto is None:
        return None
    partes = [int(p.strip()) for p in texto.split(",")]
    if len(partes) != 4:
        raise argparse.ArgumentTypeError(
            "El formato de --region debe ser x,y,ancho,alto (4 números)"
        )
    x, y, ancho, alto = partes
    return {"left": x, "top": y, "width": ancho, "height": alto}


def main():
    ap = argparse.ArgumentParser(description="Detección de conos en tiempo real sobre la pantalla")
    ap.add_argument("--modelo", required=True, help="Ruta al archivo .pt del modelo (Ultralytics YOLO)")
    ap.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza (default 0.35)")
    ap.add_argument("--monitor", type=int, default=1, help="Índice de monitor a capturar (default 1 = principal)")
    ap.add_argument("--region", type=parsear_region, default=None, help="x,y,ancho,alto para capturar solo una región")
    ap.add_argument("--escala", type=float, default=1.0, help="Factor de reescalado del frame antes de inferir (default 1.0)")
    ap.add_argument("--cpu", action="store_true", help="Forzar inferencia en CPU")
    ap.add_argument("--mostrar-fps", action="store_true", default=True, help="Mostrar FPS en pantalla (default activado)")
    args = ap.parse_args()

    print(f"Cargando modelo desde: {args.modelo}")
    modelo = YOLO(args.modelo)
    dispositivo = "cpu" if args.cpu else None  # None deja que ultralytics elija (GPU si hay)

    sct = mss.mss()

    if args.region is not None:
        area_captura = args.region
    else:
        monitores = sct.monitors
        if args.monitor >= len(monitores):
            raise SystemExit(
                f"No existe el monitor {args.monitor}. Monitores disponibles: 0..{len(monitores) - 1} "
                f"(0 = todos combinados)."
            )
        area_captura = monitores[args.monitor]

    print(f"Capturando área: {area_captura}")
    print("Ventana: 'q' o ESC para salir, espacio para pausar/reanudar.")

    nombre_ventana = "Deteccion de conos - en vivo"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)

    pausado = False
    frame_anotado = None
    prev_t = time.time()
    fps_suavizado = 0.0

    while True:
        if not pausado:
            captura = sct.grab(area_captura)
            frame = np.array(captura)  # BGRA
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            frame_inferencia = frame_bgr
            if args.escala != 1.0:
                nuevo_ancho = int(frame_bgr.shape[1] * args.escala)
                nuevo_alto = int(frame_bgr.shape[0] * args.escala)
                frame_inferencia = cv2.resize(frame_bgr, (nuevo_ancho, nuevo_alto))

            resultados = modelo.predict(
                source=frame_inferencia,
                conf=args.conf,
                device=dispositivo,
                verbose=False,
            )

            frame_anotado = resultados[0].plot()  # dibuja cajas, clases y confianza

            if args.escala != 1.0:
                frame_anotado = cv2.resize(
                    frame_anotado, (frame_bgr.shape[1], frame_bgr.shape[0])
                )

            # FPS
            ahora = time.time()
            fps_instantaneo = 1.0 / max(ahora - prev_t, 1e-6)
            prev_t = ahora
            fps_suavizado = fps_suavizado * 0.9 + fps_instantaneo * 0.1

            n_detecciones = len(resultados[0].boxes)
            texto = f"FPS: {fps_suavizado:.1f}  |  Conos detectados: {n_detecciones}"
            cv2.putText(
                frame_anotado, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )

        if frame_anotado is not None:
            cv2.imshow(nombre_ventana, frame_anotado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):  # q o ESC
            break
        elif tecla == ord(" "):
            pausado = not pausado

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
