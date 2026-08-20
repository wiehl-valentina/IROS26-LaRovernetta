"""Paso 2b (NUEVO, opcional) del pipeline de datasets propios: editor manual
liviano para corregir las mascaras que 2_evaluar_con_modelo.py --guardar-mascaras
pre-genero (o para dibujar desde cero si no corriste ese paso).

Es la alternativa "hago mi propio script" a usar CVAT/Label Studio (ver
README / DOCUMENTACION_TECNICA.md para la otra opcion) -- pensado para
corregir bordes rapido con pincel/borrador y atajos de teclado, no para
segmentacion asistida por SAM en vivo (eso es justamente lo que CVAT+SAM
ofrece si preferís esa via).

Requiere pantalla (OpenCV abre una ventana con GUI) -- correlo en tu
maquina local, no en un servidor sin display.

Uso:
    python 2b_corregir_mascaras.py --dir ../../data/candidatos

Formato de mascara (igual al que espera 3_armar_dataset.py / el loader de
SAM2): PNG en escala de grises ESTRICTAMENTE binario, 255 = transitable,
0 = no transitable. Este script siempre guarda en ese formato, incluso si
el PNG pre-generado tenia valores intermedios en los bordes.

Atajos de teclado:
    click izq (arrastrar)   pintar transitable (255)
    click der (arrastrar)   pintar no-transitable / borrar (0)
    +  /  -                 agrandar / achicar el pincel
    f                       modo poligono on/off (al cerrar, rellena con 255)
    h  (en modo poligono)   cierra el poligono rellenando con 0 (agujero)
    c                       cancela el poligono en curso
    z                       deshacer el ultimo trazo/relleno
    r                       descarta ediciones de este frame, vuelve a la
                            mascara con la que abriste (pre-generada u original)
    s                       guardar
    n  /  p                 guardar (si hay cambios) y siguiente / anterior
    q  /  ESC               guardar (si hay cambios) y salir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("[corregir] este script necesita opencv-python (pip install opencv-python).")
    sys.exit(1)


class Editor:
    def __init__(self, directory: Path, brush_size: int):
        self.dir = directory
        self.pares = sorted(self.dir.glob("*_rgb.jpg"))
        if not self.pares:
            raise SystemExit(f"[corregir] no encontre *_rgb.jpg en {directory}")
        self.i = 0
        self.brush = brush_size
        self.drawing = False
        self.erase_mode = False
        self.poligono: list[tuple[int, int]] = []
        self.poligono_activo = False
        self._cargar_actual()

    def _mask_path(self, rgb_path: Path) -> Path:
        return rgb_path.parent / rgb_path.name.replace("_rgb.jpg", "_mask.png")

    def _cargar_actual(self) -> None:
        rgb_path = self.pares[self.i]
        self.rgb = cv2.imread(str(rgb_path))
        if self.rgb is None:
            raise SystemExit(f"[corregir] no pude abrir {rgb_path}")
        h, w = self.rgb.shape[:2]

        mask_path = self._mask_path(rgb_path)
        m = None
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape[:2] != (h, w):
                print(f"[corregir] {mask_path.name} no coincide en tamaño con el rgb, empiezo en blanco")
                m = None
        if m is None:
            m = np.zeros((h, w), dtype=np.uint8)

        self.mask = (m > 127).astype(np.uint8) * 255
        self.mask_original = self.mask.copy()
        self.undo_stack: list[np.ndarray] = []
        self.modificado = False
        self.poligono = []
        self.poligono_activo = False

    # ------------------------------------------------------------- edicion

    def _snapshot(self) -> None:
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def _pintar(self, x: int, y: int, valor: int) -> None:
        cv2.circle(self.mask, (x, y), self.brush, int(valor), -1)
        self.modificado = True

    def _cerrar_poligono(self, valor: int) -> None:
        if len(self.poligono) >= 3:
            self._snapshot()
            pts = np.array([self.poligono], dtype=np.int32)
            cv2.fillPoly(self.mask, pts, int(valor))
            self.modificado = True
        self.poligono_activo = False
        self.poligono = []

    def _guardar(self) -> None:
        mask_bin = (self.mask > 127).astype(np.uint8) * 255  # estrictamente binaria
        path = self._mask_path(self.pares[self.i])
        cv2.imwrite(str(path), mask_bin)
        self.modificado = False
        print(f"[corregir] guardado {path.name}")

    # --------------------------------------------------------------- mouse

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if self.poligono_activo:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.poligono.append((x, y))
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing, self.erase_mode = True, False
            self._snapshot()
            self._pintar(x, y, 255)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing, self.erase_mode = True, True
            self._snapshot()
            self._pintar(x, y, 0)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._pintar(x, y, 0 if self.erase_mode else 255)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = False

    # -------------------------------------------------------------- render

    def _render(self):
        verde = np.zeros_like(self.rgb)
        verde[:, :, 1] = self.mask
        overlay = cv2.addWeighted(self.rgb, 1.0, verde, 0.35, 0)

        if self.poligono_activo and self.poligono:
            pts = np.array(self.poligono, dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(overlay, [pts], False, (0, 255, 255), 2)
            for p in self.poligono:
                cv2.circle(overlay, p, 3, (0, 255, 255), -1)

        estado = "*modificado*" if self.modificado else ""
        txt = f"{self.i + 1}/{len(self.pares)} {self.pares[self.i].name}  pincel={self.brush}  {estado}"
        cv2.putText(overlay, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return overlay

    # ----------------------------------------------------------------- loop

    def run(self) -> None:
        win = "2b_corregir_mascaras (ver docstring para atajos)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)
        print(__doc__.split("Atajos de teclado:")[1])

        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            k = chr(key) if key < 128 else ""

            if key == 27 or k == "q":
                if self.modificado:
                    self._guardar()
                break
            elif k == "s":
                self._guardar()
            elif k in ("+", "="):
                self.brush = min(200, self.brush + 4)
            elif k == "-":
                self.brush = max(2, self.brush - 4)
            elif k == "z":
                if self.undo_stack:
                    self.mask = self.undo_stack.pop()
                    self.modificado = True
            elif k == "r":
                self.mask = self.mask_original.copy()
                self.modificado = True
            elif k == "f":
                if self.poligono_activo:
                    self._cerrar_poligono(255)
                else:
                    self.poligono_activo = True
                    self.poligono = []
            elif k == "h" and self.poligono_activo:
                self._cerrar_poligono(0)
            elif k == "c":
                self.poligono_activo = False
                self.poligono = []
            elif k == "n":
                if self.modificado:
                    self._guardar()
                if self.i < len(self.pares) - 1:
                    self.i += 1
                    self._cargar_actual()
            elif k == "p":
                if self.modificado:
                    self._guardar()
                if self.i > 0:
                    self.i -= 1
                    self._cargar_actual()

        cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="carpeta con *_rgb.jpg y *_mask.png (ej. data/candidatos)")
    ap.add_argument("--brush-size", type=int, default=12)
    args = ap.parse_args()

    directory = Path(args.dir)
    if not directory.exists():
        print(f"[corregir] {directory} no existe")
        return 1

    Editor(directory, args.brush_size).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
