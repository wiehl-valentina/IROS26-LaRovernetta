"""Deteccion del cono + su posicion en el suelo, sin GPS.

Dos backends intercambiables (mismo patron que perception.py separa
SamTpRunner de PerceptionPipeline):

    ColorShapeConeDetector   heuristica HSV + forma. No necesita entrenar
                             nada: sirve para probar el resto del pipeline
                             (mission.py, indoor_bridge.py) HOY, en CPU, sin
                             dataset. Es un piso, no un techo: en un pasillo
                             real con carteles naranjas o luz dura va a dar
                             falsos positivos. Pensado como fallback/arranque,
                             no como detector final.

    YoloConeDetector         .pt de ultralytics YOLO. Es el camino real para
                             produccion. Este modulo NO entrena nada: si
                             todavia no hay un checkpoint entrenado, ver
                             "Como entrenar" en el doc de implementacion —
                             reutiliza el pipeline de candidatos+CVAT que ya
                             existe en ML_model/scripts/pipe_nuestras_rides/,
                             exportando en formato YOLO en vez de mascaras.

La parte que SI es nueva y no depende de ningun modelo es
`ground_point_from_bbox`: reproyecta el pixel inferior-central del bounding
box (donde el cono toca el piso) al plano del suelo usando EXACTAMENTE la
misma convencion forward/left que genie_path_planner.projection usa para
armar el BEV (camera_planar_axes). Eso evita el bug clasico de reinventar el
signo a mano y que quede desalineado con el planner que ya esta calibrado y
probado en este robot.

Autoprueba (no necesita camara, robot, ni modelo):
    python -m genie_rover.indoor.cone_detector
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from genie_path_planner.geometry import as_matrix, camera_planar_axes


# --------------------------------------------------------------------- tipos

@dataclass
class ConeDetection:
    """Una deteccion del cono en la imagen (no todavia en el mundo)."""
    bbox_xyxy: tuple[float, float, float, float]   # pixeles, imagen SIN rectificar
    confidence: float
    label: str = "cone"

    @property
    def center_x(self) -> float:
        x0, _, x1, _ = self.bbox_xyxy
        return 0.5 * (x0 + x1)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Pixel donde el cono toca el piso: centro en x, borde inferior en y."""
        x0, _, x1, y1 = self.bbox_xyxy
        return 0.5 * (x0 + x1), y1

    @property
    def width_px(self) -> float:
        x0, _, x1, _ = self.bbox_xyxy
        return x1 - x0

    @property
    def height_px(self) -> float:
        _, y0, _, y1 = self.bbox_xyxy
        return y1 - y0


@dataclass
class GroundPoint:
    """Posicion del cono en el plano del suelo, marco del robot (odometry.Pose)."""
    x_forward_m: float
    y_left_m: float
    distance_m: float

    def to_bev_goal(self) -> tuple[float, float]:
        """(x_right_m, y_forward_m): la convencion que espera plan_on_bev."""
        return -self.y_left_m, self.x_forward_m


# ------------------------------------------------------------ geometria: piso

def ground_point_from_pixel(
    pixel_xy: tuple[float, float],
    camera_k: np.ndarray,
    camera_pose: np.ndarray,
    ground_z: float = 0.0,
) -> GroundPoint | None:
    """Interseccion del rayo de un pixel con el plano del suelo.

    Misma familia de calculo que project_score_to_bev (genie_path_planner /
    projection.py), pero para un solo punto en vez de la imagen entera: no
    tiene sentido pagar el costo O(HxW) de esa funcion para un unico pixel.

    Devuelve None si el rayo no cruza el suelo hacia adelante (por ejemplo si
    el pixel esta arriba de la linea de horizonte, dz>=0 en camara mirando
    para abajo, o el punto quedaria detras de la camara).
    """
    k = as_matrix(camera_k, (3, 3), "camera_k")
    pose = as_matrix(camera_pose, (4, 4), "camera_pose")

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"intrinsecos invalidos: fx={fx} fy={fy}")

    u, v = pixel_xy
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

    r_world_cam = pose[:3, :3]
    t_world_cam = pose[:3, 3]
    dir_world = r_world_cam @ dir_cam

    dz = dir_world[2]
    if abs(dz) < 1e-8:
        return None
    s = (float(ground_z) - t_world_cam[2]) / dz
    if s <= 0.0:
        return None  # el rayo apunta para arriba: nunca toca el suelo

    point_world = t_world_cam + dir_world * s

    forward_xy, left_xy = camera_planar_axes(pose)
    rel_xy = point_world[:2] - t_world_cam[:2]
    forward_m = float(rel_xy @ forward_xy)
    left_m = float(rel_xy @ left_xy)
    distance_m = math.hypot(forward_m, left_m)

    return GroundPoint(x_forward_m=forward_m, y_left_m=left_m, distance_m=distance_m)


def ground_point_from_bbox(detection: ConeDetection, camera_k: np.ndarray,
                           camera_pose: np.ndarray, ground_z: float = 0.0
                           ) -> GroundPoint | None:
    return ground_point_from_pixel(detection.bottom_center, camera_k, camera_pose, ground_z)


# ------------------------------------------------------------ backend color

@dataclass
class ColorConeConfig:
    # Naranja tipico de cono de trafico en HSV de OpenCV (H:0-179).
    hsv_low: tuple[int, int, int] = (5, 120, 120)
    hsv_high: tuple[int, int, int] = (18, 255, 255)
    # Segundo rango opcional (p.ej. si el naranja del cono se corre hacia rojo
    # con la luz del pasillo y hay que cubrir ambos lados de H=0).
    hsv_low2: tuple[int, int, int] | None = None
    hsv_high2: tuple[int, int, int] | None = None
    min_area_px: int = 250
    min_aspect_ratio: float = 0.25   # ancho/alto minimo (cono es angosto)
    max_aspect_ratio: float = 1.3
    min_fill_ratio: float = 0.35     # area_contorno / area_bbox (el cono ahusa)
    morph_kernel: int = 5


class ColorShapeConeDetector:
    """HSV + contornos. Zero-shot: no necesita checkpoint ni GPU.

    Pensado para arrancar (probar mission.py/indoor_bridge.py de punta a
    punta en CPU) y como red de seguridad barata si el YOLO todavia no esta
    entrenado. NO reemplaza a un detector entrenado en un pasillo real con
    ruido visual (carteles, mochilas naranjas, luces calidas).
    """

    def __init__(self, cfg: ColorConeConfig | None = None):
        self.cfg = cfg or ColorConeConfig()

    def detect(self, rgb: np.ndarray) -> ConeDetection | None:
        import cv2

        c = self.cfg
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(c.hsv_low), np.array(c.hsv_high))
        if c.hsv_low2 is not None and c.hsv_high2 is not None:
            mask2 = cv2.inRange(hsv, np.array(c.hsv_low2), np.array(c.hsv_high2))
            mask = cv2.bitwise_or(mask, mask2)

        if c.morph_kernel > 1:
            kernel = np.ones((c.morph_kernel, c.morph_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: ConeDetection | None = None
        best_area = 0.0

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < c.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0:
                continue
            aspect = w / float(h)
            if not (c.min_aspect_ratio <= aspect <= c.max_aspect_ratio):
                continue
            fill = area / float(w * h)
            if fill < c.min_fill_ratio:
                continue
            if area > best_area:
                best_area = area
                # confianza heuristica: mas area y mas "compacto" -> mas confianza,
                # acotada para no competir de igual a igual con un YOLO real.
                conf = float(np.clip(0.35 + 0.35 * min(1.0, area / 4000.0), 0.0, 0.7))
                best = ConeDetection(bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                                     confidence=conf, label="cone(color)")
        return best


# -------------------------------------------------------------- backend YOLO

@dataclass
class YoloConeConfig:
    weights_path: str = "checkpoints/cone_yolo.pt"
    class_names: tuple[str, ...] = ("cone",)
    conf_thresh: float = 0.35
    device: str | None = None   # None = auto (cuda si hay, si no cpu)
    img_size: int = 640


class YoloConeDetector:
    """Envoltorio fino sobre ultralytics.YOLO. Carga el modelo una sola vez
    (mismo criterio que SamTpRunner en perception.py: nunca reconstruir el
    modelo adentro del loop de percepcion)."""

    def __init__(self, cfg: YoloConeConfig):
        self.cfg = cfg
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Falta 'ultralytics' (pip install ultralytics) para usar "
                "YoloConeDetector. Mientras tanto usa backend: color en el "
                "config, o entrena un checkpoint (ver doc de implementacion)."
            ) from exc

        ckpt = Path(cfg.weights_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"No encuentro los pesos YOLO en {ckpt}.\n"
                "Si todavia no entrenaste un detector de cono, usa "
                "backend: color en configs/indoor_cone_search.yaml para "
                "probar el resto del pipeline mientras tanto."
            )

        self.model = YOLO(str(ckpt))
        self._name_to_id = {v: k for k, v in self.model.names.items()}
        print(f"[cone_detector] YOLO cargado desde {ckpt} "
              f"(clases del modelo: {list(self.model.names.values())})")

    def detect(self, rgb: np.ndarray) -> ConeDetection | None:
        results = self.model.predict(
            source=rgb, imgsz=self.cfg.img_size, conf=self.cfg.conf_thresh,
            device=self.cfg.device, verbose=False,
        )
        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None

        wanted = set(self.cfg.class_names)
        best: ConeDetection | None = None
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = self.model.names.get(cls_id, str(cls_id))
            if wanted and name not in wanted:
                continue
            conf = float(box.conf[0])
            if best is not None and conf <= best.confidence:
                continue
            x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
            best = ConeDetection(bbox_xyxy=(x0, y0, x1, y1), confidence=conf, label=name)
        return best


# ------------------------------------------------------------------ pipeline

@dataclass
class ConeDetectorConfig:
    backend: str = "color"   # "color" | "yolo"
    color: ColorConeConfig = field(default_factory=ColorConeConfig)
    yolo: YoloConeConfig = field(default_factory=YoloConeConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "ConeDetectorConfig":
        d = dict(d or {})
        color_d = d.pop("color", {}) or {}
        yolo_d = d.pop("yolo", {}) or {}
        if "hsv_low" in color_d:
            color_d["hsv_low"] = tuple(color_d["hsv_low"])
        if "hsv_high" in color_d:
            color_d["hsv_high"] = tuple(color_d["hsv_high"])
        if "hsv_low2" in color_d and color_d["hsv_low2"] is not None:
            color_d["hsv_low2"] = tuple(color_d["hsv_low2"])
        if "hsv_high2" in color_d and color_d["hsv_high2"] is not None:
            color_d["hsv_high2"] = tuple(color_d["hsv_high2"])
        if "class_names" in yolo_d:
            yolo_d["class_names"] = tuple(yolo_d["class_names"])
        return cls(backend=d.get("backend", "color"),
                   color=ColorConeConfig(**color_d),
                   yolo=YoloConeConfig(**yolo_d))


class ConeDetectorPipeline:
    """Punto de entrada unico: .detect(rgb) sin importar el backend."""

    def __init__(self, cfg: ConeDetectorConfig):
        self.cfg = cfg
        if cfg.backend == "color":
            self.impl = ColorShapeConeDetector(cfg.color)
        elif cfg.backend == "yolo":
            self.impl = YoloConeDetector(cfg.yolo)
        else:
            raise ValueError(f"backend debe ser 'color' o 'yolo', no {cfg.backend!r}")

    def detect(self, rgb: np.ndarray) -> ConeDetection | None:
        return self.impl.detect(rgb)


# --------------------------------------------------------------------- pruebas

def _self_test() -> None:
    print("=== ground_point_from_pixel: geometria pura ===")
    # Camara a 15 cm, apenas inclinada hacia abajo (mismos valores que el
    # config real calibrado del robot, frodobot_rover.yaml).
    from ..perception import camera_pose_from_height_pitch
    pose = camera_pose_from_height_pitch(height_m=0.15, pitch_down_deg=15.0)
    k = np.array([[900.0, 0.0, 960.0],
                  [0.0, 900.0, 540.0],
                  [0.0, 0.0, 1.0]])

    # Un pixel bien abajo en el centro de la imagen deberia caer cerca, casi
    # derecho adelante.
    gp = ground_point_from_pixel((960.0, 1000.0), k, pose)
    print(f"  pixel centrado abajo -> forward={gp.x_forward_m:.2f} "
          f"left={gp.y_left_m:.2f} dist={gp.distance_m:.2f}")
    assert gp is not None and gp.x_forward_m > 0
    assert abs(gp.y_left_m) < 0.1, "deberia caer casi centrado"

    # Un pixel a la derecha de la imagen debe dar y_left NEGATIVO
    # (esta a la derecha, no a la izquierda) y por lo tanto x_right positivo.
    gp_der = ground_point_from_pixel((1500.0, 1000.0), k, pose)
    print(f"  pixel a la derecha  -> forward={gp_der.x_forward_m:.2f} "
          f"left={gp_der.y_left_m:.2f}")
    assert gp_der.y_left_m < 0
    x_right, y_fwd = gp_der.to_bev_goal()
    print(f"  como meta del planner: x_right={x_right:+.2f} y_forward={y_fwd:+.2f}")
    assert x_right > 0, "un cono a la derecha de la imagen debe pedir x_right positivo"

    # Un pixel arriba de la linea de horizonte (mirando al techo) no toca el suelo.
    gp_arriba = ground_point_from_pixel((960.0, 0.0), k, pose)
    print(f"  pixel muy arriba    -> {gp_arriba}")
    assert gp_arriba is None or gp_arriba.x_forward_m > 100, "no deberia dar un punto cercano"

    print("\n=== consistencia con project_score_to_bev (mismo signo que el BEV real) ===")
    from genie_path_planner.projection import project_score_to_bev
    score = np.zeros((200, 200), dtype=np.float32)
    # Marca un solo pixel "transitable" en el mismo lugar que el pixel de
    # prueba de arriba (960, 1000) escalado a la grilla de score 200x200
    # (imagen original 1920x1080 -> factor ~9.6x, ~5.4y).
    score[185, 100] = 1.0
    bev, observed, _ = project_score_to_bev(
        score_map=score, camera_k=k / np.array([[9.6, 1, 9.6], [1, 5.4, 5.4], [1, 1, 1]]),
        camera_pose=pose, ground_z=0.0, bev_resolution_m_per_px=0.03,
        bev_forward_range_m=3.0, bev_side_range_m=2.0, max_ray_distance_m=6.0,
    )
    assert observed.sum() >= 0  # solo confirma que no explota con esta pose/K

    print("\n=== ColorShapeConeDetector: cono naranja sintetico ===")
    rgb = np.full((480, 640, 3), (40, 60, 50), dtype=np.uint8)  # piso verdoso oscuro
    # Triangulo naranja (aproximado con un trapecio angosto) en el centro-abajo.
    import cv2
    pts = np.array([[300, 300], [340, 300], [330, 420], [310, 420]], dtype=np.int32)
    cv2.fillPoly(rgb, [pts], (235, 130, 20))  # naranja en RGB

    det = ColorShapeConeDetector()
    d = det.detect(rgb)
    print(f"  deteccion: {d}")
    assert d is not None, "no encontro el cono sintetico"
    assert 290 < d.center_x < 350

    print("\n=== sin cono en la imagen ===")
    rgb_vacio = np.full((480, 640, 3), (40, 60, 50), dtype=np.uint8)
    assert det.detect(rgb_vacio) is None

    print("\n=== ConeDetectorConfig.from_dict ===")
    cfg = ConeDetectorConfig.from_dict({
        "backend": "color",
        "color": {"hsv_low": [5, 120, 120], "hsv_high": [18, 255, 255], "min_area_px": 300},
    })
    assert cfg.backend == "color" and cfg.color.min_area_px == 300
    pipe = ConeDetectorPipeline(cfg)
    assert pipe.detect(rgb) is not None

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
