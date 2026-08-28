"""Percepcion: SAM-TP -> mapa de transitabilidad -> BEV.

Diferencias respecto de sam2.sam_tp.SAM_TP:
  * detecta el device (SAM_TP hardcodea "cuda" via el default de build_sam2)
  * construye el SAM2ImagePredictor una sola vez, no en cada frame
  * corrige la distorsion del lente antes de inferir, porque
    project_score_to_bev asume un modelo pinhole puro

Prueba standalone (no necesita robot, solo el checkpoint y una imagen):
    python -m genie_rover.perception --config configs/frodobot_rover.yaml \
        --image alguna_foto.jpg --out debug/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genie_path_planner.projection import logits_to_traversability, project_score_to_bev


# ------------------------------------------------------------------ pose camara

def camera_pose_from_height_pitch(height_m: float, pitch_down_deg: float) -> np.ndarray:
    """Construye T_world_camera para una camara fija mirando adelante.

    Mundo: x adelante, y izquierda, z arriba, origen en el suelo bajo la camara.
    Camara (optical frame, que es lo que asume projection.py):
        +x = derecha de la imagen, +y = abajo de la imagen, +z = hacia adelante.

    pitch_down_deg es cuanto esta inclinada la camara HACIA ABAJO respecto de
    la horizontal (positivo = mira al piso). En un rover de vereda suele estar
    entre 10 y 30 grados.
    """
    th = math.radians(float(pitch_down_deg))
    z_cam = np.array([math.cos(th), 0.0, -math.sin(th)])   # adelante y abajo
    x_cam = np.array([0.0, -1.0, 0.0])                     # derecha de imagen = -y mundo
    y_cam = np.cross(z_cam, x_cam)                         # abajo de imagen

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_cam
    pose[:3, 1] = y_cam
    pose[:3, 2] = z_cam
    pose[:3, 3] = np.array([0.0, 0.0, float(height_m)])
    return pose


# ------------------------------------------------------------------- distorsion

class Undistorter:
    """Rectifica el frame para que valga el modelo pinhole de projection.py."""

    def __init__(self, camera_k: np.ndarray, dist_coeffs: np.ndarray | None,
                 image_size: tuple[int, int]):
        import cv2

        self.cv2 = cv2
        self.k_orig = np.asarray(camera_k, dtype=np.float64).reshape(3, 3)
        self.enabled = dist_coeffs is not None and np.any(np.abs(np.asarray(dist_coeffs)) > 1e-9)
        w, h = int(image_size[0]), int(image_size[1])

        if not self.enabled:
            self.k = self.k_orig
            self.map1 = self.map2 = None
            return

        self.dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        # alpha=0 recorta los bordes invalidos: preferimos perder FOV antes que
        # proyectar pixeles basura al BEV.
        self.k, _ = cv2.getOptimalNewCameraMatrix(self.k_orig, self.dist, (w, h), 0, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.k_orig, self.dist, None, self.k, (w, h), cv2.CV_16SC2
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        return self.cv2.remap(image, self.map1, self.map2, self.cv2.INTER_LINEAR)


# ----------------------------------------------------------------- SAM-TP

class SamTpRunner:
    def __init__(self, config_path: str, checkpoint_path: str,
                 device: str | None = None, score_thresh: float = 0.0,
                 precision: str = "auto"):
        import torch

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.autocast_dtype = None

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Se pidio device=cuda pero torch no ve ninguna GPU. "
                                   "Revisa el driver con nvidia-smi y la build de torch.")
            props = torch.cuda.get_device_properties(0)
            cc = f"{props.major}.{props.minor}"

            # bfloat16 recien existe en hardware desde Ampere (SM 8.0). En Turing
            # (RTX 20xx, SM 7.5) el autocast bf16 que usa el codigo de SAM2 se
            # emula y termina siendo mas lento que fp32. fp16 si tiene tensor
            # cores en Turing y es la eleccion correcta.
            if precision == "auto":
                self.autocast_dtype = torch.bfloat16 if props.major >= 8 else torch.float16
            elif precision == "bf16":
                self.autocast_dtype = torch.bfloat16
            elif precision == "fp16":
                self.autocast_dtype = torch.float16
            elif precision == "fp32":
                self.autocast_dtype = None
            else:
                raise ValueError(f"precision debe ser auto/bf16/fp16/fp32, no {precision!r}")

            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            # La entrada de SAM2 siempre se reescala a 1024x1024, asi que la
            # forma es constante y a cudnn le conviene buscar los mejores kernels.
            torch.backends.cudnn.benchmark = True

            dtype_name = "fp32" if self.autocast_dtype is None else str(self.autocast_dtype).split(".")[-1]
            print(f"[samtp] GPU: {props.name} (SM {cc}, {props.total_memory / 2**30:.1f} GB) "
                  f"-> precision {dtype_name}")
            if props.major < 8 and precision == "bf16":
                print("[samtp] AVISO: forzaste bf16 en una GPU pre-Ampere. Va a andar "
                      "mucho mas lento que fp16.")

        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"No encuentro el checkpoint de SAM-TP en {ckpt}.\n"
                "Bajalo del Google Drive del repo y ponelo en esa ruta exacta "
                "(ojo que el directorio se llama literalmente '...yaml')."
            )

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print(f"[samtp] cargando modelo en {device} ...")
        self.model = build_sam2(config_path, str(ckpt), device=device)
        self.predictor = SAM2ImagePredictor(sam_model=self.model, mask_threshold=float(score_thresh))
        print("[samtp] listo")

        if device == "cpu":
            print("[samtp] AVISO: en CPU la inferencia tarda ~1-4 s por frame a 1024px. "
                  "Bajale la velocidad al rover en consecuencia.")

    def traversability(self, rgb: np.ndarray) -> np.ndarray:
        """Devuelve un mapa HxW en [0,1]: 1 = transitable."""
        h, w = rgb.shape[:2]
        # Los 3 puntos del fondo de la imagen son vestigiales: el
        # CustomPromptEncoderLarger ignora los prompts y usa su token aprendido.
        # Los mantenemos porque la API del predictor los exige.
        pts = np.array([[0, h - 1], [w - 1, h - 1], [(w - 1) // 2, h - 1]], dtype=np.float32)
        labels = np.ones(len(pts), dtype=np.int32)

        with self.torch.inference_mode():
            ctx = (self.torch.autocast("cuda", dtype=self.autocast_dtype)
                   if self.autocast_dtype is not None else _nullcontext())
            with ctx:
                self.predictor.reset_predictor()
                self.predictor.set_image(rgb)
                masks, _iou, _low = self.predictor.predict(
                    point_coords=pts,
                    point_labels=labels,
                    multimask_output=False,
                    return_logits=True,
                    normalize_coords=False,
                )

        logits = np.asarray(masks[0], dtype=np.float32)
        return logits_to_traversability(logits, transform="sigmoid")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------------- pipeline

@dataclass
class BevResult:
    traversability: np.ndarray   # HxW en [0,1], -1 = desconocido
    observed: np.ndarray         # HxW uint8
    image_traversability: np.ndarray
    stats: dict


class PerceptionPipeline:
    def __init__(self, cfg: dict):
        cam = cfg["camera"]
        proj = cfg["projection"]

        self.camera_k = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
        self.dist = cam.get("dist_coeffs")
        self.image_size = tuple(cam["image_size"])  # (w, h) de la calibracion
        self._rectifiers: dict[tuple[int, int], Undistorter] = {}
        self._warned_sizes: set[tuple[int, int]] = set()

        if "pose" in cam and cam["pose"] is not None:
            self.camera_pose = np.asarray(cam["pose"], dtype=np.float64).reshape(4, 4)
        else:
            self.camera_pose = camera_pose_from_height_pitch(
                cam["height_m"], cam["pitch_down_deg"]
            )

        self.ground_z = float(proj.get("ground_z", 0.0))
        self.resolution = float(proj["resolution_m_per_px"])
        self.forward_range = float(proj["forward_range_m"])
        self.side_range = float(proj["side_range_m"])
        self.max_ray = float(proj.get("max_ray_distance_m", 6.0))
        # project_score_to_bev recorre cada pixel del mapa de scores, asi que
        # su costo es cuadratico en la resolucion. Con celdas de 3 cm, 1920x1080
        # es mucho mas de lo que el BEV puede aprovechar: medido, submuestrear
        # a la mitad baja la proyeccion de 263 ms a 68 ms perdiendo solo un 9%
        # de celdas observadas.
        self.projection_downscale = max(1, int(proj.get("projection_downscale", 2)))

        s = cfg["samtp"]
        self.runner = SamTpRunner(
            config_path=s["config_path"],
            checkpoint_path=s["checkpoint_path"],
            device=s.get("device"),
            score_thresh=float(s.get("score_thresh", 0.0)),
            precision=s.get("precision", "auto"),
        )

    def _rectifier_for(self, size: tuple[int, int]) -> Undistorter:
        """Devuelve (y cachea) el rectificador para una resolucion dada.

        El stream del Earth Rover Mini es H.264 adaptativo: puede llegar a
        1280x720, 1024x576 o 480x270 segun el ancho de banda, y puede cambiar
        en medio de un recorrido. Los intrinsecos son proporcionales a la
        resolucion, asi que en vez de fallar los reescalamos. Los coeficientes
        de distorsion no se tocan: viven en coordenadas normalizadas.
        """
        if size in self._rectifiers:
            return self._rectifiers[size]

        w0, h0 = self.image_size
        w, h = size
        sx, sy = w / float(w0), h / float(h0)

        k = self.camera_k.copy()
        k[0, 0] *= sx; k[0, 2] *= sx
        k[1, 1] *= sy; k[1, 2] *= sy

        if size != self.image_size and size not in self._warned_sizes:
            self._warned_sizes.add(size)
            ar0, ar = w0 / float(h0), w / float(h)
            print(f"[percep] frame {w}x{h} != calibracion {w0}x{h0}: reescalo K "
                  f"(sx={sx:.3f}, sy={sy:.3f})")
            if abs(ar - ar0) > 0.01:
                print(f"[percep] ADVERTENCIA: cambio la relacion de aspecto "
                      f"({ar0:.3f} -> {ar:.3f}). El stream esta recortando, no "
                      f"solo reescalando, y el reescalado de K no alcanza.")

        rect = Undistorter(k, self.dist, size)
        self._rectifiers[size] = rect
        return rect

    def process(self, rgb: np.ndarray) -> BevResult:
        h, w = rgb.shape[:2]
        rect = self._rectifier_for((w, h))

        rectified = rect(rgb)
        trav_img = self.runner.traversability(rectified)

        score, k = trav_img, rect.k
        d = self.projection_downscale
        if d > 1:
            import cv2
            h2, w2 = trav_img.shape[0] // d, trav_img.shape[1] // d
            score = cv2.resize(trav_img, (w2, h2), interpolation=cv2.INTER_AREA)
            k = k.copy()
            k[0, 0] /= d; k[0, 2] /= d
            k[1, 1] /= d; k[1, 2] /= d

        bev, observed, stats = project_score_to_bev(
            score_map=score,
            camera_k=k,
            camera_pose=self.camera_pose,
            ground_z=self.ground_z,
            bev_resolution_m_per_px=self.resolution,
            bev_forward_range_m=self.forward_range,
            bev_side_range_m=self.side_range,
            max_ray_distance_m=self.max_ray,
        )
        return BevResult(bev, observed, trav_img, stats)


# --------------------------------------------------------------------- self test

def _self_test(config_path: str, image_path: str, out_dir: str) -> None:
    import yaml
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = yaml.safe_load(Path(config_path).read_text())
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

    pipe = PerceptionPipeline(cfg)
    res = pipe.process(rgb)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB")
    axes[1].imshow(res.image_traversability, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("SAM-TP (rojo = transitable)")
    vis = np.where(res.traversability < 0, np.nan, res.traversability)
    axes[2].imshow(vis, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    axes[2].set_title("BEV (arriba = adelante)")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "perception_debug.png", dpi=110)
    print(f"Escrito {out / 'perception_debug.png'}")

    # Throughput real. Importa para elegir max_linear: si el modelo tarda 200 ms
    # y el rover va a 0.3 m/s, avanza 6 cm entre decisiones. Si tarda 3 s, 90 cm.
    import time as _time
    n = 10
    pipe.process(rgb)  # descartar la primera, incluye la compilacion de kernels
    t0 = _time.perf_counter()
    for _ in range(n):
        pipe.process(rgb)
    dt = (_time.perf_counter() - t0) / n
    print(f"\nInferencia + proyeccion: {dt * 1000:.0f} ms por frame ({1 / dt:.1f} Hz)")
    print(f"Celdas observadas en el BEV: {res.stats['bev_observed_cells']:.0f} "
          f"de {res.traversability.size}")
    print("\nQue mirar: el suelo transitable delante del robot tiene que aparecer "
          "verde en el BEV, con forma de abanico que se abre hacia adelante. Si "
          "sale torcido o comprimido, revisa height_m y pitch_down_deg.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="debug")
    a = ap.parse_args()
    _self_test(a.config, a.image, a.out)
