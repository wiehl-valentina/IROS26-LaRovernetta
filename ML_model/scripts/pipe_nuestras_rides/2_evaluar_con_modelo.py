
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raiz de ML_model
import config_paths  # noqa: E402  (agrega genie/ al sys.path si esta en .env -- ya no necesitamos traversability/)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm esta en requirements.txt, pero
    # no queremos que el script se rompa si alguien lo corre sin instalar.
    def tqdm(it, **kwargs):
        return it


def _puntaje_incertidumbre(mask: np.ndarray) -> float:
    """0 = el modelo esta seguro en todos lados (todo cerca de 0 o 1).
    1 = el modelo esta confundido en todos lados (todo cerca de 0.5).
    Frames con puntaje alto son buenos candidatos a etiquetar: son
    justamente donde el modelo no sabe que decir."""
    incertidumbre_por_pixel = 1.0 - 2.0 * np.abs(mask - 0.5)  # pico en mask=0.5
    return float(incertidumbre_por_pixel.mean())


def _guardar_mascara_binaria(mask: np.ndarray, path: Path) -> None:
    """Guarda `mask` (float, 0..1) como PNG en escala de grises ESTRICTAMENTE
    binario: 255 = transitable, 0 = no transitable. Este es el formato que
    3_armar_dataset.py / el loader de SAM2 esperan -- no tocar sin revisar
    ese docstring."""
    mask_bin = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask_bin, mode="L").save(path)


def _overlay_verde_rojo(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay verde (transitable) / rojo (no transitable) sobre el rgb
    original, mezclado con alpha -- mismo criterio de color que usaba el
    overlay de `rover_traversability.demo` (que ya no llamamos), asi las
    overlays viejas y las nuevas se siguen leyendo igual a ojo."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.float32)
    m = np.clip(mask, 0.0, 1.0)
    color[..., 1] = m * 255.0          # verde ~ transitable
    color[..., 0] = (1.0 - m) * 255.0  # rojo ~ no transitable
    out = rgb.astype(np.float32) * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolver_ruta(valor: str, base_dir: Path, nav_repo_root: Path | None) -> Path:
    """Resuelve una ruta de config que puede venir absoluta, relativa al yaml,
    o relativa a la raiz del repo de navegacion -- el mismo esquema que usa
    `genie_path_planner.pipeline.load_samtp_model` (y por lo tanto bridge.py):
    primero prueba la raiz del repo, y si no existe ahi, relativa al
    directorio del propio config."""
    p = Path(valor).expanduser()
    if p.is_absolute():
        return p
    if nav_repo_root is not None:
        candidata = nav_repo_root / p
        if candidata.exists():
            return candidata.resolve()
    return (base_dir / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True, help="carpeta con *_rgb.jpg")
    ap.add_argument("--out", required=True, help="jsonl de salida con el puntaje por frame")
    ap.add_argument("--overlays-dir", default=None,
                    help="si se pasa, guarda ahi el overlay verde/rojo de cada frame")
    ap.add_argument("--bridge-config", required=True,
                    help="ruta al yaml que usa bridge.py (ej: genie/configs/frodobot_rover.yaml). "
                         "De ahi se leen samtp.checkpoint_path, samtp.config_path, samtp.device, "
                         "samtp.precision y samtp.score_thresh -- se evalua SIEMPRE con lo mismo "
                         "que hoy corre en el rover, salvo que pises algo con --checkpoint/"
                         "--samtp-config/--device.")
    ap.add_argument("--checkpoint", default=None,
                    help="pisa samtp.checkpoint_path del yaml (para probar un checkpoint nuevo "
                         "sin tocarlo, por ejemplo para comparar overlays viejo vs. nuevo)")
    ap.add_argument("--samtp-config", default=None,
                    help="pisa samtp.config_path del yaml")
    ap.add_argument("--device", default=None,
                    help="pisa samtp.device del yaml (cuda/cpu; por default null = autodetectar)")
    ap.add_argument("--guardar-mascaras", action="store_true",
                    help="pre-anotacion automatica: guarda <id>_mask.png binarizada "
                         "junto a cada _rgb.jpg en --frames-dir")
    ap.add_argument("--forzar-mascaras", action="store_true",
                    help="con --guardar-mascaras, sobrescribe <id>_mask.png aunque ya "
                         "exista (por default no se pisa una mascara ya corregida a mano)")
    args = ap.parse_args()

    try:
        from genie_rover.perception import SamTpRunner  # noqa: F401  (solo para validar el import temprano)
    except ImportError as exc:
        print(f"[evaluar] no pude importar genie_rover.perception ({exc}).")
        print("Revisá que NAV_REPO_PATH apunte al repo de navegacion (el que tiene genie/ adentro, "
              "con genie_rover/ y sam2/) -- ver config_paths.py / .env. Ya no hace falta instalar "
              "traversability/ para este script.")
        return 1

    frames_dir = Path(args.frames_dir)
    frames = sorted(frames_dir.glob("*_rgb.jpg"))
    if not frames:
        print(f"[evaluar] no encontre *_rgb.jpg en {frames_dir}")
        return 1

    cfg_path = Path(args.bridge_config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"[evaluar] --bridge-config {cfg_path} no existe.")
        return 1

    import yaml
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    samtp_cfg = cfg.get("samtp", {})
    if not isinstance(samtp_cfg, dict):
        samtp_cfg = {}

    env_nav_repo = os.environ.get("NAV_REPO_PATH")
    if env_nav_repo:
        nav_repo_root = Path(env_nav_repo).expanduser().resolve()
    else:
        # frodobot_rover.yaml vive tipicamente en <repo>/genie/configs/, asi
        # que dos niveles arriba del yaml es la raiz del repo de navegacion
        # -- solo como fallback si NAV_REPO_PATH no esta seteada.
        nav_repo_root = cfg_path.parents[2] if len(cfg_path.parents) >= 3 else None

    checkpoint_valor = args.checkpoint or samtp_cfg.get("checkpoint_path")
    samtp_config_valor = args.samtp_config or samtp_cfg.get("config_path")
    if not checkpoint_valor or not samtp_config_valor:
        print(f"[evaluar] falta samtp.checkpoint_path o samtp.config_path -- ni en {cfg_path} "
              "ni por --checkpoint/--samtp-config.")
        return 1

    checkpoint_path = _resolver_ruta(checkpoint_valor, cfg_path.parent, nav_repo_root)
    samtp_config_path = _resolver_ruta(samtp_config_valor, cfg_path.parent, nav_repo_root)

    for etiqueta, p in (("checkpoint", checkpoint_path), ("config de SAM-TP", samtp_config_path)):
        if not p.is_file():
            print(f"[evaluar] no encuentro el {etiqueta} en {p} -- revisá NAV_REPO_PATH o la ruta "
                  "declarada en el yaml / --checkpoint / --samtp-config.")
            return 1

    device = args.device or samtp_cfg.get("device")
    precision = samtp_cfg.get("precision", "auto")
    score_thresh = float(samtp_cfg.get("score_thresh", 0.0))

    print("[evaluar] cargando SAM-TP (el primer frame puede tardar)...")
    origen = "--checkpoint explícito" if args.checkpoint else f"samtp.checkpoint_path de {cfg_path.name}"
    print(f"[evaluar] checkpoint: {checkpoint_path} ({origen})")
    print(f"[evaluar] config SAM-TP: {samtp_config_path}")
    print(f"[evaluar] device={device or 'autodetectar'} precision={precision} "
          f"score_thresh={score_thresh}")

    runner = SamTpRunner(
        config_path=str(samtp_config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
        score_thresh=score_thresh,
        precision=precision,
    )

    if args.overlays_dir:
        Path(args.overlays_dir).mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar de qué checkpoint/config vino esta corrida al lado del jsonl --
    # asi "qué modelo era este run" siempre se puede responder (mismo
    # criterio que "Shippear el checkpoint" en el README: registrar el
    # sha256 junto a los resultados).
    info = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "samtp_config_path": str(samtp_config_path),
        "bridge_config": str(cfg_path),
        "device": device,
        "precision": precision,
    }
    info_path = out_path.with_name(out_path.stem + "_model_info.json")
    info_path.write_text(json.dumps(info, indent=2))
    print(f"[evaluar] info del modelo -> {info_path}")

    mascaras_generadas = 0
    mascaras_saltadas = 0

    with open(out_path, "w") as f:
        for frame_path in tqdm(frames, desc="[evaluar] evaluando frames", unit="frame"):
            rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)

            t0 = time.perf_counter()
            mask = runner.traversability(rgb)  # HxW float32 en [0,1], 1 = transitable
            inference_s = time.perf_counter() - t0

            score = _puntaje_incertidumbre(mask)
            row = {
                "frame_file": frame_path.name,
                "incertidumbre": round(score, 4),
                "drivable_frac": round(float(mask.mean()), 4),
                "inference_s": round(inference_s, 3),
            }
            f.write(json.dumps(row) + "\n")

            if args.overlays_dir:
                overlay = _overlay_verde_rojo(rgb, mask)
                Image.fromarray(overlay).save(
                    Path(args.overlays_dir) / frame_path.name.replace("_rgb.jpg", "_overlay.png"))

            if args.guardar_mascaras:
                mask_path = frame_path.parent / frame_path.name.replace("_rgb.jpg", "_mask.png")
                if mask_path.exists() and not args.forzar_mascaras:
                    mascaras_saltadas += 1
                else:
                    _guardar_mascara_binaria(mask, mask_path)
                    mascaras_generadas += 1

    print(f"\n[evaluar] listo -> {out_path}")
    print("Ordená por 'incertidumbre' descendente para saber que etiquetar/corregir primero.")
    if args.guardar_mascaras:
        print(f"[evaluar] {mascaras_generadas} mascaras pre-generadas "
              f"({mascaras_saltadas} ya existian y no se tocaron -- usá --forzar-mascaras "
              "para pisarlas)")
        print("Siguiente paso: corregí los bordes donde el modelo se equivocó con "
              "2b_corregir_mascaras.py (o CVAT/Label Studio) y despues 3_armar_dataset.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
