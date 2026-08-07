from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import as_matrix, camera_planar_axes, reference_frame_from_pose


@dataclass
class BEVObservation:
    """
    Contenedor de datos que representa una única observación (un 'fotograma' o barrido de sensores)
    proyectada al espacio de vista aérea (Bird's Eye View - BEV).
    
    Guarda versiones independientes de la transitabilidad calculada según la cámara RGB, el sensor 
    de Profundidad (Depth), y una fusión de ambos (RGB-D), manteniendo un registro de la postura
    exacta de la cámara y el chasis en el instante en que se capturó.
    """
    name: str
    camera_pose: np.ndarray
    bev_resolution_m: float
    rgb_bev: np.ndarray
    rgb_observed: np.ndarray
    depth_bev: np.ndarray | None = None
    depth_observed: np.ndarray | None = None
    rgbd_bev: np.ndarray | None = None
    rgbd_observed: np.ndarray | None = None
    robot_pose: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def logits_to_traversability(logits: np.ndarray, transform: str = "sigmoid") -> np.ndarray:
    """
    Convierte la salida cruda de una Red Neuronal (Logits) en probabilidades matemáticas puras [0.0, 1.0].

    CONCEPTOS CLAVE:
    - Logits: Son números sin límite (ej. -25.5, +120.3) que produce la última capa de una IA como SAM.
    - Transformación Sigmoide (sigmoid): Es la función matemática clásica `1 / (1 + e^-x)`. 
      "Aplasta" el rango infinito de los logits hacia el rango `[0.0, 1.0]`. Esto significa que un 
      logit de valor muy negativo (obstáculo) se convierte casi en 0.0, y uno muy positivo (asfalto) 
      se convierte casi en 1.0. 
      Nota: Se usa np.clip() entre -40 y 40 para evitar errores de desbordamiento (Overflow) al calcular `e^-x`.
    """
    logits_f = np.asarray(logits, dtype=np.float32)
    if transform == "sigmoid":
        x = np.clip(logits_f, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    if transform == "minmax":
        finite = np.isfinite(logits_f)
        out = np.zeros_like(logits_f, dtype=np.float32)
        if np.any(finite):
            lo = float(np.min(logits_f[finite]))
            hi = float(np.max(logits_f[finite]))
            out[finite] = ((logits_f[finite] - lo) / max(1e-8, hi - lo)).astype(np.float32)
        return out
    if transform in {"none", "identity"}:
        return logits_f.astype(np.float32)
    raise ValueError(f"Unknown score transform: {transform}")


def project_score_to_bev(
    score_map: np.ndarray,
    camera_k: np.ndarray,
    camera_pose: np.ndarray,
    ground_z: float,
    bev_resolution_m_per_px: float,
    bev_forward_range_m: float,
    bev_side_range_m: float,
    max_ray_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Proyecta un mapa de transitabilidad visto en perspectiva (como si fuera una foto) al plano 
    del suelo (vista desde arriba / Bird's Eye View), usando Raycasting Inverso.

    CONCEPTOS CLAVE:
    - Raycasting (Lanzamiento de Rayos): Por cada píxel en la imagen 2D (score_map), la función 
      traza una línea imaginaria (un rayo) que sale desde la lente de la cámara (cx, cy) hacia adelante,
      atraviesa el píxel y sigue viajando hasta chocar contra la altura `ground_z` (el suelo del mundo 3D).
    - Modelo de Pin-hole Camera (Cámara Estenopeica): Usa la matriz de calibración intrínseca `camera_k` 
      (focal fx, fy y centros cx, cy) para calcular el ángulo físico exacto con el que cada rayo 
      de luz entró a la lente.
    - Mapeo de Densidad: Como los rayos proyectados lejos en el piso se separan (divergen), a veces
      varios píxeles de la cámara apuntan a la misma celda de la cuadrícula en el piso. El algoritmo
      usa `np.bincount` para acumular todos los valores y promediarlos eficientemente por cada celda.

    Args:
        score_map: Matriz 2D proveniente de la IA indicando si el píxel de la foto es asfalto u obstáculo.
        camera_k: Matriz intrínseca 3x3 de la cámara.
        camera_pose: Postura global (matriz 4x4) de la cámara en el mundo real.
        ground_z: Altura (en eje Z) donde se considera que está el nivel del piso físico.
        bev_resolution_m_per_px, bev_forward_range_m, bev_side_range_m: Dimensiones y escala del mapa final.
        max_ray_distance_m: Distancia límite para proyectar. Los píxeles del cielo o el horizonte lejano 
                            se descartan porque la precisión trigonométrica cae drásticamente.

    Returns:
        bev_flat: La matriz cuadrada de tránsito vista desde el aire.
        observed: Máscara que indica qué cuadros del mapa aéreo fueron tocados por algún rayo.
        metadata: Estadísticas del proceso (píxeles de entrada, rayos impactados, etc).
    """
    if score_map.ndim != 2:
        raise ValueError("score_map must be HxW")
    camera_k = as_matrix(camera_k, (3, 3), "camera_K")
    camera_pose = as_matrix(camera_pose, (4, 4), "camera_pose")
    if bev_resolution_m_per_px <= 0:
        raise ValueError("bev_resolution_m_per_px must be > 0")
    if bev_forward_range_m <= 0:
        raise ValueError("bev_forward_range_m must be > 0")
    if bev_side_range_m <= 0:
        raise ValueError("bev_side_range_m must be > 0")

    bev_h = max(1, int(np.ceil(float(bev_forward_range_m) / float(bev_resolution_m_per_px))))
    bev_w = max(1, int(np.ceil((2.0 * float(bev_side_range_m)) / float(bev_resolution_m_per_px))))

    # np.indices genera dos matrices (Ys y Xs) indicando la posición 2D de cada píxel de la imagen
    ys, xs = np.indices(score_map.shape, dtype=np.float64)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    scores = np.asarray(score_map, dtype=np.float32).reshape(-1)
    finite = np.isfinite(scores)
    xs = xs[finite]
    ys = ys[finite]
    scores = scores[finite]
    if xs.size == 0:
        bev = np.full((bev_h, bev_w), -1.0, dtype=np.float32)
        observed = np.zeros((bev_h, bev_w), dtype=np.uint8)
        return bev, observed, {
            "input_pixels": float(score_map.size),
            "valid_score_pixels": 0.0,
            "projected_ground_points": 0.0,
            "bev_observed_cells": 0.0,
        }

    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])
    cx = float(camera_k[0, 2])
    cy = float(camera_k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid camera intrinsics fx/fy: fx={fx}, fy={fy}")

    # Paso 1: Convertir píxeles 2D en vectores de dirección 3D (Rayos) respecto a la lente.
    dirs_cam = np.empty((xs.size, 3), dtype=np.float64)
    dirs_cam[:, 0] = (xs - cx) / fx
    dirs_cam[:, 1] = (ys - cy) / fy
    dirs_cam[:, 2] = 1.0

    # Paso 2: Rotar los rayos desde la perspectiva de la cámara al sistema de coordenadas del mundo global.
    r_world_cam = camera_pose[:3, :3]
    t_world_cam = camera_pose[:3, 3]
    dirs_world = dirs_cam @ r_world_cam.T

    # Paso 3: Calcular matemáticamente en qué factor hay que alargar cada rayo (ray_scale) 
    # para que su componente Z intercepte exactamente la altura del piso (ground_z).
    dz = dirs_world[:, 2]
    valid = np.abs(dz) > 1e-8
    ray_scale = np.zeros_like(dz, dtype=np.float64)
    ray_scale[valid] = (float(ground_z) - float(t_world_cam[2])) / dz[valid]
    
    # Descartar rayos que apuntan al cielo (ray_scale negativo no choca nunca contra el piso)
    valid &= ray_scale > 0.0

    # Multiplicar los vectores dirección por la escala calculada para obtener el impacto en 3D
    points_world = t_world_cam[None, :] + dirs_world * ray_scale[:, None]
    
    if max_ray_distance_m > 0.0:
        dist_xy = np.linalg.norm(points_world[:, :2] - t_world_cam[None, :2], axis=1)
        valid &= dist_xy <= float(max_ray_distance_m)

    forward_xy, left_xy = camera_planar_axes(camera_pose)
    rel_xy = points_world[:, :2] - t_world_cam[None, :2]
    forward_m = rel_xy @ forward_xy
    left_m = rel_xy @ left_xy

    # Filtrar solo impactos que caigan dentro del tamaño del mapa físico configurado (bounding box)
    valid &= forward_m >= 0.0
    valid &= forward_m < float(bev_forward_range_m)
    valid &= np.abs(left_m) < float(bev_side_range_m)
    if not np.any(valid):
        bev = np.full((bev_h, bev_w), -1.0, dtype=np.float32)
        observed = np.zeros((bev_h, bev_w), dtype=np.uint8)
        return bev, observed, {
            "input_pixels": float(score_map.size),
            "valid_score_pixels": float(xs.size),
            "projected_ground_points": 0.0,
            "bev_observed_cells": 0.0,
        }

    forward_valid = forward_m[valid]
    left_valid = left_m[valid]
    score_valid = scores[valid]
    
    # Conversión geométrica: transformar los metros del impacto (forward/left) en índices de la grilla 2D (rows/cols)
    rows = bev_h - 1 - np.floor(forward_valid / float(bev_resolution_m_per_px)).astype(np.int32)
    cols = (bev_w // 2) - np.floor(left_valid / float(bev_resolution_m_per_px)).astype(np.int32)
    in_bounds = (rows >= 0) & (rows < bev_h) & (cols >= 0) & (cols < bev_w)
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    score_valid = score_valid[in_bounds]

    flat_size = bev_h * bev_w
    flat_idx = rows.astype(np.int64) * bev_w + cols.astype(np.int64)
    # Acumulación: bincount suma todos los scores que cayeron en la misma celda de la grilla
    sum_flat = np.bincount(flat_idx, weights=score_valid.astype(np.float64), minlength=flat_size)
    cnt_flat = np.bincount(flat_idx, minlength=flat_size)
    observed_flat = cnt_flat > 0

    bev_flat = np.full(flat_size, -1.0, dtype=np.float32)
    # Promediar el puntaje de la celda (suma total / cantidad de rayos que impactaron en esa celda)
    bev_flat[observed_flat] = (sum_flat[observed_flat] / cnt_flat[observed_flat]).astype(np.float32)
    observed = observed_flat.reshape(bev_h, bev_w).astype(np.uint8)
    
    return bev_flat.reshape(bev_h, bev_w), observed, {
        "input_pixels": float(score_map.size),
        "valid_score_pixels": float(xs.size),
        "projected_ground_points": float(np.count_nonzero(valid)),
        "bev_observed_cells": float(np.count_nonzero(observed_flat)),
    }


def depth_to_bev_height_and_traversability(
    depth_m: np.ndarray,
    camera_k: np.ndarray,
    camera_pose: np.ndarray,
    ground_z: float,
    reliable_depth_m: float,
    min_depth_m: float,
    obstacle_height_thresh_m: float,
    bev_resolution_m_per_px: float,
    bev_forward_range_m: float,
    bev_side_range_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Proyecta un mapa de profundidad 3D (nube de puntos) al plano superior y convierte la altura física 
    de los obstáculos en un puntaje de "Transitabilidad" para el planificador.

    CONCEPTOS CLAVE:
    - Conversión Geométrica (Point Cloud): A diferencia de la cámara RGB (donde suponíamos que el rayo 
      chocaba contra el piso plano), aquí el sensor YA NOS DICE a qué distancia (profundidad z) chocó. 
      La función reconstruye la coordenada 3D exacta de cada punto en el mundo real.
    - Cálculo de Altura: Resta la coordenada Z del punto reconstruido contra la altura teórica del piso (`ground_z`).
      El resultado es la altura física en metros de los elementos (ej. la vereda = 0.15m, una pared = 2.0m).
    - Mapa de Elevación (Max Height): Si varios puntos de profundidad caen en la misma celda de la cuadrícula,
      utiliza `np.maximum.at` para quedarse con el punto MÁS ALTO de esa celda (ej. prefiero registrar 
      la copa de un árbol a 3m que su raíz, para forzar al planificador a rodearlo).
    - Regla de transitabilidad por altura:
        * Altura 0.0m = Tránsito 1.0 (Camino libre).
        * Altura > umbral (`obstacle_height_thresh_m`, ej. 0.20m = paragolpes) = Tránsito 0.0 (Choque seguro).
        * Altura intermedia: Tránsito proporcional.
    """
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be HxW")
    camera_k = as_matrix(camera_k, (3, 3), "camera_K")
    camera_pose = as_matrix(camera_pose, (4, 4), "camera_pose")
    if reliable_depth_m <= 0.0:
        raise ValueError("reliable_depth_m must be > 0")
    if obstacle_height_thresh_m <= 0.0:
        raise ValueError("obstacle_height_thresh_m must be > 0")
    if bev_resolution_m_per_px <= 0.0:
        raise ValueError("bev_resolution_m_per_px must be > 0")

    bev_h = max(1, int(np.ceil(float(bev_forward_range_m) / float(bev_resolution_m_per_px))))
    bev_w = max(1, int(np.ceil((2.0 * float(bev_side_range_m)) / float(bev_resolution_m_per_px))))

    h, w = depth_m.shape
    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])
    cx = float(camera_k[0, 2])
    cy = float(camera_k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid camera intrinsics fx/fy: fx={fx}, fy={fy}")

    ys, xs = np.indices((h, w), dtype=np.float64)
    z = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(z)
    valid &= z >= float(min_depth_m)
    # Filtra los puntos ruidosos (ej. el láser apuntando al cielo que rebota al infinito)
    valid &= z <= float(reliable_depth_m)

    if not np.any(valid):
        height_map = np.full((bev_h, bev_w), np.nan, dtype=np.float32)
        trav_map = np.full((bev_h, bev_w), -1.0, dtype=np.float32)
        observed_mask = np.zeros((bev_h, bev_w), dtype=bool)
        return height_map, trav_map, observed_mask, {
            "input_pixels": float(h * w),
            "valid_depth_pixels": 0.0,
            "bev_observed_cells": 0.0,
            "bev_non_traversable_cells": 0.0,
        }

    xs_v = xs[valid]
    ys_v = ys[valid]
    z_v = z[valid]
    
    # Reconstrucción 3D (Deshacer la proyección de perspectiva de la lente de la cámara)
    x_cam = (xs_v - cx) * z_v / fx
    y_cam = (ys_v - cy) * z_v / fy
    points_cam = np.stack([x_cam, y_cam, z_v], axis=1)

    # Multiplica los puntos reconstruidos por la postura global de la cámara
    points_world = points_cam @ camera_pose[:3, :3].T + camera_pose[:3, 3][None, :]
    
    # Calcula la altura física restando el nivel del piso. np.maximum elimina ruidos negativos (agujeros).
    height_m = np.maximum(points_world[:, 2] - float(ground_z), 0.0)

    forward_xy, left_xy = camera_planar_axes(camera_pose)
    rel_xy = points_world[:, :2] - camera_pose[:2, 3][None, :]
    forward_m = rel_xy @ forward_xy
    left_m = rel_xy @ left_xy
    valid_bev = forward_m >= 0.0
    valid_bev &= forward_m < float(bev_forward_range_m)
    valid_bev &= np.abs(left_m) < float(bev_side_range_m)

    if not np.any(valid_bev):
        height_map = np.full((bev_h, bev_w), np.nan, dtype=np.float32)
        trav_map = np.full((bev_h, bev_w), -1.0, dtype=np.float32)
        observed_mask = np.zeros((bev_h, bev_w), dtype=bool)
        return height_map, trav_map, observed_mask, {
            "input_pixels": float(h * w),
            "valid_depth_pixels": float(z_v.size),
            "bev_observed_cells": 0.0,
            "bev_non_traversable_cells": 0.0,
        }

    forward_v = forward_m[valid_bev]
    left_v = left_m[valid_bev]
    height_v = height_m[valid_bev]
    rows = bev_h - 1 - np.floor(forward_v / float(bev_resolution_m_per_px)).astype(np.int32)
    cols = (bev_w // 2) - np.floor(left_v / float(bev_resolution_m_per_px)).astype(np.int32)
    in_bounds = (rows >= 0) & (rows < bev_h) & (cols >= 0) & (cols < bev_w)
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    height_v = height_v[in_bounds]

    flat_size = bev_h * bev_w
    flat_idx = rows.astype(np.int64) * bev_w + cols.astype(np.int64)
    max_height_flat = np.full(flat_size, -np.inf, dtype=np.float32)
    
    # Esta operación acumula la máxima elevación topográfica posible por cada celda de la matriz.
    np.maximum.at(max_height_flat, flat_idx, height_v.astype(np.float32))

    observed_flat = max_height_flat > -np.inf
    height_map = max_height_flat.reshape(bev_h, bev_w)
    observed = observed_flat.reshape(bev_h, bev_w)
    height_map[~observed] = np.nan

    trav_map = np.full((bev_h, bev_w), -1.0, dtype=np.float32)
    h_obs = height_map[observed]
    h_clip = np.clip(h_obs, 0.0, float(obstacle_height_thresh_m))
    
    # Conversión lineal inversa: altura máxima(0.2m) da 0.0 de transitabilidad; altura suelo(0.0m) da 1.0.
    score = 1.0 - (h_clip / float(obstacle_height_thresh_m))
    score = np.clip(score, 0.0, 1.0)
    score[h_obs >= float(obstacle_height_thresh_m)] = 0.0
    trav_map[observed] = score.astype(np.float32)

    return height_map, trav_map, observed, {
        "input_pixels": float(h * w),
        "valid_depth_pixels": float(z_v.size),
        "bev_observed_cells": float(np.count_nonzero(observed)),
        "bev_non_traversable_cells": float(np.count_nonzero(observed & (trav_map <= 1e-6))),
    }


def blend_modalities(
    rgb_bev: np.ndarray,
    rgb_observed: np.ndarray,
    depth_bev: np.ndarray,
    depth_observed: np.ndarray,
    rgb_weight: float,
    depth_weight: float,
    require_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fusiona (mezcla) el mapa generado por Inteligencia Artificial (Cámara RGB) con el mapa geométrico
    puro generado por el láser de Profundidad (Depth), creando un único mapa de obstáculos robusto.

    CONCEPTOS CLAVE:
    - Redundancia (Sensor Fusion): La IA RGB es excelente identificando qué tipo de terreno es 
      (ej. pasto vs asfalto), pero es mala calculando distancias exactas. El Depth sensor calcula distancias
      milimétricas pero no diferencia entre asfalto gris y hielo gris. Esta función unifica lo mejor 
      de ambos mundos aplicando promedios ponderados (`rgb_weight` y `depth_weight`).
    """
    if rgb_bev.shape != depth_bev.shape:
        raise ValueError(f"rgb_bev and depth_bev shape mismatch: {rgb_bev.shape} vs {depth_bev.shape}")
    rgb_known = rgb_observed.astype(bool) & np.isfinite(rgb_bev) & (rgb_bev >= 0.0)
    depth_known = depth_observed.astype(bool) & np.isfinite(depth_bev) & (depth_bev >= 0.0)

    out = np.full(rgb_bev.shape, -1.0, dtype=np.float32)
    only_rgb = rgb_known & (~depth_known)
    only_depth = depth_known & (~rgb_known)
    both = rgb_known & depth_known
    
    if not require_depth:
        out[only_rgb] = rgb_bev[only_rgb].astype(np.float32)
    out[only_depth] = depth_bev[only_depth].astype(np.float32)
    
    denom = float(rgb_weight) + float(depth_weight)
    if np.any(both):
        if denom <= 1e-8:
            out[both] = (0.5 * (rgb_bev[both] + depth_bev[both])).astype(np.float32)
        else:
            # Promedio ponderado para píxeles donde AMBOS sensores concuerdan que hay visión
            out[both] = (
                (float(rgb_weight) * rgb_bev[both] + float(depth_weight) * depth_bev[both]) / denom
            ).astype(np.float32)
            
    observed = (np.isfinite(out) & (out >= 0.0)).astype(np.uint8)
    return out, observed


def select_observation_map(record: BEVObservation, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Rutina selectora para extraer del contenedor de datos la modalidad de visualización específica solicitada.
    """
    mode_l = str(mode).lower()
    if mode_l == "rgb":
        return record.rgb_bev, record.rgb_observed
    if mode_l == "depth":
        if record.depth_bev is None or record.depth_observed is None:
            raise ValueError(f"Observation {record.name!r} has no depth BEV map")
        return record.depth_bev, record.depth_observed
    if mode_l == "rgbd":
        if record.rgbd_bev is None or record.rgbd_observed is None:
            raise ValueError(f"Observation {record.name!r} has no RGB-D BEV map")
        return record.rgbd_bev, record.rgbd_observed
    raise ValueError(f"Unsupported BEV mode: {mode}")


def observation_world_points_from_map(
    camera_pose: np.ndarray,
    bev_score: np.ndarray,
    bev_observed: np.ndarray,
    bev_resolution_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae (desproyecta) los píxeles de un mapa BEV local y los convierte en coordenadas
    absolutas del mundo (World X, Y). Necesario antes de intentar pegar o fusionar 
    mapas capturados en momentos distintos de tiempo.
    """
    observed = bev_observed.astype(bool) & np.isfinite(bev_score) & (bev_score >= 0.0)
    rows, cols = np.nonzero(observed)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float32)

    bev_h, bev_w = bev_score.shape
    res = float(bev_resolution_m)
    # Convierte índices fila/columna de vuelta a metros (físicos) corrigiendo el offset por la resolución
    forward = ((bev_h - 1 - rows).astype(np.float64) + 0.5) * res
    left = ((bev_w // 2 - cols).astype(np.float64) + 0.5) * res

    camera_pose = as_matrix(camera_pose, (4, 4), "camera_pose")
    cam_xy = camera_pose[:2, 3].astype(np.float64)
    forward_xy, left_xy = camera_planar_axes(camera_pose)
    world_xy = cam_xy[None, :] + forward[:, None] * forward_xy[None, :] + left[:, None] * left_xy[None, :]
    return world_xy, bev_score[rows, cols].astype(np.float32)


def fuse_bev_observations(
    records: list[BEVObservation],
    mode: str,
    reference_pose: np.ndarray,
    reference_frame: str,
    bev_resolution_m: float,
    bev_forward_range_m: float,
    bev_side_range_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Ejecuta una Fusión Espacio-Temporal global. Toma múltiples mapas BEV generados en 
    el pasado, alinea todas sus coordenadas y los unifica en un súper-mapa centralizado y actualizado.

    CONCEPTOS CLAVE:
    - SLAM Simplificado: Como el robot avanza hacia adelante, la visión que tuvo hace 3 fotogramas 
      atrás ya no está enfrente suyo, sino al lado o detrás de sus ruedas. Esta función toma todos los mapas 
      históricos, los convierte a un sistema unificado global de coordenadas `world_xy` y los vuelve a 
      recortar (`rel`) en relación exclusiva a la posición EXACTA Y ACTUAL (`reference_pose`) del robot.
    - Superposición y Promedio: Las zonas del mapa que fueron vistas repetidamente en diferentes momentos 
      se apilan (`np.bincount(weights)`). El planificador promedia esos puntajes de historial repetido para 
      generar una certidumbre altísima sobre si una porción de terreno específica tiene o no un obstáculo.

    Returns:
        fused_flat: La matriz BEV gigante con la información consolidada de todos los instantes de tiempo.
    """
    if not records:
        raise ValueError("records must not be empty")
    res = float(bev_resolution_m)
    if res <= 0.0:
        raise ValueError("bev_resolution_m must be > 0")
    bev_h = max(1, int(np.ceil(float(bev_forward_range_m) / res)))
    bev_w = max(1, int(np.ceil((2.0 * float(bev_side_range_m)) / res)))
    ref_xy, ref_forward, ref_left = reference_frame_from_pose(reference_pose, reference_frame)

    all_world_xy: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    selected_names: list[str] = []
    
    # 1. Recuperar los datos de cada fotograma individual y convertirlos a coordenadas mundiales
    for rec in records:
        score_map, obs_map = select_observation_map(rec, mode)
        world_xy, scores = observation_world_points_from_map(
            camera_pose=rec.camera_pose,
            bev_score=score_map,
            bev_observed=obs_map,
            bev_resolution_m=rec.bev_resolution_m,
        )
        if world_xy.shape[0] == 0:
            continue
        all_world_xy.append(world_xy)
        all_scores.append(scores)
        selected_names.append(rec.name)

    if not all_world_xy:
        fused = np.full((bev_h, bev_w), -1.0, dtype=np.float32)
        observed = np.zeros((bev_h, bev_w), dtype=np.uint8)
        return fused, observed, {
            "mode": mode,
            "selected_observations": selected_names,
            "fusion_resolution_m": res,
            "reference_frame": reference_frame,
            "cell_count": 0,
        }

    # 2. Acoplar temporalmente todos los fotogramas en nubes de datos gigantes y continuas
    world_xy = np.concatenate(all_world_xy, axis=0)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    
    # 3. Traducir el mapa mundial global de vuelta a un mapa relativo al punto ciego actual del vehículo
    rel = world_xy - ref_xy[None, :]
    forward = rel @ ref_forward
    left = rel @ ref_left
    valid = forward >= 0.0
    valid &= forward < float(bev_forward_range_m)
    valid &= np.abs(left) < float(bev_side_range_m)
    if not np.any(valid):
        fused = np.full((bev_h, bev_w), -1.0, dtype=np.float32)
        observed = np.zeros((bev_h, bev_w), dtype=np.uint8)
        return fused, observed, {
            "mode": mode,
            "selected_observations": selected_names,
            "fusion_resolution_m": res,
            "reference_frame": reference_frame,
            "cell_count": 0,
        }

    forward_v = forward[valid]
    left_v = left[valid]
    scores_v = scores[valid]
    
    # Proyectar geométricamente los metros otra vez a coordenadas de imagen/grilla 2D
    rows = bev_h - 1 - np.floor(forward_v / res).astype(np.int32)
    cols = (bev_w // 2) - np.floor(left_v / res).astype(np.int32)
    in_bounds = (rows >= 0) & (rows < bev_h) & (cols >= 0) & (cols < bev_w)
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    scores_v = scores_v[in_bounds]

    flat_size = bev_h * bev_w
    flat_idx = rows.astype(np.int64) * bev_w + cols.astype(np.int64)
    # Acumular y sobre-escribir píxeles de fotogramas viejos usando bincount
    sum_flat = np.bincount(flat_idx, weights=scores_v.astype(np.float64), minlength=flat_size)
    cnt_flat = np.bincount(flat_idx, minlength=flat_size)
    observed_flat = cnt_flat > 0
    fused_flat = np.full(flat_size, -1.0, dtype=np.float32)
    fused_flat[observed_flat] = (sum_flat[observed_flat] / cnt_flat[observed_flat]).astype(np.float32)
    
    observed = observed_flat.reshape(bev_h, bev_w).astype(np.uint8)
    return fused_flat.reshape(bev_h, bev_w), observed, {
        "mode": mode,
        "selected_observations": selected_names,
        "fusion_resolution_m": res,
        "reference_frame": reference_frame,
        "cell_count": int(np.count_nonzero(observed_flat)),
        "input_projected_points": int(world_xy.shape[0]),
        "points_inside_reference_window": int(np.count_nonzero(valid)),
    }


def traversability_vis(score_map: np.ndarray, draw_robot_marker: bool = True) -> np.ndarray:
    """
    Función de renderizado gráfico por telemetría.
    
    Renderiza matricialmente la transitabilidad del entorno fusionado al espectro visible 
    humano con un código de semáforo simple (Verde alto/seguro, Rojo bajo/peligro, Negro incierto), 
    pintando un marcador blanco fijo en la posición central del chasis para referencia física.
    """
    h, w = score_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    known = np.isfinite(score_map) & (score_map >= 0.0)
    if np.any(known):
        s = np.clip(score_map[known], 0.0, 1.0)
        vis[known] = np.stack(
            [
                ((1.0 - s) * 255.0).astype(np.uint8),
                (s * 255.0).astype(np.uint8),
                np.zeros_like(s, dtype=np.uint8),
            ],
            axis=1,
        )
    if draw_robot_marker:
        center_col = w // 2
        center_row = h - 1
        vis[max(0, center_row - 2) : min(h, center_row + 3), max(0, center_col - 2) : min(w, center_col + 3)] = (
            255,
            255,
            255,
        )
        arrow_top = max(0, center_row - 28)
        vis[arrow_top : center_row + 1, max(0, center_col - 1) : min(w, center_col + 1)] = (255, 255, 255)
    return vis