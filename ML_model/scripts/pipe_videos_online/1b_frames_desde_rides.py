#!/usr/bin/env python3
"""Puente entre pipe_videos_online y pipe_nuestras_rides.

Toma los rides ya limpiados (clean/<ride>.synced.parquet) + los segmentos de
video en raw/<ride>/recordings/, elige frames espaciados por distancia GPS
real y los guarda como <ride>_<n>_rgb.jpg en la carpeta de candidatos --
exactamente el formato que espera 2_evaluar_con_modelo.py.

Formato real de recordings/ en este dataset (Mini-4K): NO hay un .mp4 unico
por camara. Cada camara es un stream HLS trozado en segmentos .ts de ~15-16
segundos, nombrados así:

    <hash>_ride_<id>__uid_s_<UID>__uid_e_video_<YYYYMMDDHHMMSSmmm>.ts

Puede haber mas de un UID en la misma carpeta (varias camaras/streams). El
timestamp en el nombre es un epoch UTC en milisegundos -- coincide exacto
con la columna `epoch_ms` del .synced.parquet. Este script usa esa
coincidencia para:
  1. Agrupar los segmentos por UID.
  2. Elegir automaticamente cual UID es la camara frontal: el que arranca
     mas cerca en el tiempo del primer frame del parquet (podes forzarlo
     con --uid-front si el automatico se equivoca).
  3. Para cada frame elegido, ubicar en que segmento cae su timestamp y
     abrir SOLO ese segmento con cv2, en vez de intentar reproducir todo
     el stream seguido.

Uso tipico (desde ML_model/):

    python scripts/pipe_videos_online/1b_frames_desde_rides.py \
        --clean-dir scripts/pipe_videos_online/clean_mx \
        --raw-dir   scripts/pipe_videos_online/raw_mx \
        --rides     scripts/pipe_videos_online/curated_mx.csv \
        --out       data/candidatos \
        --max-por-ride 40

Nota sobre --every-n-m: por default esta en 0, es decir DESACTIVADO. La
razon es que `clean` (--target-step-m) ya espacia los puntos por distancia
RECORRIDA sobre el camino (acumulando cada tramo, incluidas curvas). Si acá
se vuelve a filtrar por distancia, pero midiendo en línea recta entre el
ultimo elegido y el candidato, dos puntos separados por una curva pueden
quedar a pocos metros en linea recta aunque hayan sido 5 m de manejo real
-- y se descartan de nuevo, perdiendo la mitad o mas de los frames en rides
con muchas vueltas. Con --every-n-m 0 se usan tal cual los puntos que ya
vienen de `clean` (recortando solo a --max-por-ride). Solo tiene sentido
poner --every-n-m > 0 si además de lo que hace `clean` querés un espaciado
extra más grande (por ejemplo, `clean` a 5 m pero acá pedir 10 m).

Primero corrolo con --inspeccionar para ver que columnas trae tu parquet:

    python 1b_frames_desde_rides.py --clean-dir clean_mx --inspeccionar
"""

from __future__ import annotations

import argparse
import bisect
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

# ----------------------------------------------------------------------
# deteccion de columnas (los nombres varian segun version del dataset)
# ----------------------------------------------------------------------

CAND_LAT = ["lat", "latitude", "gps_lat", "gps_latitude"]
CAND_LON = ["lon", "lng", "longitude", "gps_lon", "gps_longitude"]
CAND_TS = ["front_ts", "front_timestamp", "frame_ts", "timestamp", "ts"]

SEG_RE = re.compile(r"uid_s_(\w+)__uid_e_(video|audio)_(\d{17})\.ts$")


def _pick(df: pd.DataFrame, candidatos: list[str], que: str) -> str:
    for c in candidatos:
        if c in df.columns:
            return c
    raise SystemExit(
        f"[frames] no encontre columna de {que}. Columnas disponibles:\n"
        f"  {list(df.columns)}\n"
        f"Pasa --col-{que} explicitamente."
    )


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _elegir_indices(df: pd.DataFrame, col_lat: str, col_lon: str,
                    every_n_m: float, max_por_ride: int) -> list[int]:
    """Devuelve indices de filas a usar, hasta max_por_ride.

    Si every_n_m <= 0 (default): no vuelve a filtrar por distancia -- toma
    los puntos tal cual vienen del parquet, que ya estan espaciados por
    distancia RECORRIDA (no en linea recta) desde `clean`. Volver a filtrar
    aca por distancia en linea recta descarta de mas en curvas (ver nota en
    el docstring del modulo).

    Si every_n_m > 0: aplica un espaciado ADICIONAL en linea recta, para
    quien quiera menos frames todavia que los que ya vienen de `clean`.
    """
    elegidos: list[int] = []
    ult_lat = ult_lon = None
    for i, row in enumerate(df.itertuples(index=False)):
        lat = getattr(row, col_lat)
        lon = getattr(row, col_lon)
        if pd.isna(lat) or pd.isna(lon):
            continue
        if every_n_m <= 0 or ult_lat is None or _haversine_m(ult_lat, ult_lon, lat, lon) >= every_n_m:
            elegidos.append(i)
            ult_lat, ult_lon = lat, lon
            if len(elegidos) >= max_por_ride:
                break
    return elegidos


# ----------------------------------------------------------------------
# segmentos HLS (.ts) por camara
# ----------------------------------------------------------------------

def _parse_seg_timestamp(ts_str: str) -> int:
    """'20250331131331184' (YYYYMMDDHHMMSSmmm, UTC) -> epoch en milisegundos."""
    y, mo, d = int(ts_str[0:4]), int(ts_str[4:6]), int(ts_str[6:8])
    h, mi, s = int(ts_str[8:10]), int(ts_str[10:12]), int(ts_str[12:14])
    ms = int(ts_str[14:17])
    dt = datetime(y, mo, d, h, mi, s, ms * 1000, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _listar_segmentos(raw_dir: Path, ride: str) -> dict[str, list[tuple[int, Path]]]:
    """Agrupa los segmentos *_video_*.ts de recordings/ por uid (cada uid es
    una camara/stream distinta), con el epoch ms de inicio de cada uno."""
    base = raw_dir / ride / "recordings"
    grupos: dict[str, list[tuple[int, Path]]] = {}
    if not base.exists():
        return grupos
    for p in base.glob("*.ts"):
        m = SEG_RE.search(p.name)
        if not m or m.group(2) != "video":
            continue
        uid, ts_str = m.group(1), m.group(3)
        try:
            start_ms = _parse_seg_timestamp(ts_str)
        except ValueError:
            continue
        grupos.setdefault(uid, []).append((start_ms, p))
    for uid in grupos:
        grupos[uid].sort(key=lambda t: t[0])
    return grupos


def _elegir_uid_frontal(grupos: dict[str, list[tuple[int, Path]]],
                        ref_epoch_ms: float, uid_forzado: str | None) -> str | None:
    if uid_forzado:
        return uid_forzado if uid_forzado in grupos else None
    if not grupos:
        return None
    # el uid cuyo primer segmento arranca mas cerca en el tiempo del primer
    # frame del parquet -- en la practica da match exacto (mismo epoch ms)
    return min(grupos, key=lambda uid: abs(grupos[uid][0][0] - ref_epoch_ms))


def procesar_ride(parquet: Path, raw_dir: Path, out_dir: Path, args) -> int:
    ride = parquet.name.replace(".synced.parquet", "")
    df = pd.read_parquet(parquet)
    if df.empty:
        print(f"[frames] {ride}: parquet vacio, salteo")
        return 0

    col_lat = args.col_lat or _pick(df, CAND_LAT, "lat")
    col_lon = args.col_lon or _pick(df, CAND_LON, "lon")

    if "epoch_ms" in df.columns:
        col_epoch_ms = "epoch_ms"
    else:
        col_ts = args.col_ts or _pick(df, CAND_TS, "ts")
        df = df.copy()
        df["_epoch_ms_derivado"] = (df[col_ts].astype(float) * 1000).round().astype("int64")
        col_epoch_ms = "_epoch_ms_derivado"

    grupos = _listar_segmentos(raw_dir, ride)
    if not grupos:
        print(f"[frames] {ride}: no encontre segmentos .ts en "
              f"{raw_dir / ride / 'recordings'} -- bajaste con --skip-recordings? salteo")
        return 0

    uid_frontal = _elegir_uid_frontal(grupos, float(df[col_epoch_ms].iloc[0]), args.uid_front)
    if uid_frontal is None:
        print(f"[frames] {ride}: no pude elegir camara frontal entre uids {list(grupos)} "
              f"-- pasa --uid-front a mano. salteo")
        return 0

    segmentos = grupos[uid_frontal]
    starts = [s for s, _ in segmentos]

    indices = _elegir_indices(df, col_lat, col_lon, args.every_n_m, args.max_por_ride)
    if not indices:
        print(f"[frames] {ride}: ningun frame con GPS valido, salteo")
        return 0

    slug = re.sub(r"[^A-Za-z0-9_-]", "_", ride)
    guardados = 0
    cap = None
    cap_path = None

    for n, idx in enumerate(indices):
        target_ms = int(df[col_epoch_ms].iloc[idx])
        pos = bisect.bisect_right(starts, target_ms) - 1
        if pos < 0:
            continue
        seg_start, seg_path = segmentos[pos]
        offset_ms = target_ms - seg_start

        if cap_path != seg_path:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(str(seg_path))
            cap_path = seg_path
            if not cap.isOpened():
                print(f"[frames] {ride}: no pude abrir segmento {seg_path.name}, salteo frame")
                cap, cap_path = None, None
                continue

        cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        destino = out_dir / f"{slug}_{n:04d}_rgb.jpg"
        cv2.imwrite(str(destino), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        guardados += 1

    if cap is not None:
        cap.release()

    print(f"[frames] {ride}: uid frontal={uid_frontal} ({len(segmentos)} segmentos), "
          f"{guardados} frames -> {out_dir}")
    return guardados


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-dir", required=True, help="carpeta con los *.synced.parquet")
    p.add_argument("--raw-dir", help="carpeta raw/ con los videos (recordings/)")
    p.add_argument("--rides", help="CSV de curate; si se pasa, solo se procesan esos rides")
    p.add_argument("--out", default="data/candidatos")
    p.add_argument("--every-n-m", type=float, default=0.0,
                   help="0 (default) = no volver a filtrar por distancia, usar los puntos "
                        "tal cual vienen de `clean` (recomendado). Un valor > 0 aplica un "
                        "espaciado ADICIONAL en linea recta, para achicar aun mas.")
    p.add_argument("--max-por-ride", type=int, default=40)
    p.add_argument("--col-lat"), p.add_argument("--col-lon"), p.add_argument("--col-ts")
    p.add_argument("--uid-front", default=None,
                   help="fuerza el uid de la camara frontal (por default se detecta solo "
                        "comparando el timestamp del primer segmento contra el primer frame "
                        "del parquet)")
    p.add_argument("--inspeccionar", action="store_true",
                   help="imprime las columnas del primer parquet y sale")
    args = p.parse_args()

    clean_dir = Path(args.clean_dir)
    parquets = sorted(clean_dir.glob("*.synced.parquet"))
    if not parquets:
        print(f"[frames] no hay *.synced.parquet en {clean_dir}")
        return 1

    if args.inspeccionar:
        df = pd.read_parquet(parquets[0])
        print(f"[frames] {parquets[0].name} — {len(df)} filas")
        print("columnas:", list(df.columns))
        print(df.head(3).to_string())
        return 0

    if not args.raw_dir:
        print("[frames] falta --raw-dir")
        return 1

    if args.rides:
        sel = pd.read_csv(args.rides)
        col = "ride_folder" if "ride_folder" in sel.columns else "folder"
        quiero = set(sel[col].astype(str))
        parquets = [q for q in parquets
                    if q.name.replace(".synced.parquet", "") in quiero]
        print(f"[frames] {len(parquets)} rides tras filtrar por {args.rides}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = sum(procesar_ride(q, Path(args.raw_dir), out_dir, args) for q in parquets)
    print(f"\n[frames] total: {total} frames en {out_dir}")
    print("Siguiente paso: 2_evaluar_con_modelo.py --guardar-mascaras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())