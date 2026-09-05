#!/usr/bin/env python3
"""
pipeline.py — limpieza y curación de BitRobot/FrodoBots-Mini-4K
para el Earth Rover Challenge (fase F2 del plan).[cite: 5]

Cuatro etapas, cada una un subcomando. Corren en orden:[cite: 5]

    python pipeline.py select   --out selected_rides.csv[cite: 5]
    python pipeline.py fetch    --rides selected_rides.csv --raw-dir raw[cite: 5]
    python pipeline.py clean    --raw-dir raw --clean-dir clean --index cleaned_index.csv[cite: 5]
    python pipeline.py curate   --index cleaned_index.csv --hours 300 --out curated_rides.csv[cite: 5]
"""
from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "BitRobot/FrodoBots-Mini-4K"
REPO_TYPE = "dataset"

# ----------------------------------------------------------------------
# utilidades compartidas
# ----------------------------------------------------------------------

def _hf_download(filename: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type=REPO_TYPE)


def load_metadata() -> pd.DataFrame:
    path = _hf_download("metadata.parquet")
    return pd.read_parquet(path)


_COUNTRY_UTC_OFFSET_H = {
    "China": 8,
    "Philippines": 8,
    "United States": -5,
    "Vietnam": 7,
    "India": 5.5,
    "Kenya": 3,
    "Mexico": -6,
    "Taiwan": 8,
    "New Zealand": 12,
    "Panama": -5,
    "Canada": -5,
    "Costa Rica": -6,
    "Malaysia": 8,
    "United Kingdom": 0,
    "Germany": 1,
    "Botswana": 2,
    "Australia": 10,
    "Brazil": -3,
    "Mauritius": 4,
    "Indonesia": 7,
    "Sweden": 1,
    "Turkey": 3,
    "Nigeria": 1,
    "Chile": -4,
    "Argentina": -3,
    "Estonia": 2,
    "Netherlands": 1,
    "Singapore": 8,
    "Japan": 9,
}


def is_night(start_utc: str, country: str) -> bool | None:
    offset = _COUNTRY_UTC_OFFSET_H.get(country)
    if offset is None:
        return None
    try:
        ts = pd.Timestamp(start_utc)
    except Exception:
        return None
    local_hour = (ts.hour + offset) % 24
    return local_hour >= 20 or local_hour < 6


def gps_cell(lat: float, lon: float, cell_size_m: float = 15.0) -> tuple[int, int] | None:
    if lat is None or lon is None:
        return None
    if abs(lat - 1000) < 1e-6 and abs(lon - 1000) < 1e-6:
        return None
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    if m_per_deg_lon <= 1e-6:
        return None
    return (
        round(lat * m_per_deg_lat / cell_size_m),
        round(lon * m_per_deg_lon / cell_size_m),
    )


# ----------------------------------------------------------------------
# 1) select — filtrar a nivel metadata, sin bajar tars
# ----------------------------------------------------------------------

def cmd_select(args: argparse.Namespace) -> None:
    meta = load_metadata()
    sel = meta.copy()

    required_always = ["folder", "shard", "db_dur_sec"]
    missing = [c for c in required_always if c not in sel.columns]
    if missing:
        raise SystemExit(
            f"[select] faltan columnas imprescindibles {missing} en metadata.parquet.\n"
        )

    def col(name: str, default=True):
        if name not in sel.columns:
            return pd.Series(default, index=sel.index)
        return sel[name]

    if args.require_rear:
        sel = sel[col("has_rear_camera") == True]  # noqa: E712

    sel = sel[
        (col("has_control") == True)  # noqa: E712
        & (col("has_gps") == True)  # noqa: E712
        & (col("has_imu") == True)  # noqa: E712
        & (col("has_front_ts") == True)  # noqa: E712
    ]

    has_country = "country" in sel.columns

    if args.countries and has_country:
        wanted = {c.strip() for c in args.countries.split(",")}
        sel = sel[sel["country"].isin(wanted)]

    if args.max_per_country and has_country:
        sel = sel.sort_values("db_dur_sec", ascending=False)
        rank = sel.groupby("country").cumcount()
        sel = sel[rank < args.max_per_country]

    sel = sel.sort_values("db_dur_sec", ascending=False)

    if has_country and "start_utc" in sel.columns:
        sel["is_night"] = [is_night(u, c) for u, c in zip(sel["start_utc"], sel["country"])]
    else:
        sel["is_night"] = None

    keep_cols = [c for c in ['folder', 'ride_id', 'device_ref_id', 'start_utc', 'source', 'db_dur_sec', 'n_epochs', 'hardware_version', 'drive_mode', 'country', 'city', 'front_footage_sec', 'rear_footage_sec', 'has_rear_camera', 'has_control', 'has_gps', 'has_imu', 'has_front_ts', 'has_rear_ts', 'has_mic_ts', 'has_speaker_ts', 'n_video_ts', 'has_video', 'complete', 'audio_only', 'shard', 'shard_member_bytes'] if c in sel.columns]
    sel = sel[keep_cols]

    sel.to_csv(args.out, index=False)
    total_h = sel["db_dur_sec"].sum() / 3600
    print(f"[select] {len(sel)} rides seleccionados, {total_h:.1f} h totales -> {args.out}")
    if has_country:
        print("[select] rides por país:")
        print(sel["country"].value_counts().to_string())


# ----------------------------------------------------------------------
# 2) fetch — bajar solo los shards necesarios, extraer solo esos rides
# ----------------------------------------------------------------------

def cmd_fetch(args: argparse.Namespace) -> None:
    sel = pd.read_csv(args.rides)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Identificar columna de carpeta de ride (puede llamarse 'folder' o 'ride_folder')
    folder_col = "ride_folder" if "ride_folder" in sel.columns else ("folder" if "folder" in sel.columns else None)
    if not folder_col:
        raise SystemExit(f"[fetch] Error: El archivo {args.rides} no contiene una columna de carpeta válida ('folder' o 'ride_folder').")

    # Si falta la columna 'shard' (ej. archivo curated), la recuperamos del metadata oficial
    if "shard" not in sel.columns:
        print("[fetch] 'shard' no encontrado en el archivo de entrada. Cruzando con metadata.parquet...")
        meta = load_metadata()[["folder", "shard"]]
        sel = sel.merge(meta, left_on=folder_col, right_on="folder", how="left")
        if "shard" not in sel.columns or sel["shard"].isna().any():
            raise SystemExit("[fetch] Error crítico: algunos rides no pudieron asociarse a ningún 'shard'.")

    n_shards = sel["shard"].nunique()
    print(f"[fetch] {len(sel)} rides en {n_shards} shards distintos")

    for i, (shard, group) in enumerate(sel.groupby("shard"), 1):
        print(f"[fetch] ({i}/{n_shards}) {shard} — {len(group)} ride(s)")
        shard_path = _hf_download(shard)
        wanted_prefixes = tuple(group[folder_col].astype(str))
        with tarfile.open(shard_path) as t:
            members = [m for m in t.getmembers() if m.name.startswith(wanted_prefixes)]
            if args.skip_recordings:
                members = [m for m in members if "/recordings/" not in m.name]
            t.extractall(raw_dir, members=members)

    print(f"[fetch] listo -> {raw_dir}")


# ----------------------------------------------------------------------
# 3) clean — arreglar unidades, tirar GPS sin fix, sincronizar por frame
# ----------------------------------------------------------------------

def _read_ride_csvs(ride_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for kind, glob_pat, ts_col, ts_unit in [
        ("control", "control_data_*.csv", "timestamp", "s"),
        ("gps", "gps_data_*.csv", "timestamp", "ms"),
        ("imu", "imu_data_*.csv", "timestamp", "ms"),
        ("front_ts", "front_camera_timestamps_*.csv", "timestamp", "s"),
        ("rear_ts", "rear_camera_timestamps_*.csv", "timestamp", "s"),
    ]:
        matches = list(ride_dir.glob(glob_pat))
        if not matches:
            continue
        df = pd.read_csv(matches[0])
        if ts_col in df.columns:
            factor = 1000 if ts_unit == "s" else 1
            df["epoch_ms"] = (df[ts_col] * factor).astype("int64")
        out[kind] = df
    return out


def clean_ride(ride_dir: Path, cell_size_m: float, target_step_m: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Sincroniza frame+GPS+control+IMU y re-muestrea espacialmente.

    target_step_m: cada cuantos metros de movimiento real se guarda una foto
    -- 5 m por default, que es la distancia a la que dos fotos del rover ya
    dejan de verse practicamente identicas. No confundir con cell_size_m
    (que es el tamaño de celda para el dedup de `curate`, otra cosa).
    """
    data = _read_ride_csvs(ride_dir)
    if "gps" not in data or "front_ts" not in data:
        raise ValueError(f"{ride_dir.name}: faltan gps o front_camera_timestamps")

    gps = data["gps"].copy()
    n_gps_raw = len(gps)
    is_no_fix = (gps["latitude"].round(3) == 1000) & (gps["longitude"].round(3) == 1000)
    gps = gps[~is_no_fix].sort_values("epoch_ms")
    frac_valid_gps = 1 - (is_no_fix.sum() / max(n_gps_raw, 1))

    frames = data["front_ts"].sort_values("epoch_ms")

    synced = pd.merge_asof(
        frames, gps[["epoch_ms", "latitude", "longitude"]],
        on="epoch_ms", direction="nearest", tolerance=500,
    )

    if "control" in data:
        ctl = data["control"].sort_values("epoch_ms")
        cols = [c for c in ["epoch_ms", "linear", "angular"] if c in ctl.columns]
        synced = pd.merge_asof(synced, ctl[cols], on="epoch_ms", direction="nearest", tolerance=500)

    synced = synced.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. RE-MUESTREO ESPACIAL (puntos distanciados target_step_m metros)
    # ------------------------------------------------------------------
    lat = synced["latitude"].to_numpy()
    lon = synced["longitude"].to_numpy()

    selected_indices = [0]
    accumulated_dist = 0.0
    total_dist_m = 0.0
    last_idx = 0

    for i in range(1, len(lat)):
        dlat = (lat[i] - lat[last_idx]) * 111_320.0
        dlon = (lon[i] - lon[last_idx]) * 111_320.0 * math.cos(math.radians(lat[i]))
        step_dist = math.hypot(dlat, dlon)

        if step_dist < 0.2:  # Ignora micro-ruido con el rover detenido
            continue

        accumulated_dist += step_dist
        total_dist_m += step_dist
        last_idx = i

        if accumulated_dist >= target_step_m:
            selected_indices.append(i)
            accumulated_dist = 0.0

    synced = synced.iloc[selected_indices].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. CÁLCULO DE CELDA TEMPORAL + ESPACIAL Y FIX PARA PYARROW
    # ------------------------------------------------------------------
    dia = pd.to_datetime(synced["epoch_ms"], unit="ms").dt.strftime("%Y-%m-%d")
    celda_espacial = [gps_cell(la, lo, cell_size_m) for la, lo in zip(synced["latitude"], synced["longitude"])]

    # Convertimos la tupla directamente a string para evitar el ArrowTypeError
    synced["cell"] = [
        None if c is None else f"{d}_{c[0]}_{c[1]}" if isinstance(c, tuple) else f"{d}_{c}"
        for c, d in zip(celda_espacial, dia)
    ]
    synced = synced.dropna(subset=["cell"]).astype({"cell": str})

    stats = {
        "ride_folder": ride_dir.name,
        "n_frames_synced": len(synced),
        "frac_valid_gps": round(frac_valid_gps, 4),
        "dist_m": round(total_dist_m, 1),
        "n_unique_cells": synced["cell"].nunique(),
        "cells_json": json.dumps(list(synced["cell"].unique())),
    }
    return synced, stats


def cmd_clean(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    ride_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    print(f"[clean] {len(ride_dirs)} rides extraídos a limpiar")
    print(f"[clean] espaciado entre fotos: {args.target_step_m} m -- tamaño de celda para dedup: {args.cell_size_m} m")

    rows = []
    for i, ride_dir in enumerate(ride_dirs, 1):
        try:
            synced, stats = clean_ride(ride_dir, args.cell_size_m, args.target_step_m)
        except Exception as e:
            print(f"[clean]  ! {ride_dir.name}: {e}")
            continue
        synced.to_parquet(clean_dir / f"{ride_dir.name}.synced.parquet", index=False)
        rows.append(stats)
        if i % 25 == 0:
            print(f"[clean]  {i}/{len(ride_dirs)} procesados")

    index = pd.DataFrame(rows)
    index.to_csv(args.index, index=False)
    print(f"[clean] listo — índice en {args.index}")
    if len(index):
        print(f"[clean] % gps válido promedio: {index['frac_valid_gps'].mean()*100:.1f}%")


# ----------------------------------------------------------------------
# 4) curate — dedup por celda GPS, greedy hasta llegar a las horas pedidas
# ----------------------------------------------------------------------

def cmd_curate(args: argparse.Namespace) -> None:
    index = pd.read_csv(args.index)
    index["cells"] = index["cells_json"].apply(json.loads).apply(set)

    meta = load_metadata()[["folder", "db_dur_sec"]]
    index = index.merge(meta, left_on="ride_folder", right_on="folder", how="left")

    remaining = index.sort_values("n_unique_cells", ascending=False).to_dict("records")
    seen_cells: set = set()
    chosen = []
    total_h = 0.0
    target_h = args.hours

    while remaining and total_h < target_h:
        remaining.sort(key=lambda r: len(r["cells"] - seen_cells), reverse=True)
        best = remaining.pop(0)
        new_cells = best["cells"] - seen_cells
        if not new_cells and chosen:
            break
        seen_cells |= best["cells"]
        chosen.append(best)
        total_h += (best.get("db_dur_sec") or 0) / 3600

    out = pd.DataFrame(chosen).drop(columns=["cells", "cells_json", "folder"], errors="ignore")
    out.to_csv(args.out, index=False)
    print(f"[curate] {len(out)} rides elegidos, {total_h:.1f} h, {len(seen_cells)} celdas únicas -> {args.out}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="filtrar metadata.parquet sin bajar tars")
    s.add_argument("--out", default="selected_rides.csv")
    s.add_argument("--require-rear", action="store_true", help="exigir cámara trasera")
    s.add_argument("--countries", default=None, help="lista separada por comas")
    s.add_argument("--max-per-country", type=int, default=None)
    s.set_defaults(func=cmd_select)

    f = sub.add_parser("fetch", help="bajar shards y extraer solo los rides seleccionados")
    f.add_argument("--rides", default="selected_rides.csv")
    f.add_argument("--raw-dir", default="raw")
    f.add_argument("--skip-recordings", action="store_true", help="no extraer video/audio, solo CSVs")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("clean", help="unidades + filtro GPS + sincronización por frame")
    c.add_argument("--raw-dir", default="raw")
    c.add_argument("--clean-dir", default="clean")
    c.add_argument("--index", default="cleaned_index.csv")
    c.add_argument("--cell-size-m", type=float, default=15.0,
                   help="tamaño de celda GPS para el dedup de `curate` (diversidad geografica)")
    c.add_argument("--target-step-m", type=float, default=5.0,
                   help="cada cuantos metros de movimiento real se guarda una foto (evita casi-duplicados)")
    c.set_defaults(func=cmd_clean)

    u = sub.add_parser("curate", help="dedup por celda GPS hasta llegar a N horas")
    u.add_argument("--index", default="cleaned_index.csv")
    u.add_argument("--hours", type=float, default=300.0)
    u.add_argument("--out", default="curated_rides.csv")
    u.set_defaults(func=cmd_curate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
