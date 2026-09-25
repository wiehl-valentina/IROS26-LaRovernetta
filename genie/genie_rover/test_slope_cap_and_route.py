"""Pruebas unitarias para integracion de rutas grabadas como guia secundaria."""

from pathlib import Path
import pytest

from genie_rover.route import RouteConfig, RouteFollower, cargar_rutas, resolver_ruta


def test_load_mision2_route():
    """Verifica que resolver_ruta y cargar_rutas carguen correctamente la ruta grabada mision2."""
    path = resolver_ruta("mision2")
    assert path.is_file(), f"No se encontro el archivo de ruta para mision2: {path}"

    follower = cargar_rutas(["mision2"])
    assert follower is not None
    assert len(follower.puntos) >= 5
    assert follower.total_m > 30.0

    # Simular actualizacion con el primer punto
    p0 = follower.puntos[0]
    follower.update(p0.lat, p0.lon, t=100.0)
    assert follower.progreso_m is not None
    assert follower.progreso_m < 1.0

    # Obtener objetivo con lookahead
    lat_tgt, lon_tgt = follower.objetivo()
    assert abs(lat_tgt - p0.lat) > 1e-7 or abs(lon_tgt - p0.lon) > 1e-7
    assert not follower.terminada
