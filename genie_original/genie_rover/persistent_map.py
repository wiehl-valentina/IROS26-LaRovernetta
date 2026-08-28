"""Mapa de ocupacion persistente: el robot deja de olvidar.

El BEV que produce la percepcion esta pegado al robot y se descarta en cada
frame. De ahi salen los tres problemas de fondo:

  - esquiva un obstaculo y choca el de al lado, porque el segundo estaba
    fuera de la ventana de 2 metros
  - empieza a rodear por derecha, cambia a izquierda, y en el titubeo se
    queda sin espacio para ninguna de las dos
  - gira en circulos sin darse cuenta de que ya paso por ahi

Este modulo mantiene una grilla anclada AL MUNDO. El robot se mueve dentro de
ella, y cada observacion nueva se integra encima usando la pose de la
odometria. Lo que se vio hace dos segundos sigue estando aunque la camara ya
no lo mire.

Convenciones (las mismas que odometry y genie_path_planner):
  mundo:  x adelante, y izquierda, theta antihorario
  BEV:    fila 0 = lo mas lejos, ultima fila = el robot; columna 0 = izquierda
  celdas: -1 = nunca observada, 0 = intransitable, 1 = transitable

Autoprueba (no necesita robot ni GPU):
    python -m genie_rover.persistent_map
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .odometry import Pose


@dataclass
class MapConfig:
    size_m: float = 8.0                 # lado del mapa cuadrado
    resolution_m_per_px: float = 0.03   # igual que el BEV, para no reescalar
    # Cuanto pesa una observacion nueva frente a lo acumulado. Alto = reacciona
    # rapido pero olvida rapido; bajo = estable pero lento para corregirse.
    update_weight: float = 0.45
    # Por segundo, cuanto se acerca a 0.5 (desconocido) una celda que nadie
    # vuelve a observar. Evita que un obstaculo que ya no esta quede para
    # siempre, y que la deriva de la odometria se acumule indefinidamente.
    decay_per_s: float = 0.08
    # Si el robot se acerca a menos de esto del borde, el mapa se recentra.
    recenter_margin_m: float = 1.5
    # Confianza minima para que una celda cuente como observada al recortar la
    # ventana para el planner.
    min_confidence: float = 0.15


class PersistentMap:
    """Grilla de ocupacion en coordenadas del mundo.

    Guarda dos capas: 'value' (0 intransitable, 1 transitable) y 'conf'
    (cuanta evidencia hay). Separarlas permite distinguir "vi que era
    transitable" de "nunca lo vi", que para el planner es muy distinto.
    """

    def __init__(self, cfg: MapConfig):
        self.cfg = cfg
        self.n = int(round(cfg.size_m / cfg.resolution_m_per_px))
        self.value = np.full((self.n, self.n), 0.5, dtype=np.float32)
        self.conf = np.zeros((self.n, self.n), dtype=np.float32)
        # Coordenadas del mundo del centro del mapa
        self.origin_x = 0.0
        self.origin_y = 0.0
        self._last_t: float | None = None
        self.integrations = 0
        self.recenters = 0

    # ------------------------------------------------------------ coordenadas

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """(x, y) del mundo -> (fila, columna). x crece hacia arriba (fila
        decreciente), y crece hacia la izquierda (columna decreciente)."""
        r = self.cfg.resolution_m_per_px
        fila = int(round(self.n / 2 - (x - self.origin_x) / r))
        col = int(round(self.n / 2 - (y - self.origin_y) / r))
        return fila, col

    def cell_to_world(self, fila: int, col: int) -> tuple[float, float]:
        r = self.cfg.resolution_m_per_px
        x = self.origin_x + (self.n / 2 - fila) * r
        y = self.origin_y + (self.n / 2 - col) * r
        return x, y

    # -------------------------------------------------------------- recentrar

    def _maybe_recenter(self, pose: Pose) -> None:
        borde = self.cfg.size_m / 2 - self.cfg.recenter_margin_m
        dx = pose.x - self.origin_x
        dy = pose.y - self.origin_y
        if abs(dx) < borde and abs(dy) < borde:
            return

        # Desplazar el contenido en celdas enteras: asi no hay que interpolar
        # y el mapa no se degrada con cada recentrado.
        r = self.cfg.resolution_m_per_px
        d_fila = int(round(dx / r))
        d_col = int(round(dy / r))

        nuevo_v = np.full_like(self.value, 0.5)
        nuevo_c = np.zeros_like(self.conf)

        # Al mover el origen +dx, el contenido se corre -d_fila
        f0_src, f1_src = max(0, d_fila), min(self.n, self.n + d_fila)
        c0_src, c1_src = max(0, d_col), min(self.n, self.n + d_col)
        f0_dst, f1_dst = max(0, -d_fila), min(self.n, self.n - d_fila)
        c0_dst, c1_dst = max(0, -d_col), min(self.n, self.n - d_col)

        if f1_src > f0_src and c1_src > c0_src:
            nuevo_v[f0_dst:f1_dst, c0_dst:c1_dst] = self.value[f0_src:f1_src, c0_src:c1_src]
            nuevo_c[f0_dst:f1_dst, c0_dst:c1_dst] = self.conf[f0_src:f1_src, c0_src:c1_src]

        self.value, self.conf = nuevo_v, nuevo_c
        self.origin_x += d_fila * r
        self.origin_y += d_col * r
        self.recenters += 1

    # -------------------------------------------------------------- integrar

    def integrate(self, bev: np.ndarray, observed: np.ndarray, pose: Pose,
                  bev_forward_range_m: float, bev_side_range_m: float,
                  t: float | None = None) -> None:
        """Vuelca un BEV (centrado en el robot) sobre el mapa del mundo."""
        if t is not None:
            self._apply_decay(t)
        self._maybe_recenter(pose)

        h, w = bev.shape
        r = self.cfg.resolution_m_per_px

        # Coordenadas de cada celda del BEV en el marco del robot.
        # Fila h-1 = el robot; columna w/2 = el eje central.
        filas, cols = np.nonzero(observed)
        if filas.size == 0:
            return
        vals = bev[filas, cols]
        buenos = vals >= 0.0
        filas, cols, vals = filas[buenos], cols[buenos], vals[buenos]
        if filas.size == 0:
            return

        x_robot = (h - 1 - filas) * (bev_forward_range_m / max(h - 1, 1))
        y_robot = (w / 2 - cols) * (2 * bev_side_range_m / max(w, 1))

        # Robot -> mundo
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        x_mundo = pose.x + c * x_robot - s * y_robot
        y_mundo = pose.y + s * x_robot + c * y_robot

        f = np.rint(self.n / 2 - (x_mundo - self.origin_x) / r).astype(int)
        cc = np.rint(self.n / 2 - (y_mundo - self.origin_y) / r).astype(int)

        dentro = (f >= 0) & (f < self.n) & (cc >= 0) & (cc < self.n)
        f, cc, vals = f[dentro], cc[dentro], vals[dentro]
        if f.size == 0:
            return

        # Varias celdas del BEV pueden caer en la misma celda del mapa (las
        # cercanas al robot tienen mas resolucion). Promediamos por celda en
        # vez de dejar que gane la ultima.
        plano = f * self.n + cc
        orden = np.argsort(plano)
        plano, vals = plano[orden], vals[orden]
        bordes = np.concatenate([[0], np.nonzero(np.diff(plano))[0] + 1, [len(plano)]])
        idx_unicos = plano[bordes[:-1]]
        medias = np.add.reduceat(vals, bordes[:-1]) / np.diff(bordes)

        fu, cu = idx_unicos // self.n, idx_unicos % self.n
        alfa = self.cfg.update_weight
        self.value[fu, cu] = (1 - alfa) * self.value[fu, cu] + alfa * medias
        self.conf[fu, cu] = np.minimum(1.0, self.conf[fu, cu] + alfa)
        self.integrations += 1

    def _apply_decay(self, t: float) -> None:
        if self._last_t is None:
            self._last_t = t
            return
        dt = t - self._last_t
        self._last_t = t
        if dt <= 0 or dt > 5.0:
            return
        k = math.exp(-self.cfg.decay_per_s * dt)
        self.conf *= k
        # El valor tiende a "desconocido" al mismo ritmo que la confianza
        self.value = 0.5 + (self.value - 0.5) * k

    # --------------------------------------------------------------- consultar

    def extract_bev(self, pose: Pose, forward_range_m: float,
                    side_range_m: float, out_h: int, out_w: int
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Recorta una ventana del mapa como si fuera un BEV del robot.

        Devuelve (bev, observed) en el mismo formato que produce la percepcion,
        para que el planner no note la diferencia.
        """
        filas = np.arange(out_h)
        cols = np.arange(out_w)
        ff, cx = np.meshgrid(filas, cols, indexing="ij")

        x_robot = (out_h - 1 - ff) * (forward_range_m / max(out_h - 1, 1))
        y_robot = (out_w / 2 - cx) * (2 * side_range_m / max(out_w, 1))

        c, s = math.cos(pose.theta), math.sin(pose.theta)
        x_mundo = pose.x + c * x_robot - s * y_robot
        y_mundo = pose.y + s * x_robot + c * y_robot

        r = self.cfg.resolution_m_per_px
        mf = np.rint(self.n / 2 - (x_mundo - self.origin_x) / r).astype(int)
        mc = np.rint(self.n / 2 - (y_mundo - self.origin_y) / r).astype(int)

        dentro = (mf >= 0) & (mf < self.n) & (mc >= 0) & (mc < self.n)
        mf_c = np.clip(mf, 0, self.n - 1)
        mc_c = np.clip(mc, 0, self.n - 1)

        val = self.value[mf_c, mc_c]
        conf = self.conf[mf_c, mc_c]

        visto = dentro & (conf >= self.cfg.min_confidence)
        bev = np.where(visto, val, -1.0).astype(np.float32)
        observed = visto.astype(np.uint8)
        return bev, observed

    # ------------------------------------------------------------------ debug

    def to_image(self, pose: Pose | None = None) -> np.ndarray:
        """Render RGB del mapa. Rojo intransitable, verde transitable,
        gris nunca visto, azul el robot."""
        img = np.full((self.n, self.n, 3), 40, dtype=np.uint8)
        visto = self.conf >= self.cfg.min_confidence
        v = self.value
        img[..., 0] = np.where(visto, ((1 - v) * 255).astype(np.uint8), 40)
        img[..., 1] = np.where(visto, (v * 255).astype(np.uint8), 40)
        img[..., 2] = np.where(visto, 30, 40)
        if pose is not None:
            f, c = self.world_to_cell(pose.x, pose.y)
            if 0 <= f < self.n and 0 <= c < self.n:
                f0, f1 = max(0, f - 3), min(self.n, f + 4)
                c0, c1 = max(0, c - 3), min(self.n, c + 4)
                img[f0:f1, c0:c1] = (80, 80, 255)
        return img

    def stats(self) -> dict:
        visto = self.conf >= self.cfg.min_confidence
        return {
            "celdas_vistas": int(visto.sum()),
            "cobertura": float(visto.mean()),
            "integraciones": self.integrations,
            "recentrados": self.recenters,
            "origen": (self.origin_x, self.origin_y),
        }


# --------------------------------------------------------------------- pruebas

def _bev_sintetico(h=134, w=134, obstaculo_col=None, obstaculo_fila=None):
    """BEV con todo transitable salvo un cuadrado intransitable."""
    bev = np.ones((h, w), dtype=np.float32)
    observed = np.ones((h, w), dtype=np.uint8)
    if obstaculo_col is not None:
        f = obstaculo_fila if obstaculo_fila is not None else h // 3
        bev[f - 8:f + 8, obstaculo_col - 8:obstaculo_col + 8] = 0.0
    return bev, observed


def _self_test() -> None:
    print("=== ida y vuelta de coordenadas ===")
    m = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    for x, y in [(0, 0), (1.5, 0), (0, 1.5), (-2, 1), (3.2, -2.7)]:
        f, c = m.world_to_cell(x, y)
        xr, yr = m.cell_to_world(f, c)
        print(f"  ({x:+.2f},{y:+.2f}) -> celda ({f},{c}) -> ({xr:+.3f},{yr:+.3f})")
        assert abs(xr - x) < 0.03 and abs(yr - y) < 0.03

    print("\n=== LA PRUEBA QUE IMPORTA: recuerda lo que salio de la vista ===")
    m = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    bev, obs = _bev_sintetico(obstaculo_col=100)   # obstaculo a la derecha
    pose = Pose(0.0, 0.0, 0.0)
    for i in range(5):
        m.integrate(bev, obs, pose, 2.0, 1.2, t=i * 0.2)

    # El robot gira 180 grados: la camara deja de ver el obstaculo
    pose_girado = Pose(0.0, 0.0, math.pi)
    bev_vacio, obs_vacio = _bev_sintetico()        # ahora no ve nada malo
    for i in range(5):
        m.integrate(bev_vacio, obs_vacio, pose_girado, 2.0, 1.2, t=1.0 + i * 0.2)

    # Consultamos el mapa mirando hacia donde estaba el obstaculo
    bev_rec, obs_rec = m.extract_bev(Pose(0, 0, 0), 2.0, 1.2, 134, 134)
    celdas_malas = int(((bev_rec >= 0) & (bev_rec < 0.4)).sum())
    print(f"  celdas intransitables recordadas: {celdas_malas}")
    print(f"  {m.stats()}")
    assert celdas_malas > 50, "olvido el obstaculo que salio del campo visual"

    print("\n=== ve mas alla de la ventana del BEV ===")
    # Un frame cubre 2.0 x 2.4 m; a 3 cm por celda son ~5300 celdas del mapa,
    # muy por debajo de las 134x134 del BEV (el BEV es mas fino que el mapa).
    solo_uno = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    bev, obs = _bev_sintetico(obstaculo_col=110, obstaculo_fila=20)
    solo_uno.integrate(bev, obs, Pose(0, 0, 0), 2.0, 1.2, t=0.0)
    base = solo_uno.stats()["celdas_vistas"]

    m = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    m.integrate(bev, obs, Pose(0, 0, 0), 2.0, 1.2, t=0.0)
    for k in range(1, 6):
        m.integrate(bev, obs, Pose(0.3 * k, 0, 0), 2.0, 1.2, t=0.2 * k)
    s = m.stats()
    print(f"  un solo frame cubre:  {base} celdas")
    print(f"  tras avanzar 1.5 m:   {s['celdas_vistas']} celdas "
          f"({s['celdas_vistas']/base:.1f}x)")
    print(f"  cobertura del mapa: {s['cobertura']*100:.1f}%")
    assert s["celdas_vistas"] > base * 1.3, "no acumulo mas alla de un frame"

    print("\n=== rotacion: el obstaculo queda donde corresponde ===")
    m = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    bev, obs = _bev_sintetico(obstaculo_col=67, obstaculo_fila=30)  # al frente
    m.integrate(bev, obs, Pose(0, 0, math.radians(90)), 2.0, 1.2, t=0.0)
    # fila 30 de 134 sobre un alcance de 2.0 m => el obstaculo esta a
    # (133-30)/133*2.0 = 1.55 m delante del robot. Mirando a 90 grados,
    # "delante" es +y del mundo.
    d = (134 - 1 - 30) / (134 - 1) * 2.0
    f, c = m.world_to_cell(0.0, d)
    val = m.value[f, c]
    f2, c2 = m.world_to_cell(d, 0.0)   # donde estaria SIN aplicar la rotacion
    val2 = m.value[f2, c2]
    print(f"  con rotacion, mundo (0.00, {d:+.2f}): {val:.2f}  (esperado bajo)")
    print(f"  sin rotacion, mundo ({d:+.2f}, 0.00): {val2:.2f}  (deberia seguir 0.5)")
    assert val < 0.45, "la rotacion no se aplico bien al integrar"
    assert abs(val2 - 0.5) < 0.05, "puso el obstaculo sin rotar"

    print("\n=== decaimiento ===")
    m = PersistentMap(MapConfig(decay_per_s=0.5))
    bev, obs = _bev_sintetico(obstaculo_col=100)
    m.integrate(bev, obs, Pose(0, 0, 0), 2.0, 1.2, t=0.0)
    conf0 = float(m.conf.max())
    m._apply_decay(4.0)
    conf1 = float(m.conf.max())
    print(f"  confianza: {conf0:.3f} -> {conf1:.3f} tras 4 s sin observar")
    assert conf1 < conf0 * 0.5

    print("\n=== recentrado al alejarse ===")
    m = PersistentMap(MapConfig(size_m=8.0, recenter_margin_m=1.5))
    bev, obs = _bev_sintetico(obstaculo_col=100)
    m.integrate(bev, obs, Pose(0, 0, 0), 2.0, 1.2, t=0.0)
    for k in range(1, 20):
        m.integrate(bev, obs, Pose(0.5 * k, 0, 0), 2.0, 1.2, t=0.2 * k)
    s = m.stats()
    print(f"  recentrados: {s['recentrados']}, origen: "
          f"({s['origen'][0]:.2f}, {s['origen'][1]:.2f})")
    assert s["recentrados"] > 0, "no recentro al salirse"
    assert s["celdas_vistas"] > 1000, "perdio todo al recentrar"

    print("\n=== formato compatible con el planner ===")
    m = PersistentMap(MapConfig())
    bev, obs = _bev_sintetico(obstaculo_col=100)
    m.integrate(bev, obs, Pose(0, 0, 0), 2.0, 1.2, t=0.0)
    b, o = m.extract_bev(Pose(0, 0, 0), 2.0, 1.2, 134, 134)
    print(f"  bev {b.shape} {b.dtype}, rango [{b.min():.2f}, {b.max():.2f}]")
    print(f"  observed {o.shape} {o.dtype}, {int(o.sum())} celdas")
    assert b.shape == (134, 134) and b.dtype == np.float32
    assert o.dtype == np.uint8
    assert b.min() >= -1.0 and b.max() <= 1.0

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
