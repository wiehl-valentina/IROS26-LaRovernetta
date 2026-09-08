"""Chequeo frontal graduado -- reemplaza el bool de `front_is_blocked`.

Destino: `genie/genie_rover/front_guard.py` (archivo nuevo, no toca
navigation.py, asi que `front_is_blocked` sigue existiendo para el resto del
repo y para los tests).

POR QUE EXISTE
--------------
`front_is_blocked(bev, resolution)` -- tal como lo llama bridge.py hoy, SIN
argumentos -- promedia la transitabilidad de una ventana de 0.60 m de ancho
(half_width_m=0.30) entre 0.25 m y 0.90 m adelante, y frena si menos del 50%
de esa ventana esta por encima de 0.4.

En un corredor de pasto de 0.5-0.7 m de ancho eso frena SIEMPRE, aunque el
robot (0.30 m) entre sobrado: los bordes de la ventana caen sobre el pasto
(no transitable), el promedio se desploma y `free_ratio < 0.5` da True en
todos los frames. El rover queda en "OBSTACULO al frente" de forma
permanente, sin plan, y termina en recuperacion / retroceso.

Tres problemas concretos, y como los arregla este modulo:

1. **La ventana es mas ancha que el robot.** Aca el ancho se lee del yaml
   (`safety.front.half_width_m`) y se mide contra el ancho REAL del rover mas
   un margen chico. Ademas el promedio se toma FILA POR FILA, no sobre el
   bloque entero: una fila con una piedra a 0.8 m no diluye a las filas
   limpias de 0.3 m.

2. **Era binario.** Frenar (linear=0) es la accion mas cara que existe en un
   corredor angosto: perdes el avance neto y entras al ciclo giro-mira-giro.
   Aca la salida es una distancia libre continua (`clearance_m`) y un
   `speed_scale` en [0,1]: lejos -> velocidad plena, cerca -> lento, muy
   cerca -> recien ahi se frena. El rover atraviesa el pasillo despacio en
   vez de plantarse.

3. **No tenia histeresis.** Un frame malo (una sombra, un frame viejo del
   SDK) frenaba. `persist_frames` exige N frames seguidos por debajo de
   `stop_m` antes de declarar STOP, y `release_frames` para volver a soltar.
   Esto es lo que `safety.obstacle_persist_frames` prometia y nunca se
   aplicaba al freno en si.

CONVENCIONES DEL BEV (iguales a navigation.front_is_blocked)
  - `bev[r, c]` en [0,1]: 1 = transitable, 0 = no. Negativo = nunca observado.
  - La ultima fila (`h-1`) es la posicion del robot; filas hacia arriba son
    mas lejos. La columna del medio es el eje del robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CLEAR = "CLEAR"
SLOW = "SLOW"
STOP = "STOP"


def min_ground_range_m(camera: dict, safety_margin: float = 1.6) -> float:
    """Primera distancia al suelo que la camara puede ver de verdad.

    LA CUENTA QUE FALTABA. Con la calibracion real del Mini (fy=924.6,
    cy=528.4, imagen 1080 px de alto, camara a 0.150 m, pitch 1.85 grados
    hacia abajo) el rayo de la ultima fila de la imagen cae al suelo a 0.23 m.
    Todo lo que este mas cerca que eso NO EXISTE para la camara.

    Y las filas apenas por encima de ese limite son las peores del frame:
    incidencia rasante contra el suelo (32 grados), distorsion de barril
    maxima (k1=-0.26), y encima ahi abajo suele aparecer la nariz del propio
    rover, que SAM-TP marca -- correctamente -- como no transitable.

    El default viejo era `near_m=0.15`, con `row_step_m=0.05` que sobre una
    resolucion de 0.03 m/px se redondea a un paso de 0.06 m. O sea que las
    filas evaluadas eran 0.15, 0.21, 0.27, 0.33 ... y el veredicto de frenar
    salia SIEMPRE de las dos primeras. En el log del 08/09 las 48 frenadas
    dieron clearance 0.21 (19 veces) o 0.27 (24 veces): ninguna otra. No es
    que hubiera una pared; es que el guard estaba leyendo el borde inferior
    de la imagen.

    `safety_margin` multiplica el limite geometrico para quedarse en filas
    donde la proyeccion todavia sirve. 1.6 sobre 0.23 m da ~0.37 m.
    """
    K = camera.get("intrinsics")
    size = camera.get("image_size")
    h_cam = float(camera.get("height_m", 0.15))
    pitch = float(camera.get("pitch_down_deg", 0.0))
    if not K or not size:
        return 0.40
    fy = float(K[1][1])
    cy = float(K[1][2])
    v_bottom = float(size[1]) - 1.0
    ang = math.degrees(math.atan((v_bottom - cy) / fy)) + pitch
    if ang <= 1e-3:
        return 0.40
    d_min = h_cam / math.tan(math.radians(ang))
    return float(d_min * safety_margin)


@dataclass
class FrontGuardConfig:
    """Todo en metros salvo lo que diga lo contrario.

    Los defaults son para un rover de ~0.30 m de ancho en corredor angosto
    (pasto a los costados). Para terreno abierto conviene subir
    `half_width_m` y `stop_m`.
    """

    # Medio ancho del corredor que se inspecciona. REGLA: ancho_real/2 +
    # 0.05-0.08 m de margen. Con el Mini (~0.30 m) -> 0.15 + 0.07 = 0.22.
    # Este es EL numero que decide si pasas por un hueco angosto o no.
    half_width_m: float = 0.22
    # Primera fila que se mira. Mas cerca que esto la camara NO VE: el limite
    # lo fija la geometria (ver min_ground_range_m), no el gusto. Si dejas
    # esto por debajo del limite real, el guard decide sobre filas
    # inobservables o sobre la nariz del rover, y frena siempre.
    # Se sobreescribe automaticamente desde el yaml en from_cfg(safety, camera).
    near_m: float = 0.40
    # Por debajo de esta distancia libre se frena de verdad. TIENE que ser
    # mayor que near_m: si no, la condicion es inalcanzable y el guard no
    # frena nunca. La regla es stop_m ~ near_m + un paso de fila.
    stop_m: float = 0.50
    # Entre stop_m y clear_m la velocidad se escala linealmente.
    clear_m: float = 1.40
    # Hasta donde se busca. Mas alla de esto no cambia nada la decision.
    max_check_m: float = 1.50
    # Una celda cuenta como pisable si supera esto. BAJO a proposito: en
    # pasto/tierra mezclada SAM-TP devuelve mucho 0.4-0.6 y con el 0.4 de
    # antes la mitad del camino bueno contaba como pared.
    traversable_thresh: float = 0.30
    # Fraccion de celdas CONOCIDAS de una fila que tienen que ser pisables
    # para que la fila cuente como libre.
    min_free_ratio: float = 0.55
    # Filas bloqueadas seguidas para creerle a la fila (anti ruido de 1 fila).
    blocked_rows_needed: int = 2
    # Paso de muestreo en filas.
    row_step_m: float = 0.05
    # Frames seguidos en STOP antes de declararlo, y frames seguidos fuera de
    # STOP antes de soltarlo.
    persist_frames: int = 3
    release_frames: int = 2
    # Una fila sin NINGUNA celda observada: True = se considera libre (no
    # frenamos a ciegas, mismo criterio que front_is_blocked). Dejalo en True.
    unknown_is_free: bool = True
    # Piso de velocidad mientras se avanza en modo SLOW (fraccion de
    # max_linear). Por debajo de ~0.25 el rover no vence la friccion del
    # pasto y se queda "avanzando" sin moverse.
    min_speed_scale: float = 0.35

    @classmethod
    def from_cfg(cls, safety: dict, camera: dict | None = None,
                 resolution_m: float | None = None) -> "FrontGuardConfig":
        """Lee `safety.front` del yaml; las claves ausentes usan el default.

        Si se pasa `camera`, `near_m` se DERIVA de la calibracion salvo que el
        yaml lo fije explicitamente. Es la unica forma de que este numero siga
        siendo correcto cuando alguien cambie la altura o el pitch.
        """
        front = safety.get("front", {}) or {}
        base = cls()
        known = {f: getattr(base, f) for f in base.__dataclass_fields__}
        vals = {}
        for k, default in known.items():
            v = front.get(k, default)
            vals[k] = int(v) if isinstance(default, int) and not isinstance(default, bool) else (
                bool(v) if isinstance(default, bool) else float(v))

        if camera is not None and "near_m" not in front:
            vals["near_m"] = round(min_ground_range_m(camera), 2)

        # El paso de fila se redondea a celdas enteras: pedir 0.05 m sobre una
        # resolucion de 0.03 da 0.06. Dejarlo implicito fue parte del problema,
        # asi que aca se hace explicito.
        if resolution_m:
            paso = max(1, int(round(vals["row_step_m"] / resolution_m))) * resolution_m
            vals["row_step_m"] = paso

        # Invariantes. Cualquiera de las dos rotas deja el guard inutil, y sin
        # este chequeo la unica senal es un rover que no se mueve.
        if vals["stop_m"] <= vals["near_m"]:
            vals["stop_m"] = vals["near_m"] + max(vals["row_step_m"], 0.06)
            print(f"[front_guard] stop_m <= near_m: lo subo a {vals['stop_m']:.2f} m")
        if vals["clear_m"] <= vals["stop_m"]:
            vals["clear_m"] = vals["stop_m"] + 0.60
        # A proposito NO se hereda safety.obstacle_persist_frames: esa clave
        # gobierna el disparo del REGIMEN CERCANO, que en bridge.py cuenta
        # frames que YA pasaron por este guard. Si las dos fueran el mismo
        # numero estarias exigiendo el doble de frames sin verlo.
        return cls(**vals)


@dataclass
class FrontState:
    clearance_m: float
    level: str          # CLEAR | SLOW | STOP
    speed_scale: float  # multiplicador de linear en [min_speed_scale, 1.0]; 0.0 si STOP
    blocked_frames: int
    reason: str

    @property
    def is_stop(self) -> bool:
        return self.level == STOP


def clearance_by_rows(bev: np.ndarray, resolution_m: float,
                      cfg: FrontGuardConfig) -> float:
    """Distancia libre al frente, decidida FILA POR FILA.

    Devuelve `cfg.max_check_m` si nunca encuentra un bloque de
    `blocked_rows_needed` filas bloqueadas seguidas.

    La diferencia con `front_clearance_m` de navigation.py es que aquella
    evalua una VENTANA acumulada (de near_m a la fila r), asi que una franja
    mala lejana contamina el veredicto de las cercanas. Aca cada fila se
    juzga sola, que es lo que corresponde: lo unico que importa para frenar
    es donde esta la primera pared, no cuanto verde hay en promedio.
    """
    if bev is None or bev.size == 0:
        return cfg.max_check_m
    h, w = bev.shape[:2]
    c_half = max(1, int(round(cfg.half_width_m / resolution_m)))
    c_mid = w // 2
    c0, c1 = max(0, c_mid - c_half), min(w, c_mid + c_half + 1)
    if c0 >= c1:
        return cfg.max_check_m

    step_rows = max(1, int(round(cfg.row_step_m / resolution_m)))
    seguidas = 0
    d = cfg.near_m
    while d <= cfg.max_check_m:
        r = h - 1 - int(round(d / resolution_m))
        if r < 0:
            break
        fila = bev[r, c0:c1]
        conocidas = fila[fila >= 0.0]
        if conocidas.size == 0:
            libre = cfg.unknown_is_free
        else:
            libre = float(np.mean(conocidas > cfg.traversable_thresh)) >= cfg.min_free_ratio

        if libre:
            seguidas = 0
        else:
            seguidas += 1
            if seguidas >= cfg.blocked_rows_needed:
                # La pared empieza en la primera de las filas bloqueadas.
                return max(0.0, d - (seguidas - 1) * step_rows * resolution_m)
        d += step_rows * resolution_m
    return cfg.max_check_m


class FrontGuard:
    """Estado con histeresis. Una instancia por bridge, viva entre frames."""

    def __init__(self, cfg: FrontGuardConfig | None = None):
        self.cfg = cfg or FrontGuardConfig()
        self._blocked_streak = 0
        self._free_streak = 0
        self._latched_stop = False

    # ---- consulta sin estado (para usar DENTRO de maniobras: retroceso,
    #      avance forzado, etc. donde no queres tocar la histeresis) --------
    def probe(self, bev: np.ndarray, resolution_m: float) -> float:
        return clearance_by_rows(bev, resolution_m, self.cfg)

    def probe_blocked(self, bev: np.ndarray, resolution_m: float) -> bool:
        """Reemplazo 1 a 1 de `front_is_blocked(...)` para los call sites que
        de verdad quieren un bool instantaneo (dentro de _retroceder, etc.)."""
        return self.probe(bev, resolution_m) < self.cfg.stop_m

    # ---- actualizacion con histeresis (una vez por iteracion de _step) ----
    def update(self, bev: np.ndarray, resolution_m: float) -> FrontState:
        c = self.cfg
        clearance = clearance_by_rows(bev, resolution_m, c)

        if clearance < c.stop_m:
            self._blocked_streak += 1
            self._free_streak = 0
        else:
            self._free_streak += 1
            self._blocked_streak = 0

        if not self._latched_stop and self._blocked_streak >= c.persist_frames:
            self._latched_stop = True
        elif self._latched_stop and self._free_streak >= c.release_frames:
            self._latched_stop = False

        if self._latched_stop:
            return FrontState(clearance, STOP, 0.0, self._blocked_streak,
                              f"pared a {clearance:.2f} m ({self._blocked_streak} frames)")

        if clearance >= c.clear_m:
            return FrontState(clearance, CLEAR, 1.0, self._blocked_streak, "libre")

        # Rampa lineal entre stop_m (min_speed_scale) y clear_m (1.0). Ojo:
        # si clearance < stop_m pero todavia no se cumplio persist_frames,
        # caemos aca con el minimo -- avanza lento, no frena. Es a proposito:
        # un frame malo suelto no tiene que costar el avance.
        span = max(1e-6, c.clear_m - c.stop_m)
        t = float(np.clip((clearance - c.stop_m) / span, 0.0, 1.0))
        scale = c.min_speed_scale + (1.0 - c.min_speed_scale) * t
        return FrontState(clearance, SLOW, scale, self._blocked_streak,
                          f"despacio, {clearance:.2f} m libres")

    def reset(self) -> None:
        self._blocked_streak = 0
        self._free_streak = 0
        self._latched_stop = False


# ---------------------------------------------------------------- self test

def _self_test() -> None:
    res = 0.03
    h = w = 134

    libre = np.ones((h, w), dtype=np.float32)
    g = FrontGuard()
    st = g.update(libre, res)
    print(f"  BEV libre                -> {st.level:5} clearance={st.clearance_m:.2f} scale={st.speed_scale:.2f}")
    assert st.level == CLEAR and st.speed_scale == 1.0

    # EL caso de Mexico: corredor de 0.40 m entre pasto, que ADEMAS se corre
    # 0.30 m para un costado en el horizonte de 0.9 m (una curva suave), y con
    # transitabilidad moderada adentro (0.52, tipico de tierra con pasto
    # pisado encima) en vez de un 0.9 de libro.
    corredor = np.full((h, w), 0.06, dtype=np.float32)
    media = int(0.20 / res)
    for r in range(h):
        d = (h - 1 - r) * res                      # metros hacia adelante
        centro = w // 2 - int(round((0.33 * d) / res))  # se corre a un lado
        corredor[r, max(0, centro - media):min(w, centro + media)] = 0.52
    g2 = FrontGuard()
    st = g2.update(corredor, res)
    print(f"  corredor 0.40 m curvo    -> {st.level:5} clearance={st.clearance_m:.2f} scale={st.speed_scale:.2f}")
    assert st.level != STOP, "un corredor por el que el robot entra NO puede dar STOP"

    # El mismo corredor con el chequeo viejo (ventana fija de 0.60 m, umbral
    # 0.4, promedio de todo el bloque de 0.25-0.90 m):
    c_half_viejo = int(0.30 / res)
    patch = corredor[h - 1 - int(0.9 / res):h - int(0.25 / res),
                     w // 2 - c_half_viejo:w // 2 + c_half_viejo + 1]
    ratio_viejo = float(np.mean(patch > 0.4))
    print(f"  (chequeo viejo, mismo BEV: free_ratio={ratio_viejo:.2f} "
          f"-> {'BLOQUEADO' if ratio_viejo < 0.5 else 'libre'})")
    assert ratio_viejo < 0.5, "este es justo el frame que hoy da 'OBSTACULO al frente'"

    # Pared real cruzando todo a 0.5 m.
    pared = np.ones((h, w), dtype=np.float32)
    r = h - 1 - int(0.5 / res)
    pared[r - 6:r + 1, :] = 0.02
    g3 = FrontGuard()
    for i in range(3):
        st = g3.update(pared, res)
    print(f"  pared a 0.50 m x3 frames -> {st.level:5} clearance={st.clearance_m:.2f}")
    assert st.level == SLOW, "0.50 m > stop_m=0.35: tiene que ir lento, no frenar"

    pegada = np.ones((h, w), dtype=np.float32)
    r = h - 1 - int(0.25 / res)
    pegada[:r + 1, :] = 0.02
    g4 = FrontGuard()
    niveles = [g4.update(pegada, res).level for _ in range(4)]
    print(f"  pared a 0.25 m, 4 frames -> {niveles}")
    assert niveles[0] == SLOW and niveles[-1] == STOP, "histeresis: no frena al primer frame, si al tercero"

    # Suelta despues de release_frames limpios.
    for _ in range(2):
        st = g4.update(libre, res)
    print(f"  y despues 2 frames libres-> {st.level}")
    assert st.level == CLEAR

    # EL CASO DEL LOG DEL 08/09: las filas de 0.15 a 0.33 m dan no
    # transitable (nariz del rover / suelo a incidencia rasante / distorsion
    # de barril en el borde inferior del frame) y de 0.40 m en adelante el
    # corredor esta perfectamente libre. Con near_m=0.15 el guard frena para
    # siempre; con near_m derivado de la calibracion, pasa.
    fantasma = np.ones((h, w), dtype=np.float32)
    fantasma[h - 1 - int(0.35 / res):, :] = 0.02
    viejo = FrontGuard(FrontGuardConfig(near_m=0.15, stop_m=0.35))
    niveles_viejo = [viejo.update(fantasma, res).level for _ in range(4)]
    cl_viejo = viejo.probe(fantasma, res)
    print(f"  pared fantasma <0.35 m   -> viejo (near_m=0.15): {niveles_viejo[-1]} "
          f"clearance={cl_viejo:.2f}")
    assert niveles_viejo[-1] == STOP, "asi se veia el bug"
    assert abs(cl_viejo - 0.21) < 0.07, "y con el clearance que aparecia en el log"

    nuevo = FrontGuard(FrontGuardConfig())  # near_m=0.40
    niveles_nuevo = [nuevo.update(fantasma, res).level for _ in range(4)]
    print(f"                           -> nuevo (near_m=0.40): {niveles_nuevo[-1]} "
          f"clearance={nuevo.probe(fantasma, res):.2f}")
    assert STOP not in niveles_nuevo, "con el campo cercano ignorado tiene que pasar"

    # La calibracion real del Mini.
    cam = {"intrinsics": [[925.27, 0.0, 962.31], [0.0, 924.63, 528.39], [0.0, 0.0, 1.0]],
           "image_size": [1920, 1080], "height_m": 0.150, "pitch_down_deg": 1.85}
    d = min_ground_range_m(cam, safety_margin=1.0)
    print(f"  min_ground_range_m(Mini) -> {d:.2f} m (con margen 1.6: "
          f"{min_ground_range_m(cam):.2f} m)")
    assert 0.20 < d < 0.27, d
    cfg_auto = FrontGuardConfig.from_cfg({"front": {}}, camera=cam, resolution_m=res)
    assert cfg_auto.near_m >= 0.30 and cfg_auto.stop_m > cfg_auto.near_m
    assert abs(cfg_auto.row_step_m - 0.06) < 1e-9, cfg_auto.row_step_m

    # El yaml manda si lo escribis a mano, pero los invariantes se respetan.
    cfg_manual = FrontGuardConfig.from_cfg(
        {"front": {"near_m": 0.45, "stop_m": 0.40}}, camera=cam, resolution_m=res)
    assert cfg_manual.near_m == 0.45 and cfg_manual.stop_m > 0.45

    # Nunca observado: no frena a ciegas.
    desconocido = np.full((h, w), -1.0, dtype=np.float32)
    g5 = FrontGuard()
    st = g5.update(desconocido, res)
    print(f"  BEV sin observar         -> {st.level} (no frena a ciegas)")
    assert st.level == CLEAR

    print("front_guard: todos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
