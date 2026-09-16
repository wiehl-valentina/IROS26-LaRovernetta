"""Patch: rumbo anclado por GPS (GpsAnchoredHeading).

Uso (desde la carpeta genie/ del repo):
    python patch_rumbo_gps.py            # aplica
    python patch_rumbo_gps.py --check    # solo verifica que las anclas existan

Toca tres archivos y deja un .bak de cada uno:
    genie_rover/navigation.py   clase GpsAnchoredHeading + self-test
    genie_rover/bridge.py       la usa como rumbo (heading_mode: gps_anclado)
    configs/frodobot_rover.yaml claves heading_* nuevas

Cada ancla tiene que aparecer EXACTAMENTE una vez; si no, aborta sin escribir
nada. Si el archivo ya tiene el patch, lo saltea.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARCA = "GpsAnchoredHeading"

# ============================================================ navigation.py

NAV_CLASE = '''

# ------------------------------------------------------- rumbo anclado por GPS

def circ_mean_deg(xs) -> float:
    """Media circular en grados [0, 360)."""
    r = np.radians(np.asarray(list(xs), dtype=np.float64))
    return math.degrees(math.atan2(float(np.mean(np.sin(r))), float(np.mean(np.cos(r))))) % 360.0


def circ_median_deg(xs) -> float:
    """Mediana circular: mediana de las diferencias contra la media circular.
    Robusta a los tramos donde el GPS salto y dio una estimacion disparatada."""
    xs = list(xs)
    m = circ_mean_deg(xs)
    return (m + float(np.median([wrap_deg(x - m) for x in xs]))) % 360.0


class GpsAnchoredHeading:
    """Rumbo = fuente relativa + offset absoluto H0 estimado por GPS.

    La fuente relativa (rel_deg, convencion brujula: 0 = norte, horario +)
    sigue los giros al instante pero puede tener un offset desconocido:

      * "brujula": el 'orientation' del SDK. En un robot del ERC en otro pais
        el offset cambia (hierro de los motores de ESE robot + declinacion
        magnetica del lugar). H0 termina siendo ese offset, aprendido en
        marcha. No depende de la frecuencia del lazo.
      * "gyro": -degrees(pose.theta) de odometry.py (theta es antihorario).
        Sin sesgo magnetico, pero solo sigue los giros si odometry.update se
        llama mas seguido que OdometryConfig.max_dt_s: si no, los intervalos
        largos se descartan y theta se pierde parte de cada giro.

    Por que el GPS si sirve aca y en HeadingEstimator no: _heading_from_track
    compara la CUERDA entre dos muestras GPS contra el rumbo ACTUAL, y
    mientras el robot gira la cuerda no dice hacia donde mira ahora. Aca la
    cuerda se compara contra la direccion MEDIA en la que el robot anduvo
    durante ese mismo tramo, medida con la fuente relativa:

        H0 = bearing_gps(cuerda) - media_circular(rel, pesada por avance)

    Eso vale aunque el robot zigzaguee: el peso de cada muestra es el comando
    lineal (girar en el lugar pesa 0), asi que la media sale en la direccion
    de la cuerda. Si el tramo se enrosca demasiado (concentracion de rel por
    debajo de r_min, p.ej. un giro de 180) se descarta: ahi la media ya no
    representa la cuerda. H0 es la mediana circular de las ultimas
    n_estimaciones; un salto de GPS arruina a lo sumo un tramo.

    Hasta tener min_estimaciones devuelve fallback_deg (el rumbo legacy).
    """

    def __init__(self, fuente: str = "brujula", tramo_m: float = 10.0,
                 r_min: float = 0.85, v_max_mps: float = 1.5,
                 n_estimaciones: int = 7, min_estimaciones: int = 2,
                 margen_salto_m: float = 6.0):
        self.fuente = str(fuente)
        self.tramo_m = float(tramo_m)
        self.r_min = float(r_min)
        self.v_max = float(v_max_mps)
        self.min_est = max(1, int(min_estimaciones))
        self.margen_salto_m = float(margen_salto_m)
        self.est: deque[float] = deque(maxlen=max(1, int(n_estimaciones)))
        self.historial: list[float] = []   # todas las estimaciones, para el resumen
        self.h0: float | None = None
        self.rechazos = {"curva": 0, "salto": 0, "sin_fix": 0}
        self._ref: tuple[float, float] | None = None
        self._anchor: tuple[float, float, float] | None = None   # (norte, este, t)
        self._sum_s = 0.0
        self._sum_c = 0.0
        self._sum_w = 0.0
        self._heading: float | None = None
        self._source = "none"

    # ---------------------------------------------------------------- tramo

    def _abrir_tramo(self, n: float, e: float, t: float) -> None:
        self._anchor = (n, e, t)
        self._sum_s = self._sum_c = self._sum_w = 0.0

    def _avanzar_tramo(self, n: float, e: float, t: float, rel: float, peso: float) -> None:
        an, ae, at = self._anchor
        d = math.hypot(n - an, e - ae)
        if d > self.v_max * max(t - at, 0.0) + self.margen_salto_m:
            # Mas rapido de lo que el robot puede andar: salto de GPS.
            self.rechazos["salto"] += 1
            self._abrir_tramo(n, e, t)
            return
        if peso > 0.0:
            self._sum_s += peso * math.sin(math.radians(rel))
            self._sum_c += peso * math.cos(math.radians(rel))
            self._sum_w += peso
        if d < self.tramo_m or self._sum_w <= 0.0:
            return
        r = math.hypot(self._sum_s, self._sum_c) / self._sum_w
        if r < self.r_min:
            self.rechazos["curva"] += 1      # se enrosco: la media no es la cuerda
        else:
            bearing = math.degrees(math.atan2(e - ae, n - an)) % 360.0
            rel_media = math.degrees(math.atan2(self._sum_s, self._sum_c))
            est = wrap_deg(bearing - rel_media)
            self.est.append(est)
            self.historial.append(est)
            if len(self.est) >= self.min_est:
                self.h0 = wrap_deg(circ_median_deg(self.est))
        self._abrir_tramo(n, e, t)

    # --------------------------------------------------------------- update

    def update(self, lat: float | None, lon: float | None, rel_deg: float | None,
               t: float, fallback_deg: float | None = None,
               peso: float = 1.0) -> float | None:
        """rel_deg en convencion brujula. peso = cuanto avanzo el robot en
        esta muestra (el comando lineal; 0 = girando en el lugar). peso < 0
        (reversa) corta el tramo: la cuerda apuntaria al reves."""
        if rel_deg is None:
            self._heading, self._source = fallback_deg, f"{self.fuente}: sin lectura"
            return self._heading
        rel = float(rel_deg) % 360.0

        fix_ok = (lat is not None and lon is not None
                  and abs(float(lat)) <= 90.0 and abs(float(lon)) <= 180.0)
        if not fix_ok:
            self.rechazos["sin_fix"] += 1   # (1000, 1000) = sin fix en el SDK
        else:
            if self._ref is None:
                self._ref = (float(lat), float(lon))
            n, e = latlon_to_local_ne(self._ref[0], self._ref[1], float(lat), float(lon))
            if self._anchor is None or peso < 0.0:
                self._abrir_tramo(n, e, t)
            else:
                self._avanzar_tramo(n, e, t, rel, float(peso))

        if self.h0 is not None:
            self._heading = (rel + self.h0) % 360.0
            self._source = f"gps+{self.fuente} H0={self.h0:+.0f} n={len(self.est)}"
        else:
            self._heading = fallback_deg
            self._source = (f"legacy (gps+{self.fuente} sin H0: "
                            f"{len(self.est)}/{self.min_est} tramos)")
        return self._heading

    def reset_track(self) -> None:
        """Cortar el tramo en curso. H0 se conserva: sigue valiendo."""
        self._anchor = None

    @property
    def heading(self) -> float | None:
        return self._heading

    @property
    def source(self) -> str:
        return self._source
'''

NAV_ANCLA_CLASE = "\n\n# --------------------------------------------------------------------- meta\n"

NAV_TEST = '''
    print("\\n=== rumbo anclado por GPS ===")
    lat_n, lon_n = -1.2921, 36.8219      # Nairobi: mismo codigo, otra brujula

    def en(norte_m: float, este_m: float) -> tuple[float, float]:
        return (lat_n + math.degrees(norte_m / EARTH_R),
                lon_n + math.degrees(este_m / (EARTH_R * math.cos(math.radians(lat_n)))))

    rng = np.random.default_rng(0)
    sesgo = -35.0                         # la brujula marca 35 grados de menos
    ah = GpsAnchoredHeading(fuente="brujula", tramo_m=8.0, min_estimaciones=2)
    h = ah.update(*en(0, 0), 90.0 + sesgo, 0.0, fallback_deg=55.0)
    print(f"  sin tramos todavia: heading={h} ({ah.source})")
    assert h == 55.0, "sin H0 tiene que devolver el fallback"

    # 50 s hacia el ESTE en zigzag de +-35 grados (como el seguidor real),
    # 5 muestras por segundo, GPS a 1 Hz con 1 m de ruido.
    norte = este = 0.0
    t = 0.0
    fix = en(0, 0)
    for k in range(250):
        t += 0.2
        rumbo = 90.0 + 35.0 * math.sin(2 * math.pi * t / 4.0)
        norte += 0.16 * math.cos(math.radians(rumbo))
        este += 0.16 * math.sin(math.radians(rumbo))
        if k % 5 == 0:
            fix = en(norte + rng.normal(0, 1.0), este + rng.normal(0, 1.0))
        h = ah.update(*fix, rumbo + sesgo + rng.normal(0, 3.0), t, fallback_deg=55.0)
    real = 90.0 + 35.0 * math.sin(2 * math.pi * t / 4.0)
    print(f"  zigzag al este: heading={h:.1f} (real {real:.1f})  {ah.source}  "
          f"rechazos={ah.rechazos}")
    assert ah.h0 is not None and abs(wrap_deg(ah.h0 + sesgo)) < 8, "H0 no aprendio el sesgo"
    assert abs(wrap_deg(h - real)) < 12

    t += 0.2
    h = ah.update(*fix, 0.0 + sesgo, t, fallback_deg=0.0, peso=0.0)
    print(f"  giro a mirar al norte en el lugar: heading={h:.1f} (real 0)")
    assert abs(wrap_deg(h)) < 12, "el rumbo no sigue el giro al instante"

    h0_antes, saltos = ah.h0, ah.rechazos["salto"]
    ah.reset_track()
    ah.update(*en(norte, este), sesgo, t + 1.0)
    ah.update(*en(norte + 40.0, este), sesgo, t + 2.0)
    print(f"  salto de GPS de 40 m en 1 s: rechazos por salto {saltos} -> {ah.rechazos['salto']}")
    assert ah.rechazos["salto"] == saltos + 1 and ah.h0 == h0_antes

    k = len(ah.historial)
    for i in range(20):                   # reversa hacia el sur mirando al norte
        norte -= 0.8
        ah.update(*en(norte, este), sesgo, t + 3.0 + i, peso=-0.2)
    assert len(ah.historial) == k, "en reversa no tiene que estimar"

    # Media vuelta en U andando: la media de rel no representa la cuerda.
    ah.reset_track()
    curvas = ah.rechazos["curva"]
    for i in range(60):
        ang = 180.0 * i / 59.0
        norte += 0.25 * math.cos(math.radians(ang))
        este += 0.25 * math.sin(math.radians(ang))
        ah.update(*en(norte, este), ang + sesgo, t + 30.0 + 0.25 * i)
    print(f"  vuelta en U: rechazos por curva {curvas} -> {ah.rechazos['curva']}")
    assert ah.rechazos["curva"] > curvas and len(ah.historial) == k

    h = ah.update(1000.0, 1000.0, sesgo, t + 60.0)
    assert ah.rechazos["sin_fix"] == 1 and h is not None

    # Fuente gyro: theta antihorario -> rel = -theta. Arranca mirando al norte.
    ag = GpsAnchoredHeading(fuente="gyro", tramo_m=8.0, min_estimaciones=1)
    for i in range(15):
        ag.update(*en(0.8 * i, 0.0), 0.0, float(i))
    print(f"  gyro, 11 m al norte: heading={ag.heading:.1f} (real 0)")
    assert abs(wrap_deg(ag.heading)) < 5
    h = ag.update(*en(11.2, 0.0), -90.0, 15.5, peso=0.0)   # theta +90 = izquierda
    print(f"  gyro, giro 90 izquierda: heading={h:.1f} (real 270)")
    assert abs(wrap_deg(h - 270.0)) < 5
'''

NAV_ANCLA_TEST = '    print("\\nTodos los asserts pasaron.")\n'

# ================================================================= bridge.py

BR_IMPORT_OLD = "    DriveCommand,\n    HeadingEstimator,\n"
BR_IMPORT_NEW = "    DriveCommand,\n    GpsAnchoredHeading,\n    HeadingEstimator,\n"

BR_INIT_ANCLA = '        safety = cfg.get("safety", {})\n        self.stale_frame_s'
BR_INIT_NEW = '''        # ---- rumbo anclado por GPS (GpsAnchoredHeading, navigation.py) ------
        # legacy      -> el HeadingEstimator de siempre.
        # gps_anclado -> fuente relativa (brujula o gyro) + offset H0 que se
        #                re-estima en cada tramo medido por GPS.
        # El HeadingEstimator sigue corriendo igual: da el rumbo hasta que hay
        # H0 y queda en el log como 'legado=' para comparar las dos fuentes.
        self.heading_mode = str(nav.get("heading_mode", "legacy"))
        self.heading_relativo = str(nav.get("heading_relativo", "brujula"))
        self.anchored_heading: GpsAnchoredHeading | None = None
        if self.heading_mode == "gps_anclado":
            if self.heading_relativo not in ("brujula", "gyro"):
                raise SystemExit(f"navigation.heading_relativo debe ser brujula o gyro, "
                                 f"no {self.heading_relativo!r}")
            if self.heading_relativo == "gyro" and self.odometry is None:
                raise SystemExit("heading_relativo: gyro usa la odometria: hace falta "
                                 "memory.enabled: true")
            self.anchored_heading = GpsAnchoredHeading(
                fuente=self.heading_relativo,
                tramo_m=float(nav.get("heading_tramo_m", 10.0)),
                r_min=float(nav.get("heading_min_r", 0.85)),
                v_max_mps=float(nav.get("heading_v_max_mps", 1.5)),
                n_estimaciones=int(nav.get("heading_n_estimaciones", 7)),
                min_estimaciones=int(nav.get("heading_min_estimaciones", 2)),
                margen_salto_m=float(nav.get("heading_margen_salto_m", 6.0)),
            )
        elif self.heading_mode != "legacy":
            raise SystemExit(f"navigation.heading_mode debe ser legacy o gps_anclado, "
                             f"no {self.heading_mode!r}")
        # Ultimo comando lineal enviado: pesa cada muestra del tramo GPS
        # (girar en el lugar = 0, reversa = corta el tramo).
        self._last_linear = 0.0
        # Huecos de la odometria: /data trae solo los ultimos ~0.1 s de gyro y
        # ruedas, y Odometry descarta intervalos > max_dt_s. Si el lazo tarda
        # mas que eso, el movimiento de ese intervalo se pierde.
        self._odo_llamadas = 0
        self._odo_huecos = 0
        self._last_odo_update_t: float | None = None

'''

BR_SEND_OLD = '''    def send(self, cmd: DriveCommand) -> None:
        tag = "DRY-RUN" if self.dry_run else "ENVIADO"'''
BR_SEND_NEW = '''    def send(self, cmd: DriveCommand) -> None:
        self._last_linear = float(cmd.linear)
        tag = "DRY-RUN" if self.dry_run else "ENVIADO"'''

BR_HELPERS_ANCLA = "    def refresh_checkpoints(self) -> None:\n"
BR_HELPERS_NEW = '''    def _reset_heading_tracks(self) -> None:
        """Despues de maniobras en el lugar: el tramo en curso ya no vale."""
        self.heading_est.reset_track()
        if self.anchored_heading is not None:
            self.anchored_heading.reset_track()

    def _odometry_update(self, raw: dict) -> Pose:
        """odometry.update del lazo principal, contando los huecos."""
        assert self.odometry is not None
        t = time.time()
        if (self._last_odo_update_t is not None
                and t - self._last_odo_update_t > self.odometry.cfg.max_dt_s):
            self._odo_huecos += 1
        self._last_odo_update_t = t
        self._odo_llamadas += 1
        return self.odometry.update(raw)

'''

BR_STEP_OLD = '''        heading = self.heading_est.update(telem.latitude, telem.longitude,
                                          telem.orientation, telem.timestamp)
'''
BR_STEP_NEW = '''        heading = self.heading_est.update(telem.latitude, telem.longitude,
                                          telem.orientation, telem.timestamp)
        heading_legado = heading
        pose_tel: Pose | None = None
        if self.anchored_heading is not None:
            if self.heading_relativo == "gyro":
                pose_tel = self._odometry_update(telem.raw)
                rel = -math.degrees(pose_tel.theta)     # theta es antihorario
            else:
                rel = self.heading_est.last_orientation_heading
            heading = self.anchored_heading.update(
                telem.latitude, telem.longitude, rel, now,
                fallback_deg=heading_legado, peso=self._last_linear)
'''

BR_MAP_OLD = "            pose_now = self.odometry.update(telem.raw)\n"
BR_MAP_NEW = ("            pose_now = (pose_tel if pose_tel is not None\n"
              "                        else self._odometry_update(telem.raw))\n")

BR_LOG_OLD = '''        print(f"[{self.stats.iterations:04d}] rumbo={heading if heading is None else round(heading)} "
              f"({self.heading_est.source})  meta: {goal_desc}  "
              f"celdas BEV={res.stats['bev_observed_cells']:.0f}{nota_mapa}{nota_desac}")'''
BR_LOG_NEW = '''        fuente_rumbo = self.heading_est.source
        if self.anchored_heading is not None:
            fuente_rumbo = self.anchored_heading.source
            if self.anchored_heading.h0 is not None and heading_legado is not None:
                nota_desac += f"  legado={round(heading_legado)}"
        print(f"[{self.stats.iterations:04d}] rumbo={heading if heading is None else round(heading)} "
              f"({fuente_rumbo})  meta: {goal_desc}  "
              f"celdas BEV={res.stats['bev_observed_cells']:.0f}{nota_mapa}{nota_desac}")'''

BR_RESET_OLD = "self.heading_est.reset_track()"
BR_RESET_NEW = "self._reset_heading_tracks()"

BR_SUMMARY_OLD = "        self._print_heading_diagnosis()\n\n    def _print_heading_diagnosis(self) -> None:\n"
BR_SUMMARY_NEW = ("        self._print_heading_diagnosis()\n        self._print_anchored_diagnosis()\n"
                  "\n    def _print_heading_diagnosis(self) -> None:\n")

BR_DIAG_ANCLA = "\n\ndef main() -> int:\n"
BR_DIAG_NEW = '''

    def _print_anchored_diagnosis(self) -> None:
        """Huecos de odometria y estado del rumbo anclado por GPS."""
        if self._odo_llamadas:
            frac = self._odo_huecos / self._odo_llamadas
            print("\\n  --- odometria: huecos ---")
            print(f"  actualizaciones:        {self._odo_llamadas} "
                  f"(con hueco > {self.odometry.cfg.max_dt_s:.1f} s: {self._odo_huecos}, {frac:.0%})")
            if frac > 0.2:
                print("  -> El lazo es mas lento que max_dt_s: /data solo trae los ultimos"
                      " ~0.1 s, asi que la odometria pierde el avance y los giros de"
                      " cada hueco. Con esto theta NO sigue los giros: no uses"
                      " heading_relativo: gyro hasta resolverlo.")

        ah = self.anchored_heading
        if ah is None:
            return
        h = ah.historial
        print(f"\\n  --- rumbo anclado por GPS (fuente {ah.fuente}) ---")
        print(f"  tramos usados:          {len(h)}")
        print(f"  tramos descartados:     curva={ah.rechazos['curva']}  "
              f"salto_gps={ah.rechazos['salto']}  sin_fix={ah.rechazos['sin_fix']}")
        if len(h) < 3:
            print("  (pocos tramos para concluir. Si 'curva' es alto, el robot se"
                  " enrosca dentro de cada tramo: baja heading_tramo_m o heading_min_r)")
            return
        rad = np.radians(np.asarray(h, dtype=np.float64))
        media = math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
        r = float(np.hypot(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
        disp = math.degrees(math.sqrt(-2.0 * math.log(r))) if r > 1e-9 else 180.0
        print(f"  H0 medio:               {media:+.0f} grados (dispersion ~{disp:.0f})")
        if ah.fuente == "brujula":
            total = (self.heading_est.offset + media + 180.0) % 360.0 - 180.0
            print(f"  -> la brujula de ESTE robot, ACA, esta corrida {media:+.0f} grados"
                  f" respecto del offset configurado.")
            if disp < 15.0:
                print(f"     Es estable: para arrancar ya corregido, "
                      f"navigation.orientation_offset_deg: {total:+.0f}")
            else:
                print("     Varia mucho entre tramos (depende del rumbo: hierro de los"
                      " motores). Una constante no alcanza: deja gps_anclado.")
        elif disp > 20.0:
            print("  -> H0 inestable con gyro: revisar los huecos de arriba y"
                  " odometry.gyro_sign.")
'''

# ============================================================== yaml

YAML_ANCLA = "  orientation_sign: 1.0\n"
YAML_NEW = '''  orientation_sign: 1.0

  # ---- rumbo anclado por GPS (GpsAnchoredHeading en navigation.py) --------
  # El -3 grados medido arriba vale para NUESTRO robot, aca. Un robot del ERC
  # en otro pais (Africa) trae otro offset de brujula: hierro de sus motores
  # + declinacion magnetica del lugar. Con gps_anclado ese offset (H0) se
  # aprende en marcha, en cada tramo de heading_tramo_m medido por GPS
  # (sirve aunque el robot zigzaguee; se descartan los tramos que se enroscan).
  #   heading_mode:     legacy | gps_anclado
  #   heading_relativo: brujula -> no depende de la velocidad del lazo
  #                     gyro    -> sin sesgo magnetico, pero SOLO si el lazo
  #                                corre mas rapido que odometry max_dt_s
  #                                (0.5 s). Mirar "huecos" en el resumen.
  # Hasta tener heading_min_estimaciones tramos se usa el rumbo legacy.
  heading_mode: gps_anclado
  heading_relativo: brujula
  heading_tramo_m: 10.0          # ~3-4 veces la dispersion del GPS con el robot quieto
  heading_min_r: 0.85            # concentracion minima del rumbo en el tramo (zigzag +-35 ok, U no)
  heading_v_max_mps: 1.5         # desplazamiento GPS > v_max*dt + margen = salto, se descarta
  heading_margen_salto_m: 6.0    # ~2-3 veces la dispersion del GPS con el robot quieto
  heading_n_estimaciones: 7      # mediana de los ultimos N tramos
  heading_min_estimaciones: 2
'''


# ============================================================ maquinaria

def reemplazar(texto: str, viejo: str, nuevo: str, nombre: str, veces: int = 1) -> str:
    n = texto.count(viejo)
    if n != veces:
        raise SystemExit(f"[patch] ancla '{nombre}': aparece {n} veces, esperaba {veces}. "
                         "El archivo no es el que conozco; no toco nada.")
    return texto.replace(viejo, nuevo)


def insertar_antes(texto: str, ancla: str, bloque: str, nombre: str) -> str:
    return reemplazar(texto, ancla, bloque + ancla, nombre)


def patch_navigation(t: str) -> str:
    t = insertar_antes(t, NAV_ANCLA_CLASE, NAV_CLASE.rstrip("\n"), "navigation: seccion meta")
    t = insertar_antes(t, NAV_ANCLA_TEST, NAV_TEST, "navigation: fin del self-test")
    return t


def patch_bridge(t: str) -> str:
    t = reemplazar(t, BR_IMPORT_OLD, BR_IMPORT_NEW, "bridge: import")
    t = insertar_antes(t, BR_INIT_ANCLA, BR_INIT_NEW, "bridge: __init__ safety")
    t = reemplazar(t, BR_SEND_OLD, BR_SEND_NEW, "bridge: send")
    t = reemplazar(t, BR_RESET_OLD, BR_RESET_NEW, "bridge: reset_track", veces=3)
    t = insertar_antes(t, BR_HELPERS_ANCLA, BR_HELPERS_NEW, "bridge: refresh_checkpoints")
    t = reemplazar(t, BR_STEP_OLD, BR_STEP_NEW, "bridge: heading en _step")
    t = reemplazar(t, BR_MAP_OLD, BR_MAP_NEW, "bridge: odometry.update del mapa")
    t = reemplazar(t, BR_LOG_OLD, BR_LOG_NEW, "bridge: print del frame")
    t = reemplazar(t, BR_SUMMARY_OLD, BR_SUMMARY_NEW, "bridge: resumen")
    t = insertar_antes(t, BR_DIAG_ANCLA, BR_DIAG_NEW.rstrip("\n"), "bridge: def main")
    return t


def patch_yaml(t: str) -> str:
    if "heading_mode:" in t:
        raise SystemExit("[patch] el yaml ya tiene heading_mode; no lo toco.")
    return reemplazar(t, YAML_ANCLA, YAML_NEW, "yaml: orientation_sign")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="carpeta genie/ del repo")
    ap.add_argument("--config", default="configs/frodobot_rover.yaml")
    ap.add_argument("--check", action="store_true", help="verificar sin escribir")
    a = ap.parse_args()

    root = Path(a.root)
    trabajos = [
        (root / "genie_rover" / "navigation.py", patch_navigation, MARCA),
        (root / "genie_rover" / "bridge.py", patch_bridge, MARCA),
        (root / a.config, patch_yaml, "heading_mode:"),
    ]

    # Primero se calcula todo en memoria: si una sola ancla falla, no se
    # escribe ningun archivo (evita dejar el repo a medio parchear).
    salidas = []
    for path, fn, marca in trabajos:
        if not path.is_file():
            raise SystemExit(f"[patch] no encuentro {path}")
        texto = path.read_text(encoding="utf-8")
        if marca in texto:
            print(f"[patch] {path}: ya parcheado, lo salteo")
            continue
        salidas.append((path, fn(texto)))

    if a.check:
        print(f"[patch] --check: todas las anclas OK ({len(salidas)} archivos por parchear)")
        return 0
    for path, nuevo in salidas:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(nuevo, encoding="utf-8")
        print(f"[patch] {path}  (backup en {path.name}.bak)")
    print("[patch] listo. Verifica con:  python -m genie_rover.navigation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
