"""Evaluador semantico de la mision indoor por tramos -- SOLO Gemini.

Que hace, en una linea: mira el frame de la camara frontal y contesta UNA
pregunta de si/no sobre lo que se ve ("¿se ve un piano?", "¿hay una puerta a
la izquierda?"), con una confianza. No comanda motores, no elige rumbo, no
toca el planner: lo unico que hace es devolver un `VlmObservation` que
`SemanticMissionFSM` usa para decidir si el tramo actual termino.

Tres cosas que este modulo garantiza, y que son la razon de que exista en
vez de llamar al VLM directo desde `_step()`:

1. **Nunca bloquea el lazo de control.** El lazo corre a ~5 Hz; una llamada
   a Gemini tarda cientos de milisegundos o mas. La inferencia vive en un
   hilo aparte y `observe()` devuelve SIEMPRE al instante: o la ultima
   respuesta cacheada, o None. Si devuelve None el tramo sigue recto, que es
   el comportamiento seguro.
2. **Nunca voltea la mision.** Todo error (sin red, cuota agotada, JSON
   invalido, timeout) se traga con un aviso por consola y se comporta como
   "no hay respuesta".
3. **Nunca mezcla preguntas.** Cada respuesta queda etiquetada con el id de
   la pregunta que la origino. Si la FSM ya cambio de fase mientras la
   llamada estaba en vuelo, la respuesta vieja se descarta sola (la FSM
   compara el id) en vez de disparar un cambio de fase equivocado.

ELIMINADO: el backend "http" (vLLM sirviendo Qwen2-VL). Ya no existe
`base_url`, ni servidor local que levantar, ni Qwen. Los unicos valores
validos de `vlm.backend` son:

    "gemini"  (default)  llamada REST directa a la API de Gemini
                         (generativelanguage.googleapis.com), con
                         `responseSchema` para que la respuesta venga como
                         JSON estructurado y no haya que adivinar nada.
    "off"                lo apaga: la mision corre igual y cada tramo termina
                         por su fail-safe de distancia
                         (`mission.segments[].on_timeout`).

CREDENCIAL: `vlm.api_key` del config, o la variable de entorno
`GEMINI_API_KEY` / `GOOGLE_API_KEY` (en ese orden). Una API key de Google AI
Studio empieza con "AIza..."; si lo que tenes empieza con "AQ." o "ya29."
eso es un token OAuth de corta duracion, NO una API key, y va a fallar con
401 apenas expire.

CUANTO SE LE PREGUNTA (importa: la cuota de Gemini Flash es chica):

    vlm.max_rpm: 15          tope duro de pedidos por minuto (ventana
                             deslizante de 60 s). Al llegar al tope el frame
                             se saltea -- no se encola ni se bloquea el lazo.
    vlm.query_every_m: 0.5   ademas de la cadencia, exige que el robot se
    vlm.query_every_deg: 20  haya MOVIDO desde la consulta anterior: medio
                             metro de avance O 20 grados de giro, medidos
                             sobre la pose (odometry/RTAB-Map). Preguntar dos
                             veces desde el mismo lugar gasta cuota para ver
                             la misma escena.
    vlm.query_max_s: 8.0     red de seguridad: aunque el robot este quieto,
                             se pregunta igual cada tanto.

Un cambio de hito (de fase o de tramo) dispara SIEMPRE, sin esperar a
moverse: es justo cuando la respuesta cacheada dejo de servir.

VERIFICACION DE CONEXION: al construirse (`check_on_start: true`, que es el
default) hace UNA llamada de prueba con una imagen de 32x32 y reporta si
Gemini contesta, con que latencia y con que modelo. Si falla, lo dice con el
motivo y una pista de que revisar, y la mision arranca igual en modo
"sin VLM" en vez de morirse a mitad del pasillo.

OBSERVABILIDAD (lo que se ve por consola durante la mision): con
`vlm.log: true` (default) cada respuesta imprime una linea:

    [vlm] t+ 12.4s  piano          SI  conf=0.82  pos=centro  dist=3.1m  1.06s  "se ve un piano vertical contra la pared"
    [vlm] t+ 13.6s  piano          no  conf=0.30  pos=centro  dist=  -   0.91s  "solo hay un armario oscuro"

y con `vlm.log_dir: "debug/vlm"` ademas se guarda, por cada llamada, el JPG
EXACTO que se le mando y el JSON EXACTO que contesto
(`00007_piano.jpg` + `00007_piano.json`) -- asi despues podes mirar imagen y
respuesta lado a lado y entender por que decidio lo que decidio.

Autoprueba:
    # sin red: parseo, cadencia, backend off
    python -m genie_rover.Indoor.vlm_semantic

    # llamada REAL a Gemini con una imagen tuya
    python -m genie_rover.Indoor.vlm_semantic --image debug/run1/00007_rgb.jpg \\
        --prompt "¿se ve un piano?"

    # solo verificar que la credencial y el modelo andan
    python -m genie_rover.Indoor.vlm_semantic --check
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mission import VlmObservation, VlmQuery

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

INSTRUCCION = (
    "Sos el evaluador visual de un robot que avanza por un pasillo "
    "universitario. Mira la imagen de su camara frontal y responde la "
    "pregunta con un objeto JSON.\n"
    "Pone hito_presente en false si no estas seguro: una respuesta "
    "afirmativa hace que el robot cambie de rumbo.\n"
    "En 'razon' explica en una frase corta QUE viste para decidir eso.\n\n"
    "Pregunta: "
)

# Se le pasa a Gemini como responseSchema: la respuesta ya viene con estos
# campos y estos tipos, sin markdown ni texto alrededor.
ESQUEMA = {
    "type": "OBJECT",
    "properties": {
        "hito_presente": {"type": "BOOLEAN"},
        "confianza": {"type": "NUMBER"},
        "posicion": {"type": "STRING", "enum": ["izquierda", "centro", "derecha"]},
        "distancia_m": {"type": "NUMBER"},
        "razon": {"type": "STRING"},
    },
    "required": ["hito_presente", "confianza", "posicion", "razon"],
}


@dataclass
class VlmConfig:
    backend: str = "gemini"          # "gemini" | "off"
    # Modelos validos hoy: gemini-2.0-flash, gemini-2.5-flash,
    # gemini-2.5-flash-lite. OJO: gemini-1.5-flash/1.5-pro estan retirados
    # de la API publica -- si el config todavia dice 1.5, la verificacion de
    # arranque va a devolver 404 y avisar.
    model: str = "gemini-3.1-flash-lite"
    api_key: str = ""                # vacio = GEMINI_API_KEY / GOOGLE_API_KEY
    # --- cuanto se le puede preguntar ---------------------------------------
    # CUOTA DURA de la API (free tier de Gemini Flash: 15 pedidos por minuto).
    # Es una ventana deslizante de 60 s, no un promedio: si en el ultimo
    # minuto ya salieron `max_rpm` llamadas, la siguiente NO se dispara (se
    # saltea el frame, no se encola ni se bloquea el lazo). 0 = sin limite.
    max_rpm: int = 15

    interval_s: float = 1.0          # piso de tiempo entre consultas
    interval_turn_s: float = 0.5     # piso durante un giro, donde la latencia importa

    # --- cuando conviene preguntar (disparo por MOVIMIENTO) -----------------
    # Preguntar dos veces desde casi el mismo lugar gasta cuota para ver la
    # misma escena. Con esto, una consulta nueva necesita ademas que el robot
    # se haya MOVIDO desde la anterior: `query_every_m` de avance (medido
    # sobre la pose de odometry/RTAB-Map) O `query_every_deg` de giro -- el
    # OR es necesario porque en TURN_SEARCH el robot pivotea sin avanzar, y
    # con solo la distancia no preguntaria nunca.
    # 0 en ambos = desactiva el disparo por movimiento (vuelve a cadencia pura).
    query_every_m: float = 0.5
    query_every_deg: float = 20.0
    # Red de seguridad: aunque el robot este quieto (trabado, esperando,
    # alineando), se pregunta igual cada tanto. None/0 = nunca por tiempo.
    query_max_s: float = 8.0

    timeout_s: float = 6.0
    max_side_px: int = 640           # se reescala antes de mandar
    jpeg_quality: int = 80
    max_tokens: int = 128
    temperature: float = 0.0

    # --- observabilidad ------------------------------------------------------
    check_on_start: bool = True      # llamada de prueba al construirse
    log: bool = True                 # una linea por respuesta
    log_dir: str | None = None       # si se pone, guarda jpg + json de cada llamada

    @classmethod
    def from_dict(cls, d: dict) -> "VlmConfig":
        d = dict(d or {})
        # Claves del backend viejo (vLLM/Qwen): se ignoran con un aviso en vez
        # de explotar, para que un config sin actualizar siga arrancando.
        viejas = [k for k in ("base_url",) if k in d]
        if viejas:
            print(f"[vlm_semantic] AVISO: ignoro claves del backend viejo en "
                  f"`vlm:` ({', '.join(viejas)}). Ya no existe el servidor "
                  "local: el unico backend es Gemini.")
        if str(d.get("backend", "")).lower() == "http":
            print("[vlm_semantic] AVISO: vlm.backend 'http' ya no existe; "
                  "uso 'gemini'.")
            d["backend"] = "gemini"
        conocidas = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**conocidas)

    def resolved_api_key(self) -> str:
        return (self.api_key
                or os.environ.get("GEMINI_API_KEY", "")
                or os.environ.get("GOOGLE_API_KEY", "")).strip()


# ------------------------------------------------------------------- backend

class GeminiBackend:
    """Llamada REST directa a generateContent. Sin SDK, sin dependencias
    nuevas: urllib de la stdlib."""

    def __init__(self, cfg: VlmConfig):
        self.cfg = cfg
        self.api_key = cfg.resolved_api_key()
        if not self.api_key:
            raise RuntimeError(
                "falta la credencial de Gemini: poné `vlm.api_key` en el config "
                "o exportá GEMINI_API_KEY. Sacala de https://aistudio.google.com/apikey "
                "(empieza con 'AIza')."
            )
       

    def ask(self, jpg: bytes, prompt: str) -> dict:
        import urllib.request

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg",
                                     "data": base64.b64encode(jpg).decode("ascii")}},
                    {"text": INSTRUCCION + prompt},
                ],
            }],
            "generationConfig": {
                "temperature": self.cfg.temperature,
                "maxOutputTokens": self.cfg.max_tokens,
                "responseMimeType": "application/json",
                "responseSchema": ESQUEMA,
            },
        }
        req = urllib.request.Request(
            f"{API_BASE}/models/{self.cfg.model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(_explicar_error(exc, self.cfg)) from exc

        try:
            texto = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # Respuesta sin candidato: casi siempre un bloqueo por filtros de
            # seguridad o un corte por maxOutputTokens.
            raise RuntimeError(f"Gemini contesto sin candidato utilizable: "
                               f"{json.dumps(data)[:300]}")
        return _parse_json_laxo(texto)


def _explicar_error(exc: Exception, cfg: VlmConfig) -> str:
    """Traduce el error HTTP crudo a algo accionable."""
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        try:
            cuerpo = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            cuerpo = ""
        pistas = {
            400: "pedido invalido (revisa el modelo y el esquema)",
            401: "credencial rechazada: la API key esta mal o expiro",
            403: "credencial sin permiso para la API de Gemini (¿API habilitada?)",
            404: (f"el modelo {cfg.model!r} no existe o no esta disponible para "
                  "tu clave. Probá 'gemini-2.0-flash' o 'gemini-2.5-flash' "
                  "(los 1.5 estan retirados)"),
            429: "cuota agotada: bajá vlm.max_rpm o subí vlm.query_every_m",
            500: "error del lado de Google, reintentá",
            503: "modelo sobrecargado del lado de Google",
        }
        pista = pistas.get(exc.code, "")
        return f"HTTP {exc.code}: {pista}. Respuesta: {cuerpo}"
    if isinstance(exc, urllib.error.URLError):
        return (f"no hay red o DNS no resuelve generativelanguage.googleapis.com "
                f"({exc.reason})")
    return f"{type(exc).__name__}: {exc}"


def _parse_json_laxo(texto: str) -> dict:
    """Con responseSchema la respuesta ya es JSON limpio, pero el modelo a
    veces igual la envuelve en ``` o le pone texto alrededor. En vez de
    fallar, recortamos del primer '{' al ultimo '}'."""
    t = (texto or "").strip()
    if "{" in t and "}" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    return json.loads(t)


def _rgb_a_jpg(rgb: np.ndarray, max_side_px: int, quality: int) -> bytes:
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    lado = max(img.size)
    if max_side_px and lado > max_side_px:
        escala = max_side_px / float(lado)
        img = img.resize((max(1, int(img.width * escala)),
                          max(1, int(img.height * escala))))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


# -------------------------------------------------------------------- poller

class SemanticVlm:
    """Un hilo, una pregunta en vuelo a la vez, una respuesta cacheada."""

    def __init__(self, cfg: VlmConfig):
        self.cfg = cfg
        self.calls = 0
        self.errors = 0
        self.last_latency_s = 0.0
        self.latencies: list[float] = []
        self.connected: bool | None = None      # None = no se verifico

        self._backend = None
        if cfg.backend == "gemini":
            try:
                self._backend = GeminiBackend(cfg)
            except Exception as exc:
                print(f"[vlm_semantic] ERROR: no puedo usar Gemini: {exc}")
                print("[vlm_semantic] sigo SIN VLM: cada tramo va a terminar por "
                      "su fail-safe de distancia (segments[].on_timeout).")
                self._backend = None
        elif cfg.backend != "off":
            raise ValueError(f"vlm.backend desconocido: {cfg.backend!r} "
                             "(gemini | off)")

        self._log_dir = Path(cfg.log_dir) if cfg.log_dir else None
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._latest: VlmObservation | None = None
        self._busy = False
        # -inf efectivo: el primer frame dispara siempre, sin esperar
        # un intervalo completo al arrancar la mision.
        self._last_submit = float("-inf")
        self._last_query_id: str | None = None
        self._last_query_pose: tuple[float, float, float] | None = None
        self._call_times: deque[float] = deque()   # ventana de 60 s para el RPM
        self.skipped_rpm = 0                       # consultas salteadas por cuota
        self.skipped_motion = 0                    # ... por no haberse movido
        self._warned_rpm = False
        self._warned = False
        self._stop = False
        self._t0 = time.time()

        if self._backend is not None and cfg.check_on_start:
            self.check_connection()

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    # ---------------------------------------------------------- verificacion

    def check_connection(self) -> bool:
        """Una llamada de prueba, sincrona, con una imagen chica. Imprime el
        resultado y devuelve True/False. Nunca lanza."""
        if self._backend is None:
            print("[vlm_semantic] verificacion: SIN backend (backend='off' o "
                  "credencial ausente)")
            self.connected = False
            return False

        print(f"[vlm_semantic] verificando conexion con Gemini "
              f"(modelo={self.cfg.model})...")
        gris = np.full((32, 32, 3), 128, dtype=np.uint8)
        t0 = time.time()
        try:
            jpg = _rgb_a_jpg(gris, 64, 70)
            d = self._backend.ask(jpg, "¿La imagen esta completamente vacia?")
            dt = time.time() - t0
            print(f"[vlm_semantic] conexion OK: respondio en {dt:.2f} s "
                  f"-> {d}")
            self.connected = True
            return True
        except Exception as exc:
            dt = time.time() - t0
            print(f"[vlm_semantic] conexion FALLIDA tras {dt:.2f} s: {exc}")
            print("[vlm_semantic] la mision puede correr igual, pero sin VLM: "
                  "cada tramo termina por su fail-safe de distancia.")
            self.connected = False
            return False

    # ------------------------------------------------------------------- api

    def observe(self, rgb: np.ndarray, query: VlmQuery, now: float,
                urgent: bool = False, pose=None) -> VlmObservation | None:
        """Devuelve YA la ultima respuesta a `query` (o None) y, si corresponde
        disparar, lanza una consulta nueva en el hilo de fondo. Nunca bloquea
        y nunca lanza.

        `pose` es la pose actual (cualquier objeto con .x/.y/.theta, o sea la
        `Pose` de odometry.py). Si se pasa, habilita el disparo por
        movimiento: no se vuelve a preguntar hasta que el robot haya avanzado
        `query_every_m` o girado `query_every_deg` desde la ultima consulta.
        Si es None, se cae a la cadencia por tiempo de siempre.
        """
        if not self.enabled:
            return None

        if self._debe_disparar(query, now, pose):
            with self._lock:
                self._busy = True
                self._last_submit = now
                self._last_query_id = query.id
                self._last_query_pose = _desarmar_pose(pose)
                self._call_times.append(now)
                frame = np.array(rgb, copy=True)   # el lazo reusa el buffer
            threading.Thread(target=self._trabajar, args=(frame, query),
                             daemon=True).start()

        with self._lock:
            obs = self._latest
        return obs if (obs is not None and obs.id == query.id) else None

    def _debe_disparar(self, query: VlmQuery, now: float, pose) -> bool:
        """Las cuatro compuertas, en orden de costo creciente."""
        with self._lock:
            if self._busy:
                return False                    # ya hay una consulta en vuelo
            piso = self.cfg.interval_turn_s if self._es_giro(query) else self.cfg.interval_s
            if (now - self._last_submit) < piso:
                return False                    # cadencia minima

            # --- cuota dura: ventana deslizante de 60 s ---------------------
            if self.cfg.max_rpm and self.cfg.max_rpm > 0:
                while self._call_times and (now - self._call_times[0]) >= 60.0:
                    self._call_times.popleft()
                if len(self._call_times) >= self.cfg.max_rpm:
                    self.skipped_rpm += 1
                    if not self._warned_rpm:
                        self._warned_rpm = True
                        espera = 60.0 - (now - self._call_times[0])
                        print(f"[vlm] cuota alcanzada ({self.cfg.max_rpm}/min): "
                              f"no consulto hasta dentro de {espera:.1f} s. "
                              "La mision sigue con la ultima respuesta cacheada.")
                    return False

            # --- pregunta nueva: siempre se dispara -------------------------
            # Un cambio de hito (de fase, o de tramo) es justo el momento en
            # que la respuesta cacheada ya no sirve: no tiene sentido
            # esperar a moverse medio metro para hacer la primera consulta
            # del hito nuevo.
            if query.id != self._last_query_id:
                return True

            # --- disparo por movimiento -------------------------------------
            return self._se_movio(now, pose)

    def _se_movio(self, now: float, pose) -> bool:
        """True si el robot se movio lo suficiente desde la ultima consulta
        (o si vencio la red de seguridad por tiempo). Llamar con el lock."""
        umbral_m = self.cfg.query_every_m
        umbral_rad = math.radians(self.cfg.query_every_deg)
        if umbral_m <= 0 and umbral_rad <= 0:
            return True                         # disparo por movimiento apagado

        actual = _desarmar_pose(pose)
        if actual is None or self._last_query_pose is None:
            return True                         # sin pose no podemos filtrar

        x0, y0, th0 = self._last_query_pose
        avance = math.hypot(actual[0] - x0, actual[1] - y0)
        giro = abs(_wrap(actual[2] - th0))
        if (umbral_m > 0 and avance >= umbral_m) or (umbral_rad > 0 and giro >= umbral_rad):
            return True

        # Red de seguridad: quieto pero el tiempo corre igual.
        if self.cfg.query_max_s and (now - self._last_submit) >= self.cfg.query_max_s:
            return True

        self.skipped_motion += 1
        return False

    def _es_giro(self, query: VlmQuery) -> bool:
        """Las preguntas de apertura durante un giro llevan el sufijo
        `__apertura` (lo pone SemanticMissionFSM.current_query)."""
        return query.id.endswith("__apertura")

    def _trabajar(self, rgb: np.ndarray, query: VlmQuery) -> None:
        t0 = time.time()
        obs = None
        jpg = None
        crudo = None
        try:
            jpg = _rgb_a_jpg(rgb, self.cfg.max_side_px, self.cfg.jpeg_quality)
            crudo = self._backend.ask(jpg, query.prompt)
            obs = VlmObservation(
                id=query.id,
                present=bool(crudo.get("hito_presente", False)),
                confidence=float(crudo.get("confianza", 0.0) or 0.0),
                position=str(crudo.get("posicion", "centro")),
                distance_m=_float_o_none(crudo.get("distancia_m")),
                reason=str(crudo.get("razon", ""))[:120],
                t=time.time(),
            )
            self.calls += 1
        except Exception as exc:
            self.errors += 1
            # A diferencia de antes, los errores se avisan SIEMPRE (no una sola
            # vez): si Gemini deja de contestar a mitad del recorrido queres
            # enterarte en el momento. Para no inundar, del segundo en adelante
            # se imprime en una linea corta.
            if not self._warned:
                self._warned = True
                print(f"[vlm_semantic] ERROR en la llamada a Gemini: {exc}")
                print("[vlm_semantic] mientras no conteste, los tramos terminan "
                      "por su fail-safe de distancia (segments[].on_timeout).")
            else:
                print(f"[vlm] error #{self.errors}: {str(exc)[:110]}")
        finally:
            dt = time.time() - t0
            self.last_latency_s = dt
            self.latencies.append(dt)
            with self._lock:
                if obs is not None:
                    self._latest = obs
                self._busy = False
            if obs is not None:
                self._log(obs, dt)
                self._dump(jpg, crudo, obs)

    # ------------------------------------------------------------- consola

    def _log(self, obs: VlmObservation, dt: float) -> None:
        if not self.cfg.log:
            return
        veredicto = "SI" if obs.present else "no"
        dist = f"{obs.distance_m:4.1f}m" if obs.distance_m is not None else "   - "
        print(f"[vlm] t+{time.time() - self._t0:6.1f}s  {obs.id:<14} {veredicto:>2}  "
              f"conf={obs.confidence:.2f}  pos={obs.position:<9} dist={dist}  "
              f"{dt:4.2f}s  \"{obs.reason}\"")

    def _dump(self, jpg: bytes | None, crudo: dict | None,
              obs: VlmObservation) -> None:
        """Guarda el JPG exacto que se mando y el JSON exacto que contesto."""
        if self._log_dir is None or jpg is None:
            return
        try:
            base = self._log_dir / f"{self.calls:05d}_{obs.id}"
            base.with_suffix(".jpg").write_bytes(jpg)
            base.with_suffix(".json").write_text(json.dumps({
                "id": obs.id,
                "t": obs.t,
                "latencia_s": round(self.last_latency_s, 3),
                "respuesta": crudo,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[vlm_semantic] no pude guardar el debug del VLM: {exc}")

    def close(self) -> None:
        self._stop = True

    def stats_line(self) -> str:
        if self.latencies:
            lat = sorted(self.latencies)
            media = sum(lat) / len(lat)
            p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
            tiempos = f"latencia media={media:.2f}s p95={p95:.2f}s max={lat[-1]:.2f}s"
        else:
            tiempos = "sin latencias medidas"
        conectado = {True: "si", False: "no", None: "no verificado"}[self.connected]
        return (f"modelo={self.cfg.model} conectado={conectado} "
                f"llamadas={self.calls} errores={self.errors} "
                f"salteadas(cuota)={self.skipped_rpm} "
                f"salteadas(sin moverse)={self.skipped_motion} "
                f"rpm_max={self.cfg.max_rpm} {tiempos}")


def _desarmar_pose(pose) -> tuple[float, float, float] | None:
    if pose is None:
        return None
    try:
        return (float(getattr(pose, "x")), float(getattr(pose, "y")),
                float(getattr(pose, "theta", getattr(pose, "yaw", 0.0))))
    except (AttributeError, TypeError, ValueError):
        return None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _float_o_none(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------- pruebas

def _self_test(image_path: str | None, prompt: str, solo_check: bool,
               model: str | None) -> None:
    cfg_real = VlmConfig(check_on_start=False)
    if model:
        cfg_real.model = model

    if solo_check:
        SemanticVlm(VlmConfig(model=cfg_real.model, check_on_start=True))
        return

    print("=== parseo laxo de la respuesta ===")
    for crudo in ('{"hito_presente": true, "confianza": 0.8}',
                  '```json\n{"hito_presente": false, "confianza": 0.1}\n```',
                  'Claro: {"hito_presente": true, "confianza": 0.9} espero que sirva'):
        d = _parse_json_laxo(crudo)
        print(f"  {crudo[:42]!r:48} -> {d}")
        assert "hito_presente" in d

    print("\n=== backend 'off': no llama a nadie y no rompe ===")
    vlm = SemanticVlm(VlmConfig(backend="off"))
    assert not vlm.enabled
    assert vlm.observe(np.zeros((8, 8, 3), np.uint8), VlmQuery("x", "?"), 0.0) is None

    print("\n=== config viejo (backend http + base_url) no explota ===")
    migrado = VlmConfig.from_dict({"backend": "http", "base_url": "http://x:8001/v1",
                                   "model": "gemini-2.0-flash"})
    print(f"  -> backend={migrado.backend!r} model={migrado.model!r}")
    assert migrado.backend == "gemini"

    print("\n=== cadencia: una sola llamada en vuelo ===")
    falso = SemanticVlm(VlmConfig(backend="off"))
    falso._backend = object()                 # habilitado pero sin red
    llamadas = []

    def _fake(frame, query):
        llamadas.append(query.id)
        with falso._lock:
            falso._latest = VlmObservation(query.id, True, 0.9, t=time.time())
            falso._busy = False

    falso._trabajar = _fake
    q = VlmQuery("rover", "¿se ve un rover?")
    img = np.zeros((8, 8, 3), np.uint8)
    t0 = time.time()
    falso.observe(img, q, t0)
    time.sleep(0.05)
    falso.observe(img, q, t0 + 0.1)                # dentro del intervalo: no dispara
    time.sleep(0.05)
    print(f"  llamadas disparadas en 0.1 s (interval_s={falso.cfg.interval_s}): "
          f"{len(llamadas)}")
    assert len(llamadas) == 1

    obs = falso.observe(img, q, t0 + 0.2)
    print(f"  respuesta cacheada: {obs}")
    assert obs is not None and obs.id == "piano"
    otra = falso.observe(img, VlmQuery("puerta", "¿hay una puerta?"), t0 + 0.3)
    print(f"  la misma respuesta contra OTRA pregunta: {otra}")
    assert otra is None, "una respuesta de otro hito no puede devolverse"

    print("\n=== cuota: ventana deslizante de max_rpm por minuto ===")
    cuota = SemanticVlm(VlmConfig(backend="off", max_rpm=3, interval_s=0.0,
                                  query_every_m=0.0, query_every_deg=0.0))
    cuota._backend = object()
    disparos = []
    cuota._trabajar = lambda f, q: (disparos.append(q.id),
                                    setattr(cuota, "_busy", False))
    for k in range(6):
        cuota.observe(img, VlmQuery("h", "?"), 100.0 + k * 0.1)
    print(f"  6 intentos con max_rpm=3 -> {len(disparos)} llamadas, "
          f"{cuota.skipped_rpm} salteadas")
    assert len(disparos) == 3 and cuota.skipped_rpm == 3
    cuota.observe(img, VlmQuery("h", "?"), 100.0 + 61.0)   # ya paso el minuto
    print(f"  un minuto despues -> {len(disparos)} llamadas")
    assert len(disparos) == 4, "pasado el minuto la ventana tiene que liberarse"

    print("\n=== disparo por movimiento (pose) ===")
    class _P:
        def __init__(s, x, y, th=0.0): s.x, s.y, s.theta = x, y, th
    mov = SemanticVlm(VlmConfig(backend="off", interval_s=0.0, max_rpm=0,
                                query_every_m=0.5, query_every_deg=20.0,
                                query_max_s=8.0))
    mov._backend = object()
    hechas = []
    mov._trabajar = lambda f, q: (hechas.append(q.id),
                                  setattr(mov, "_busy", False))
    q2 = VlmQuery("piano", "?")
    mov.observe(img, q2, 0.0, pose=_P(0, 0))                 # primera: siempre
    mov.observe(img, q2, 1.0, pose=_P(0.2, 0))               # 20 cm: no alcanza
    print(f"  tras 0.20 m: {len(hechas)} llamada(s)")
    assert len(hechas) == 1
    mov.observe(img, q2, 2.0, pose=_P(0.6, 0))               # 60 cm: si
    print(f"  tras 0.60 m: {len(hechas)} llamada(s)")
    assert len(hechas) == 2
    mov.observe(img, q2, 3.0, pose=_P(0.6, 0, math.radians(25)))   # giro: si
    print(f"  tras girar 25 grados sin avanzar: {len(hechas)} llamada(s)")
    assert len(hechas) == 3
    mov.observe(img, q2, 4.0, pose=_P(0.6, 0, math.radians(25)))   # quieto: no
    assert len(hechas) == 3
    mov.observe(img, q2, 12.5, pose=_P(0.6, 0, math.radians(25)))  # query_max_s
    print(f"  quieto pero pasaron 8.5 s: {len(hechas)} llamada(s)")
    assert len(hechas) == 4
    mov.observe(img, VlmQuery("otro_hito", "?"), 12.6,
                pose=_P(0.6, 0, math.radians(25)))           # hito nuevo: si
    print(f"  cambio de hito sin moverse: {len(hechas)} llamada(s)")
    assert len(hechas) == 5, "un hito nuevo tiene que preguntar de inmediato"

    if image_path:
        print(f"\n=== llamada REAL a Gemini con {image_path} ===")
        from PIL import Image
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        cfg_real.check_on_start = True
        cfg_real.log_dir = cfg_real.log_dir or "debug/vlm_selftest"
        real = SemanticVlm(cfg_real)
        if real.enabled:
            real.observe(rgb, VlmQuery("prueba", prompt), time.time())
            for _ in range(40):
                time.sleep(0.25)
                o = real.observe(rgb, VlmQuery("prueba", prompt), time.time())
                if o is not None:
                    break
            print(f"  {real.stats_line()}")
            print(f"  (jpg + json guardados en {cfg_real.log_dir}/)")

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="si se pasa, hace una llamada REAL a Gemini con esa imagen")
    ap.add_argument("--prompt", default="¿Se ve un pasillo despejado hacia adelante?")
    ap.add_argument("--model", default=None, help="pisa el modelo por default")
    ap.add_argument("--check", action="store_true",
                    help="solo verificar credencial + modelo y salir")
    a = ap.parse_args()
    _self_test(a.image, a.prompt, a.check, a.model)
