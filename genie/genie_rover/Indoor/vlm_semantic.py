"""Evaluador semantico de alto nivel para la mision indoor por tramos.

Que hace, en una linea: mira el frame de la camara frontal y contesta UNA
pregunta de si/no sobre lo que se ve ("¿se ve un piano?", "¿hay una puerta a
la izquierda?"), con una confianza. No comanda motores, no elige rumbo, no
toca el planner: lo unico que hace es devolver un `VlmObservation` que
`SemanticMissionFSM` usa para decidir si el tramo actual termino.

Tres cosas que este modulo garantiza, y que son la razon de que exista en
vez de llamar al VLM directo desde `_step()`:

1. **Nunca bloquea el lazo de control.** El lazo corre a ~5 Hz; una llamada
   a un VLM tarda cientos de milisegundos o mas. La inferencia vive en un
   hilo aparte y `observe()` devuelve SIEMPRE al instante: o la ultima
   respuesta cacheada, o None. Si devuelve None el tramo sigue recto, que es
   el comportamiento seguro.
2. **Nunca voltea la mision.** Todo error (sin red, servidor caido, JSON
   invalido, timeout) se traga con un aviso por consola UNA sola vez y se
   comporta como "no hay respuesta".
3. **Nunca mezcla preguntas.** Cada respuesta queda etiquetada con el id de
   la pregunta que la origino. Si la FSM ya cambio de fase mientras la
   llamada estaba en vuelo, la respuesta vieja se descarta sola (la FSM
   compara el id) en vez de disparar un cambio de fase equivocado.

Backends
--------
`vlm.backend: "http"` (default) habla con cualquier servidor
OpenAI-compatible: es lo que levanta vLLM sirviendo Qwen2-VL. Va aparte del
proceso del bridge a proposito -- SAM-TP ya tiene la GPU, y un 2B cargado en
el mismo proceso le pelea la VRAM justo cuando el planner la necesita:

    pip install vllm
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2-VL-2B-Instruct \\
        --port 8001 --max-model-len 4096 --limit-mm-per-prompt image=1

`vlm.backend: "gemini"` reusa `programs.client.genai_client`, el mismo
cliente que ya usa `vlm_recovery.py` en el resto del repo (necesita
GEMINI_API_KEY). `vlm.backend: "off"` lo apaga: la mision corre igual y cada
tramo termina por su fail-safe de distancia (`on_timeout`), util para
probar el recorrido sin VLM.

Autoprueba (no necesita robot ni GPU; con --image y un servidor levantado
hace una llamada de verdad):
    python -m genie_rover.Indoor.vlm_semantic
    python -m genie_rover.Indoor.vlm_semantic --image debug/run1/00007_rgb.jpg \\
        --prompt "¿se ve un piano?"
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from dataclasses import dataclass

import numpy as np

from .mission import VlmObservation, VlmQuery

INSTRUCCION = (
    "Sos el evaluador visual de un robot que avanza por un pasillo "
    "universitario. Mira la imagen de su camara frontal y responde la "
    "pregunta. Respondé UNICAMENTE con un objeto JSON, sin texto alrededor "
    "y sin ```:\n"
    '{"hito_presente": true|false, "confianza": 0.0-1.0, '
    '"posicion": "izquierda"|"centro"|"derecha", '
    '"distancia_m": numero o null, "razon": "una frase corta"}\n'
    "Poné hito_presente en false si no estás seguro: una respuesta "
    "afirmativa hace que el robot cambie de rumbo.\n\nPregunta: "
)


@dataclass
class VlmConfig:
    backend: str = "http"            # "http" | "gemini" | "off"
    base_url: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen2-VL-2B-Instruct"
    api_key: str = "EMPTY"
    interval_s: float = 0.7          # cadencia normal (~1.4 Hz)
    interval_turn_s: float = 0.4     # durante un giro, donde la latencia importa
    timeout_s: float = 6.0
    max_side_px: int = 640           # se reescala antes de mandar
    jpeg_quality: int = 80
    max_tokens: int = 128
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "VlmConfig":
        d = d or {}
        conocidas = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**conocidas)


# ------------------------------------------------------------------ backends

class _HttpBackend:
    """Servidor OpenAI-compatible (vLLM sirviendo Qwen2-VL, entre otros)."""

    def __init__(self, cfg: VlmConfig):
        self.cfg = cfg

    def ask(self, jpg: bytes, prompt: str) -> dict:
        import urllib.request

        b64 = base64.b64encode(jpg).decode("ascii")
        payload = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": INSTRUCCION + prompt},
                ],
            }],
        }
        req = urllib.request.Request(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return _parse_json_laxo(data["choices"][0]["message"]["content"])


class _GeminiBackend:
    """El mismo cliente que ya usa vlm_recovery.py en el resto del repo."""

    def __init__(self, cfg: VlmConfig):
        self.cfg = cfg
        from pydantic import BaseModel, Field

        class _Hito(BaseModel):
            hito_presente: bool = Field(description="true si el hito esta en la imagen")
            confianza: float = Field(description="0 a 1")
            posicion: str = Field(description="izquierda, centro o derecha")
            razon: str = Field(description="una frase corta")

        self._modelo = _Hito
        from programs.client import genai_client
        self._genai = genai_client
        self._genai.load_credentials()

    def ask(self, jpg: bytes, prompt: str) -> dict:
        out = self._genai.ask_image_structured(
            jpg, INSTRUCCION + prompt, self._modelo, timeout_s=self.cfg.timeout_s)
        return {"hito_presente": bool(out.hito_presente),
                "confianza": float(out.confianza),
                "posicion": str(out.posicion), "razon": str(out.razon)}


def _parse_json_laxo(texto: str) -> dict:
    """El modelo a veces envuelve el JSON en ``` o le pone texto alrededor,
    por mas que el prompt lo prohiba. En vez de fallar, recortamos del
    primer '{' al ultimo '}'."""
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


# ------------------------------------------------------------------- poller

class SemanticVlm:
    """Un hilo, una pregunta en vuelo a la vez, una respuesta cacheada."""

    def __init__(self, cfg: VlmConfig):
        self.cfg = cfg
        self.calls = 0
        self.errors = 0
        self.last_latency_s = 0.0

        self._backend = None
        if cfg.backend == "http":
            self._backend = _HttpBackend(cfg)
        elif cfg.backend == "gemini":
            self._backend = _GeminiBackend(cfg)
        elif cfg.backend != "off":
            raise ValueError(f"vlm.backend desconocido: {cfg.backend!r} "
                             "(http | gemini | off)")

        self._lock = threading.Lock()
        self._latest: VlmObservation | None = None
        self._busy = False
        # -inf efectivo: el primer frame dispara siempre, sin esperar
        # un intervalo completo al arrancar la mision.
        self._last_submit = float("-inf")
        self._warned = False
        self._stop = False

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    def observe(self, rgb: np.ndarray, query: VlmQuery, now: float,
                urgent: bool = False) -> VlmObservation | None:
        """Devuelve YA la ultima respuesta a `query` (o None) y, si toca por
        cadencia y no hay otra llamada en vuelo, dispara una nueva en el hilo
        de fondo. Nunca bloquea y nunca lanza."""
        if not self.enabled:
            return None

        intervalo = self.cfg.interval_turn_s if urgent else self.cfg.interval_s
        with self._lock:
            listo = (not self._busy) and (now - self._last_submit) >= intervalo
            if listo:
                self._busy = True
                self._last_submit = now
                frame = np.array(rgb, copy=True)   # el lazo reusa el buffer
        if listo:
            threading.Thread(target=self._trabajar, args=(frame, query),
                             daemon=True).start()

        with self._lock:
            obs = self._latest
        return obs if (obs is not None and obs.id == query.id) else None

    def _trabajar(self, rgb: np.ndarray, query: VlmQuery) -> None:
        t0 = time.time()
        obs = None
        try:
            jpg = _rgb_a_jpg(rgb, self.cfg.max_side_px, self.cfg.jpeg_quality)
            d = self._backend.ask(jpg, query.prompt)
            obs = VlmObservation(
                id=query.id,
                present=bool(d.get("hito_presente", d.get("present", False))),
                confidence=float(d.get("confianza", d.get("confidence", 0.0)) or 0.0),
                position=str(d.get("posicion", d.get("position", "centro"))),
                distance_m=_float_o_none(d.get("distancia_m", d.get("distance_m"))),
                reason=str(d.get("razon", d.get("reason", "")))[:120],
                t=time.time(),
            )
            self.calls += 1
        except Exception as exc:
            self.errors += 1
            if not self._warned:
                self._warned = True
                print(f"[vlm_semantic] el VLM no responde ({type(exc).__name__}: {exc}). "
                      "Los tramos van a terminar por su fail-safe de distancia "
                      "(mission.segments[].on_timeout). No vuelvo a avisar.")
        finally:
            self.last_latency_s = time.time() - t0
            with self._lock:
                if obs is not None:
                    self._latest = obs
                self._busy = False

    def close(self) -> None:
        self._stop = True

    def stats_line(self) -> str:
        return (f"llamadas={self.calls} errores={self.errors} "
                f"ultima_latencia={self.last_latency_s:.2f}s "
                f"backend={self.cfg.backend}")


def _float_o_none(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------- pruebas

def _self_test(image_path: str | None, prompt: str) -> None:
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
    q = VlmQuery("piano", "¿se ve un piano?")
    img = np.zeros((8, 8, 3), np.uint8)
    t0 = time.time()
    falso.observe(img, q, t0)
    time.sleep(0.05)
    falso.observe(img, q, t0 + 0.1)                # dentro del intervalo: no dispara
    time.sleep(0.05)
    print(f"  llamadas disparadas en 0.1 s (interval_s=0.7): {len(llamadas)}")
    assert len(llamadas) == 1

    obs = falso.observe(img, q, t0 + 0.2)
    print(f"  respuesta cacheada: {obs}")
    assert obs is not None and obs.id == "piano"
    otra = falso.observe(img, VlmQuery("puerta", "¿hay una puerta?"), t0 + 0.3)
    print(f"  la misma respuesta contra OTRA pregunta: {otra}")
    assert otra is None, "una respuesta de otro hito no puede devolverse"

    if image_path:
        print(f"\n=== llamada real al backend http con {image_path} ===")
        from PIL import Image
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        real = SemanticVlm(VlmConfig(backend="http"))
        real.observe(rgb, VlmQuery("prueba", prompt), time.time())
        for _ in range(40):
            time.sleep(0.25)
            o = real.observe(rgb, VlmQuery("prueba", prompt), time.time())
            if o is not None:
                print(f"  {o}")
                break
        else:
            print("  sin respuesta (¿esta levantado el servidor en "
                  f"{real.cfg.base_url}?)")
        print(f"  {real.stats_line()}")

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="si se pasa, hace una llamada REAL al backend http")
    ap.add_argument("--prompt", default="¿Se ve un pasillo despejado hacia adelante?")
    a = ap.parse_args()
    _self_test(a.image, a.prompt)
