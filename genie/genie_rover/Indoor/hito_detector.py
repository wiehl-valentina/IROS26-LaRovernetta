"""Detector de hitos por IMAGENES DE REFERENCIA (DINO), reemplazo de SemanticVlm.

Por que existe: preguntarle a Gemini "¿se ve un piano?" cuesta ~1-6 s, gasta
cuota (15 rpm) y necesita red. Este modulo contesta la MISMA pregunta en
~50 ms, sin red y sin cuota, comparando el frame contra las fotos de
referencia del hito que hay en `reference_images/<hito>/`.

DINO (facebook/dinov2-small) da un embedding de 384 numeros por imagen que
cambia poco cuando cambia la luz, el angulo o la distancia -- que es
justamente el problema de comparar fotos crudas: la misma puerta con la luz
prendida y apagada tiene pixeles totalmente distintos, pero embeddings
parecidos. La similitud coseno entre el embedding del frame y el de cada
foto de referencia da el score; se queda con el mejor.

INTERFAZ: identica a SemanticVlm (observe/enabled/stats_line/close), asi que
indoor_bridge.py no cambia -- se elige uno u otro en el config con
`vlm.backend`.

TRES MODOS (`vlm.backend` en el yaml):

    "dino"    solo imagenes de referencia. Sin red, sin cuota, ~50 ms.
    "hybrid"  DINO decide cuando esta seguro (score alto o bajo) y solo
              consulta a Gemini en la zona gris. Gasta ~70% menos cuota que
              "gemini" puro y responde al instante en la mayoria de los
              frames.
    "gemini"  el de siempre (vlm_semantic.SemanticVlm), sin DINO.

COMO SE MAPEA UN HITO A SUS FOTOS: por el `milestone.id` (o `turn.id`) del
segmento en el yaml. El id "piano" busca las fotos en
`reference_images/piano/`; si no existe, prueba con los alias declarados en
`vlm.hito_dirs` del config. Un hito sin carpeta (o con carpeta vacia) no
rompe nada: se comporta como "no hay respuesta" y el tramo termina por su
fail-safe de distancia, igual que con el VLM caido.

CADENCIA: la misma logica de disparo por movimiento que SemanticVlm
(query_every_m / query_every_deg / query_max_s), porque aunque DINO sea
barato, comparar el mismo frame dos veces desde el mismo lugar no aporta
nada.

Y, tambien como SemanticVlm, la comparacion corre en un HILO DE FONDO. No es
por latencia de red (aca no hay red) sino por presupuesto del lazo: medido,
una comparacion cuesta ~75 ms en CPU y el lazo de control tiene 200 ms para
percepcion + planificacion + control. Gastar un tercio de eso esperando al
detector haria que el rover reaccione tarde a un obstaculo. Con el hilo,
`observe()` devuelve al instante la ultima respuesta y la nueva llega al
frame siguiente.

Autoprueba (no necesita robot ni red):
    python -m genie_rover.Indoor.hito_detector --check
    python -m genie_rover.Indoor.hito_detector --image foto.jpg --hito piano
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mission import VlmObservation, VlmQuery


# ---------------------------------------------------------------- embeddings

class DinoEncoder:
    """Envuelve el modelo DINO. Se carga una sola vez y cachea los embeddings
    de las fotos de referencia (que nunca cambian durante una mision)."""

    def __init__(self, model_name: str = "facebook/dinov2-small",
                 device: str = "auto", max_side_px: int = 224):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.max_side_px = max_side_px

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: array HxWx3 uint8. Devuelve un vector unitario de 384."""
        from PIL import Image

        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
        if max(img.size) > self.max_side_px:
            escala = self.max_side_px / max(img.size)
            img = img.resize((max(1, int(img.width * escala)),
                              max(1, int(img.height * escala))))

        with self._torch.no_grad():
            entradas = self.processor(images=img, return_tensors="pt")
            entradas = {k: v.to(self.device) for k, v in entradas.items()}
            salida = self.model(**entradas)
            vec = salida.last_hidden_state[:, 0, :].cpu().numpy().ravel()

        norma = np.linalg.norm(vec)
        return (vec / norma if norma > 0 else vec).astype(np.float32)

    def encode_file(self, path: Path) -> np.ndarray:
        clave = str(path)
        if clave not in self._cache:
            from PIL import Image

            self._cache[clave] = self.encode(
                np.array(Image.open(path).convert("RGB"))
            )
        return self._cache[clave]


# ------------------------------------------------------------------- galeria

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class GaleriaHitos:
    """Las fotos de referencia en disco, agrupadas por id de hito."""

    def __init__(self, raiz: Path, alias: dict[str, str] | None = None):
        self.raiz = Path(raiz)
        self.alias = {k: str(v) for k, v in (alias or {}).items()}
        self._fotos: dict[str, list[Path]] = {}

    def fotos_de(self, hito_id: str) -> list[Path]:
        if hito_id in self._fotos:
            return self._fotos[hito_id]

        candidatos = [hito_id]
        if hito_id in self.alias:
            candidatos.insert(0, self.alias[hito_id])

        encontradas: list[Path] = []
        for nombre in candidatos:
            carpeta = self.raiz / nombre
            if carpeta.is_dir():
                encontradas = sorted(
                    p for p in carpeta.iterdir()
                    if p.suffix.lower() in EXTENSIONES
                )
                if encontradas:
                    break

        self._fotos[hito_id] = encontradas
        return encontradas

    def resumen(self) -> str:
        if not self.raiz.is_dir():
            return f"(no existe {self.raiz})"
        partes = []
        for carpeta in sorted(p for p in self.raiz.iterdir() if p.is_dir()):
            n = sum(1 for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
            partes.append(f"{carpeta.name}:{n}")
        return ", ".join(partes) if partes else "(vacia)"


# --------------------------------------------------------------------- config

@dataclass
class HitoDetectorConfig:
    backend: str = "dino"                  # "dino" | "hybrid"
    reference_dir: str = "reference_images"
    model: str = "facebook/dinov2-small"
    device: str = "auto"

    # Umbrales de decision sobre la similitud coseno [0, 1].
    # Arriba de `confirm`: el hito esta. Abajo de `reject`: no esta. En el
    # medio es zona gris (en "hybrid" es lo unico que se le pregunta a Gemini).
    confirm_threshold: float = 0.72
    reject_threshold: float = 0.58

    # Cadencia: misma semantica que en VlmConfig.
    interval_s: float = 0.4
    interval_turn_s: float = 0.2
    query_every_m: float = 0.25
    query_every_deg: float = 10.0
    query_max_s: float = 4.0

    max_side_px: int = 224
    hito_dirs: dict[str, str] = field(default_factory=dict)

    log: bool = True
    log_dir: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "HitoDetectorConfig":
        d = dict(d or {})
        conocidas = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**conocidas)


# ------------------------------------------------------------------ detector

def _desarmar_pose(pose) -> tuple[float, float, float] | None:
    if pose is None:
        return None
    try:
        return float(pose.x), float(pose.y), float(pose.theta)
    except (AttributeError, TypeError, ValueError):
        return None


class HitoDetector:
    """Contesta VlmQuery comparando el frame contra las fotos de referencia.

    Mismo contrato que SemanticVlm: `observe()` nunca bloquea mas de lo que
    tarda un forward de DINO (~50 ms), nunca lanza, y devuelve None cuando no
    hay nada nuevo que decir.
    """

    def __init__(self, cfg: HitoDetectorConfig, vlm_fallback=None):
        self.cfg = cfg
        self.vlm = vlm_fallback              # SemanticVlm o None
        self.calls = 0
        self.errors = 0
        self.grises = 0                      # cuantas veces cayo en zona gris
        self.last_latency_s = 0.0
        self.latencies: list[float] = []

        self.galeria = GaleriaHitos(Path(cfg.reference_dir), cfg.hito_dirs)

        self._encoder: DinoEncoder | None = None
        try:
            self._encoder = DinoEncoder(cfg.model, cfg.device, cfg.max_side_px)
            print(f"[hito_detector] DINO listo ({cfg.model}, {self._encoder.device}). "
                  f"Fotos de referencia en {cfg.reference_dir}: {self.galeria.resumen()}")
        except Exception as exc:
            print(f"[hito_detector] ERROR: no pude cargar DINO ({exc}).")
            print("[hito_detector] sigo SIN deteccion por imagen: cada tramo va a "
                  "terminar por su fail-safe de distancia.")

        self._lock = threading.Lock()
        self._busy = False
        self._ultima: VlmObservation | None = None
        self._ultimo_disparo = 0.0
        self._ultima_query_id: str | None = None
        self._ultima_pose: tuple[float, float, float] | None = None

        self._log_dir = Path(cfg.log_dir) if cfg.log_dir else None
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- interfaz

    @property
    def enabled(self) -> bool:
        return self._encoder is not None

    def observe(self, rgb: np.ndarray, query: VlmQuery, now: float,
                urgent: bool = False, pose=None) -> VlmObservation | None:
        """Devuelve YA la ultima respuesta a `query` (o None) y, si corresponde,
        lanza la comparacion nueva en el hilo de fondo. Nunca bloquea.

        El hilo no es por latencia de red (DINO corre local) sino por
        presupuesto del lazo: una comparacion cuesta ~75 ms en CPU, y el lazo
        de control tiene 200 ms para percepcion + planificacion + control.
        Gastar un tercio de eso esperando al detector haria que el rover
        reaccione tarde a un obstaculo. Con el hilo, la respuesta llega al
        frame siguiente -- 200 ms mas tarde, irrelevante para decidir si se
        vio un piano.
        """
        if not self.enabled:
            return None

        if self._debe_disparar(query, now, pose, urgent):
            fotos = self.galeria.fotos_de(query.id)
            # Hito sin fotos: que lo cierre el fail-safe de distancia, no yo.
            if fotos:
                with self._lock:
                    self._busy = True
                    self._ultimo_disparo = now
                    self._ultima_query_id = query.id
                    self._ultima_pose = _desarmar_pose(pose)
                    frame = np.array(rgb, copy=True)  # el lazo reusa el buffer
                threading.Thread(target=self._trabajar,
                                 args=(frame, query, fotos, now),
                                 daemon=True).start()

        return self._cacheada(query)

    def _trabajar(self, rgb: np.ndarray, query: VlmQuery,
                  fotos: list[Path], now: float) -> None:
        """Corre en el hilo de fondo. No lanza: todo error queda en un aviso."""
        t0 = time.time()
        try:
            obs = self._comparar(rgb, query, fotos, now)
        except Exception as exc:
            self.errors += 1
            print(f"[hito_detector] error comparando '{query.id}': {exc}")
            return
        finally:
            with self._lock:
                self._busy = False

        latencia = time.time() - t0
        self.last_latency_s = latencia
        self.latencies.append(latencia)
        self.calls += 1

        if self.cfg.log:
            print(f"[hito] t+{now:7.1f}s  {query.id:<16} "
                  f"{'SI' if obs.present else 'no'}  conf={obs.confidence:.2f}  "
                  f"{latencia * 1000:4.0f}ms  {obs.reason}")

        with self._lock:
            self._ultima = obs

    def stats_line(self) -> str:
        if not self.enabled:
            return "deshabilitado (DINO no cargo)"
        p50 = float(np.median(self.latencies)) if self.latencies else 0.0
        extra = f", {self.grises} en zona gris" if self.cfg.backend == "hybrid" else ""
        return (f"{self.calls} comparaciones, {self.errors} errores, "
                f"mediana {p50 * 1000:.0f} ms{extra}")

    def close(self) -> None:
        if self.vlm is not None:
            self.vlm.close()

    # -------------------------------------------------------------- interno

    def _cacheada(self, query: VlmQuery) -> VlmObservation | None:
        with self._lock:
            obs = self._ultima
        return obs if (obs is not None and obs.id == query.id) else None

    def _debe_disparar(self, query: VlmQuery, now: float, pose, urgent: bool) -> bool:
        with self._lock:
            if self._busy:
                return False            # ya hay una comparacion en vuelo

            # Cambio de hito: la respuesta cacheada ya no sirve para nada.
            if query.id != self._ultima_query_id:
                return True

            piso = self.cfg.interval_turn_s if urgent else self.cfg.interval_s
            if now - self._ultimo_disparo < piso:
                return False

            if self.cfg.query_max_s and \
                    now - self._ultimo_disparo >= self.cfg.query_max_s:
                return True

            actual = _desarmar_pose(pose)
            if actual is None or self._ultima_pose is None:
                return True

            avance = math.hypot(actual[0] - self._ultima_pose[0],
                                actual[1] - self._ultima_pose[1])
            giro = abs(math.degrees(_wrap(actual[2] - self._ultima_pose[2])))

        return bool((self.cfg.query_every_m and avance >= self.cfg.query_every_m)
                    or (self.cfg.query_every_deg and giro >= self.cfg.query_every_deg))

    def _comparar(self, rgb: np.ndarray, query: VlmQuery,
                  fotos: list[Path], now: float) -> VlmObservation:
        emb = self._encoder.encode(rgb)
        scores = [float(np.dot(emb, self._encoder.encode_file(p))) for p in fotos]
        mejor = max(scores)
        cual = fotos[int(np.argmax(scores))].name

        if mejor >= self.cfg.confirm_threshold:
            obs = VlmObservation(
                id=query.id, present=True, confidence=_a_confianza(
                    mejor, self.cfg.confirm_threshold, alto=True),
                position="centro", distance_m=None,
                reason=f"DINO {mejor:.2f} vs {cual}", t=now,
            )
        elif mejor <= self.cfg.reject_threshold:
            obs = VlmObservation(
                id=query.id, present=False, confidence=_a_confianza(
                    mejor, self.cfg.reject_threshold, alto=False),
                position="centro", distance_m=None,
                reason=f"DINO {mejor:.2f} (bajo umbral)", t=now,
            )
        else:
            obs = self._zona_gris(rgb, query, mejor, cual, now)

        self._guardar_log(query, mejor, cual, scores, fotos, obs)
        return obs

    def _zona_gris(self, rgb: np.ndarray, query: VlmQuery, mejor: float,
                   cual: str, now: float) -> VlmObservation:
        """Score ambiguo: en 'hybrid' desempata Gemini; en 'dino' no confirma."""
        self.grises += 1

        if self.cfg.backend == "hybrid" and self.vlm is not None and self.vlm.enabled:
            del rgb  # el fallback dispara su propia consulta con su cadencia
            respuesta = self.vlm._cacheada(query) if hasattr(self.vlm, "_cacheada") else None
            if respuesta is not None:
                combinada = 0.5 * _a_confianza(mejor, self.cfg.confirm_threshold,
                                               alto=True) + 0.5 * respuesta.confidence
                return VlmObservation(
                    id=query.id, present=respuesta.present, confidence=combinada,
                    position=respuesta.position, distance_m=respuesta.distance_m,
                    reason=f"DINO {mejor:.2f} + VLM {respuesta.confidence:.2f}", t=now,
                )

        # Sin desempate: no confirmar es lo seguro (el tramo sigue recto).
        return VlmObservation(
            id=query.id, present=False,
            confidence=_a_confianza(mejor, self.cfg.confirm_threshold, alto=True),
            position="centro", distance_m=None,
            reason=f"DINO {mejor:.2f} (zona gris vs {cual})", t=now,
        )

    def _guardar_log(self, query: VlmQuery, mejor: float, cual: str,
                     scores: list[float], fotos: list[Path],
                     obs: VlmObservation) -> None:
        if self._log_dir is None:
            return
        destino = self._log_dir / f"{self.calls:05d}_{query.id}.json"
        try:
            destino.write_text(json.dumps({
                "hito": query.id,
                "mejor_score": mejor,
                "mejor_foto": cual,
                "scores": {p.name: s for p, s in zip(fotos, scores)},
                "present": obs.present,
                "confidence": obs.confidence,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def _wrap(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def _a_confianza(score: float, umbral: float, alto: bool) -> float:
    """Reescala la similitud coseno a una confianza [0, 1] comparable con la
    que devuelve el VLM, para que `milestone.min_confidence` del yaml
    signifique lo mismo con los dos backends."""
    if alto:
        return float(np.clip((score - umbral) / max(1e-6, 1.0 - umbral) * 0.5 + 0.5,
                             0.0, 1.0))
    return float(np.clip((umbral - score) / max(1e-6, umbral) * 0.5 + 0.5, 0.0, 1.0))


# ------------------------------------------------------------------ fabrica

def build_detector(cfg_vlm: dict):
    """Devuelve el detector que pide `vlm.backend` del config.

    "dino"/"hybrid" -> HitoDetector (con SemanticVlm adentro si es hybrid)
    cualquier otro  -> SemanticVlm, el de siempre
    """
    backend = str((cfg_vlm or {}).get("backend", "gemini")).lower()

    if backend not in ("dino", "hybrid"):
        from .vlm_semantic import SemanticVlm, VlmConfig
        return SemanticVlm(VlmConfig.from_dict(cfg_vlm))

    cfg = HitoDetectorConfig.from_dict(cfg_vlm)
    cfg.backend = backend

    fallback = None
    if backend == "hybrid":
        from .vlm_semantic import SemanticVlm, VlmConfig
        vcfg = VlmConfig.from_dict({**cfg_vlm, "backend": "gemini"})
        fallback = SemanticVlm(vcfg)

    return HitoDetector(cfg, vlm_fallback=fallback)


# ----------------------------------------------------------------- autoprueba

def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference-dir", default="reference_images")
    ap.add_argument("--image", help="frame a evaluar")
    ap.add_argument("--hito", help="id del hito contra el que comparar")
    ap.add_argument("--check", action="store_true",
                    help="solo listar que fotos de referencia hay")
    args = ap.parse_args()

    galeria = GaleriaHitos(Path(args.reference_dir))
    print(f"carpeta: {args.reference_dir}")
    print(f"hitos:   {galeria.resumen()}")

    if args.check or not args.image:
        return

    if not args.hito:
        ap.error("--image necesita --hito")

    from PIL import Image

    det = HitoDetector(HitoDetectorConfig(reference_dir=args.reference_dir))
    if not det.enabled:
        return

    rgb = np.array(Image.open(args.image).convert("RGB"))
    obs = det.observe(rgb, VlmQuery(args.hito, ""), now=0.0)
    print(f"\nresultado: {obs}")


if __name__ == "__main__":
    _main()
