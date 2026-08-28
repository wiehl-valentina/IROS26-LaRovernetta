"""Recuperacion con VLM: que direccion tiene mejor pinta cuando el rover se
atasca, en vez de girar a ciegas.

Replica en un unico prompt las dos etapas del paper de GeNIE (clasificar
on-road/off-road, despues elegir rumbo), para no pagar la latencia de 4
llamadas separadas por cada barrido de 360 grados.

Deliberadamente NO manda nada al SDK del rover: solo mira una imagen y
devuelve una decision. Quien la ejecuta es genie_rover.bridge._recover().
Si esto fallara (sin red, API caida, timeout) el bridge tiene que poder
seguir funcionando con el barrido ciego de siempre — por eso esta funcion
nunca levanta una excepcion hacia afuera, devuelve None.

Autoprueba (necesita GEMINI_API_KEY, no necesita rover):
    python -m genie_rover.vlm_recovery --image screenshots/imagen.jpg
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from programs.client.genai_client import (
    GenaiCallError,
    GenaiCredentialsError,
    ask_image_structured,
)
from dotenv import load_dotenv

ENV_PATH = PROJECT_ROOT / "earth-rovers-sdk" / ".env"
load_dotenv(ENV_PATH)
Heading = Literal["izquierda", "derecha", "adelante", "atras"]

RECOVERY_PROMPT = """Sos el modulo de recuperacion de un robot terrestre (rover) que \
acaba de quedarse sin camino transitable segun su planificador geometrico. \
Mira la imagen de la camara frontal del robot y decidi hacia donde conviene \
girar para encontrar terreno despejado.

Reglas:
- "on_road" = true si el robot esta actualmente sobre una superficie \
transitable (vereda, camino, pasto corto, piso liso). false si esta sobre \
o mirando hacia zanjas, escaleras, agua, vegetacion densa, o cualquier \
obstaculo claro.
- "heading" es la direccion mas prometedora vista DESDE la imagen actual: \
"izquierda", "derecha", "adelante" (si el frente en realidad esta despejado \
y el atasco fue una falsa alarma), o "atras" (si las tres direcciones \
delanteras se ven mal y conviene retroceder).
- "confianza" de 0.0 a 1.0: que tan seguro estas de la eleccion. Poné un \
numero bajo si la imagen es ambigua, esta borrosa, o no alcanza a ver \
terreno claramente transitable en ninguna direccion.

Respondé solo con el JSON pedido."""


class _GeminiRecoveryResponse(BaseModel):
    on_road: bool = Field(description="si el robot esta hoy sobre terreno transitable")
    heading: Heading = Field(description="direccion recomendada para retomar la marcha")
    confianza: float = Field(ge=0.0, le=1.0, description="confianza de la eleccion, 0 a 1")
    razon: str = Field(description="una oracion corta explicando la eleccion, para logging")


@dataclass
class RecoveryDecision:
    heading: Heading
    on_road: bool
    confidence: float
    reason: str


def _frame_to_jpg_bytes(rgb: np.ndarray, quality: int = 82, max_side: int = 768) -> bytes:
    """Reduce el frame antes de mandarlo: menos latencia y menos costo por
    llamada, y para esta decision (4 direcciones posibles) no hace falta
    resolucion completa."""
    img = Image.fromarray(rgb)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def ask_recovery_heading(
    rgb: np.ndarray,
    min_confidence: float = 0.35,
    timeout_s: float = 4.0,
) -> RecoveryDecision | None:
    """Le pregunta a Gemini hacia donde girar. Devuelve None (nunca levanta
    excepcion) si la llamada falla, tarda de mas, o la confianza reportada
    es menor a min_confidence — en todos esos casos el bridge debe caer al
    barrido ciego en vez de confiar en una respuesta dudosa.
    """
    try:
        jpg = _frame_to_jpg_bytes(rgb)
        resp = ask_image_structured(
            jpg, RECOVERY_PROMPT, _GeminiRecoveryResponse, timeout_s=timeout_s
        )
    except (GenaiCallError, GenaiCredentialsError) as exc:
        print(f"[vlm_recovery] sin respuesta util de Gemini: {exc}")
        return None
    except Exception as exc:  # nunca dejar que un fallo aca tumbe el bridge
        print(f"[vlm_recovery] error inesperado, sigo con el barrido ciego: {exc}")
        return None

    if resp.confianza < min_confidence:
        print(f"[vlm_recovery] confianza baja ({resp.confianza:.2f} < {min_confidence}), "
              f"descarto la sugerencia: {resp.razon}")
        return None

    return RecoveryDecision(
        heading=resp.heading, on_road=resp.on_road,
        confidence=resp.confianza, reason=resp.razon,
    )


# --------------------------------------------------------------------- self test

def _self_test(image_path: str) -> None:
    from programs.client.genai_client import load_credentials

    load_credentials()
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    d = ask_recovery_heading(rgb)
    if d is None:
        print("Sin decision utilizable (fallo, timeout, o confianza baja).")
    else:
        print(f"heading={d.heading}  on_road={d.on_road}  "
              f"confianza={d.confidence:.2f}\n razon: {d.reason}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    _self_test(a.image)
