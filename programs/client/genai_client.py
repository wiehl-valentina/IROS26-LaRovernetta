"""Cliente de bajo nivel para Gemini (google-genai / Interactions API).

Separado a proposito de cualquier logica de navegacion: esta modulo SOLO
sabe hablar con Gemini y devolver texto/JSON. No conoce el rover, no manda
comandos a nada. Eso es responsabilidad de quien lo llame (por ejemplo
genie_rover.vlm_recovery).

Requiere:
    pip install google-genai python-dotenv

Variables de entorno esperadas (en .env o exportadas):
    GEMINI_API_KEY=...

Autoprueba (necesita GEMINI_API_KEY valido, no necesita rover):
    python -m programs.client.genai_client --image screenshots/imagen.jpg
"""

from __future__ import annotations

import base64
import concurrent.futures
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Modelo por defecto: el mas rapido/barato de la generacion actual, pensado
# para automatizacion de alto volumen (recomendado por Google como sucesor
# directo de 2.5 Flash, que ya esta deprecado). Si en tu cuenta preferis mas
# calidad a costa de latencia, cambialo a "gemini-3.5-flash" o
# "gemini-3.6-flash" via el parametro model= o la env var GENAI_MODEL.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
 
_client = None  # singleton, se crea recien en load_credentials()
 
 
class GenaiCredentialsError(RuntimeError):
    pass
 
 
class GenaiCallError(RuntimeError):
    pass
 
 
def load_credentials(env_path: str | None = None) -> None:
    """Carga GEMINI_API_KEY desde .env (si existe) y crea el cliente.
 
    Llamar UNA vez al arrancar el proceso (bridge.py, demo, lo que sea).
    Idempotente: si ya hay un cliente creado, no hace nada.
    """
    global _client
    if _client is not None:
        return
 
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)  # no falla si el archivo no existe
    except ImportError:
        pass  # python-dotenv es opcional si ya exportaste la env var a mano
 
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GenaiCredentialsError(
            "No encuentro GEMINI_API_KEY. Poné GEMINI_API_KEY=tu_clave en un "
            "archivo .env en la raiz del proyecto, o exportala en la terminal "
            "antes de correr el bridge."
        )
 
    from google import genai
    _client = genai.Client(api_key=api_key)
    print("[genai_client] credenciales cargadas, cliente listo")
 
 
def _get_client():
    if _client is None:
        raise GenaiCredentialsError(
            "genai_client no esta inicializado. Llama a load_credentials() "
            "antes de pedir nada."
        )
    return _client
 
 
def _image_bytes_to_part(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    return {
        "type": "image",
        "data": base64.b64encode(image_bytes).decode("utf-8"),
        "mime_type": mime_type,
    }
 
 
def ask_image_structured(
    image_bytes: bytes,
    prompt: str,
    schema: type[T],
    model: str | None = None,
    timeout_s: float = 4.0,
    mime_type: str = "image/jpeg",
) -> T:
    """Le manda una imagen + prompt a Gemini y devuelve la respuesta validada
    contra un modelo Pydantic (structured output: Gemini se obliga a devolver
    JSON con esa forma exacta, no hay que parsear texto libre).
 
    Lanza GenaiCallError si la llamada falla, tarda mas de timeout_s, o la
    respuesta no valida contra el schema. El llamador decide el fallback.
    """
    client = _get_client()
    mdl = model or os.environ.get("GENAI_MODEL", DEFAULT_MODEL)
 
    def _call() -> T:
        interaction = client.interactions.create(
            model=mdl,
            input=[
                {"type": "text", "text": prompt},
                _image_bytes_to_part(image_bytes, mime_type),
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            },
            timeout=timeout_s,  # limite de la propia llamada HTTP del SDK, en segundos
        )
        return schema.model_validate_json(interaction.output_text)
 
    # OJO: future.result(timeout=...) corta la ESPERA de este hilo, pero no
    # mata el hilo de fondo si Gemini nunca contesta. Por eso el timeout de
    # arriba (en la llamada del SDK) es el que realmente corta la conexion;
    # este es un backstop. Y por eso NO usamos "with ThreadPoolExecutor()":
    # su __exit__ espera a que el hilo termine, lo que anularia el timeout
    # entero si el hilo queda colgado. shutdown(wait=False) evita eso.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        return future.result(timeout=timeout_s + 1.0)  # margen sobre el timeout del SDK
    except concurrent.futures.TimeoutError as exc:
        raise GenaiCallError(f"Gemini tardo mas de {timeout_s:.1f}s, corto") from exc
    except Exception as exc:
        raise GenaiCallError(f"fallo la llamada a Gemini: {exc}") from exc
    finally:
        executor.shutdown(wait=False)
 
 
# --------------------------------------------------------------------- self test
 
def _self_test(image_path: str) -> None:
    from pydantic import Field
 
    class Descripcion(BaseModel):
        que_se_ve: str = Field(description="una oracion describiendo la imagen")
 
    load_credentials()
    data = Path(image_path).read_bytes()
    out = ask_image_structured(
        data, "Describi brevemente que se ve en esta imagen.", Descripcion, timeout_s=10.0
    )
    print(f"Gemini dice: {out.que_se_ve}")
 
 
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    _self_test(a.image)
 