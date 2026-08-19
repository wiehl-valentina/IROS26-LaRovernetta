"""Helper para que los scripts y notebooks de ML_model puedan importar
paquetes del repo de navegacion (genie/, traversability/) sin vendorear una
copia acá. Sigue la decision de mantener los repos separados por ahora.

DOS formas de resolverlo, elegí una:

Opcion A (recomendada) -- instalar como paquete editable, una vez:

    pip install --no-build-isolation -e /ruta/a/tu/repo/de/navegacion/genie
    pip install -e '/ruta/a/tu/repo/de/navegacion/traversability[hf]'

Con esto `import sam2` y `from rover_traversability import ...` funcionan en
cualquier lado de este repo sin este helper. Si ya hiciste esto, no
necesitas usar config_paths.py para nada.

Opcion B -- sin instalar, para notebooks sueltos o pruebas rapidas:

    1. cp .env.example .env
    2. editá .env con la ruta real: NAV_REPO_PATH=/ruta/a/tu/repo/de/navegacion
    3. en el notebook/script, ANTES de importar sam2/rover_traversability:

        import config_paths  # agrega el repo al sys.path

Este archivo asume que el repo de navegacion tiene, en su raiz, las
carpetas `genie/` (contiene el paquete `sam2`) y `traversability/` (contiene
el paquete `rover_traversability`) -- que es como esta hoy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _cargar_env(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def configurar(nav_repo_path: str | None = None) -> bool:
    """Agrega genie/ y traversability/ del repo de navegacion al sys.path.

    Devuelve True si encontro y agrego algo, False si no pudo (y avisa por
    que). Se puede llamar manualmente con una ruta explicita, o dejar que
    lea NAV_REPO_PATH de .env / variable de entorno.
    """
    here = Path(__file__).resolve().parent
    env = {**_cargar_env(here / ".env"), **os.environ}

    nav_repo = nav_repo_path or env.get("NAV_REPO_PATH")
    if not nav_repo:
        print(
            "[config_paths] AVISO: NAV_REPO_PATH no esta seteado (.env o "
            "variable de entorno). No vas a poder importar "
            "rover_traversability/sam2 hasta que lo configures -- ver "
            ".env.example, o instalá los paquetes con pip -e (opcion A del "
            "docstring de este archivo)."
        )
        return False

    nav_path = Path(nav_repo).expanduser().resolve()
    if not nav_path.exists():
        print(f"[config_paths] AVISO: NAV_REPO_PATH apunta a {nav_path}, que no existe.")
        return False

    agregado = False
    for sub in ("traversability", "genie"):
        p = nav_path / sub
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            agregado = True
        elif not p.exists():
            print(f"[config_paths] AVISO: no encontre {p} (¿la ruta es correcta?)")

    if agregado:
        print(f"[config_paths] repo de navegacion encontrado en {nav_path}, agregado al path")
    return agregado


# Se ejecuta automaticamente al hacer `import config_paths`.
configurar()
