# ==============================================================================
# Test de Seguridad: ROVER_MODE en docker-compose y entrypoint.sh (Incidente O.2)
# ==============================================================================
# Verifica que ningún archivo docker-compose*.yml ni entrypoint.sh vuelva a
# configurar ROVER_MODE="full" por defecto sin revisión explícita de seguridad.
# ==============================================================================
import re
from pathlib import Path
import pytest


def get_repo_root() -> Path:
    # Este archivo está en src/er_bringup/test/ -> sube 3 niveles hasta la raíz del workspace
    p = Path(__file__).resolve().parent.parent.parent.parent
    if (p / "entrypoint.sh").exists():
        return p
    # Fallback si se ejecuta desde /root/ros2_ws adentro del contenedor
    return Path("/root/ros2_ws") if Path("/root/ros2_ws/entrypoint.sh").exists() else p


def test_docker_compose_rover_mode_manual():
    root = get_repo_root()
    compose_files = list(root.glob("docker-compose*.yml"))
    assert len(compose_files) > 0, f"No se encontraron archivos docker-compose en {root}"

    violations = []
    for cf in compose_files:
        content = cf.read_text(encoding="utf-8")
        match_full = re.search(r'ROVER_MODE:\s*["\']?full["\']?', content)
        if match_full:
            violations.append(f"{cf.name}: configurado con ROVER_MODE: 'full'")

        match_manual = re.search(r'ROVER_MODE:\s*["\']?manual["\']?', content)
        assert match_manual is not None, (
            f"{cf.name} debe definir explícitamente ROVER_MODE: 'manual' para evitar "
            f"auto-arranque de stacks de navegación y colisiones DDS."
        )

    assert not violations, (
        f"[FALLO CRÍTICO DE SEGURIDAD] Violaciones detectadas en docker-compose:\n"
        + "\n".join(violations)
        + "\nROVER_MODE DEBE ser 'manual' por defecto para evitar colisiones DDS por contenedores huérfanos."
    )


def test_entrypoint_default_rover_mode():
    root = get_repo_root()
    entrypoint_path = root / "entrypoint.sh"
    assert entrypoint_path.exists(), f"No se encontró entrypoint.sh en {root}"

    content = entrypoint_path.read_text(encoding="utf-8")
    assert 'ROVER_MODE="${ROVER_MODE:-full}"' not in content, (
        "entrypoint.sh no debe tener como default 'full'. Debe ser 'manual'."
    )
    assert 'ROVER_MODE="${ROVER_MODE:-manual}"' in content, (
        "entrypoint.sh debe tener como default 'manual' (ROVER_MODE=\"${ROVER_MODE:-manual}\")."
    )
