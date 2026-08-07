#!/usr/bin/env python3
"""
Punto de entrada principal por línea de comandos (CLI) para el planificador de trayectorias GeNIE.

CONCEPTOS CLAVE:
- Interfaz de Línea de Comandos (CLI): Este script permite a los desarrolladores y a otros 
  programas ejecutar todo el sistema de conducción (percepción, mapeo y planificación) directamente 
  desde la terminal del sistema operativo, sin necesidad de escribir código adicional.
- Sobrescritura de Parámetros (Overrides): El núcleo del sistema se basa en un archivo de 
  configuración (YAML o JSON) que dicta cómo debe comportarse el robot. Sin embargo, este script 
  está diseñado para permitir que argumentos pasados por consola (como `--image` o `--goal-x`) 
  tengan prioridad y "sobrescriban" temporalmente lo que diga el archivo. Esto es vital para 
  hacer pruebas rápidas y depuración (debugging) sin tener que editar constantemente el archivo base.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from genie_path_planner.io_utils import load_config
from genie_path_planner.pipeline import run_offline_path_planner


def parse_args() -> argparse.Namespace:
    """
    Define, configura y procesa los argumentos que el usuario puede ingresar por la terminal.

    CONCEPTOS CLAVE:
    - argparse: Librería estándar de Python para construir interfaces de terminal robustas. 
      Genera automáticamente menús de ayuda (al usar `--help`) y valida que los tipos de datos 
      sean correctos (por ejemplo, asegurando que `--goal-x` sea un número flotante).
    - Flexibilidad de Entradas: Observa que el único argumento estrictamente obligatorio (required=True) 
      es `--config`. Todo lo demás es opcional (default=None). Si el usuario omite la postura de la 
      cámara o la ruta de la imagen, el sistema intentará extraer esos datos del archivo de configuración.

    Returns:
        argparse.Namespace: Un objeto que contiene todos los parámetros capturados de la consola, 
        accesibles como atributos (ej. args.config, args.image, args.mode).
    """
    parser = argparse.ArgumentParser(
        description="Run SAM-TP projection, optional depth/observation fusion, and BEV path planning."
    )
    parser.add_argument("--config", required=True, help="Planner YAML/JSON config")
    parser.add_argument("--image", default=None, help="RGB image path. Overrides config observations when set.")
    parser.add_argument("--depth", default=None, help="Optional depth path (.npy meters or depth image)")
    parser.add_argument(
        "--score-map",
        default=None,
        help="Optional HxW .npy traversability/logits map to skip SAM-TP inference.",
    )
    parser.add_argument(
        "--score-map-type",
        choices=["traversability", "logits"],
        default=None,
        help="Interpret --score-map as traversability or raw logits.",
    )
    parser.add_argument(
        "--camera-k",
        default=None,
        help=(
            "Optional camera intrinsics .npy/.json/.yaml. If omitted, "
            "camera.intrinsics or camera.intrinsics_path from the config is used."
        ),
    )
    parser.add_argument(
        "--camera-pose",
        default=None,
        help=(
            "Optional T_world_camera .npy/.json/.yaml. If omitted, "
            "camera.pose or camera.pose_path from the config is used."
        ),
    )
    parser.add_argument(
        "--robot-pose-xy-yaw",
        default=None,
        help="Optional reference robot pose as 'x,y,yaw_rad'. Requires camera.T_base_camera in config.",
    )
    parser.add_argument("--goal-x", type=float, default=None, help="Goal x in meters (+right)")
    parser.add_argument("--goal-y", type=float, default=None, help="Goal y in meters (+forward)")
    parser.add_argument("--mode", choices=["rgb", "depth", "rgbd"], default=None, help="Planning mode override")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    return parser.parse_args()


def main() -> None:
    """
    Función principal que orquesta la inicialización y ejecución del planificador.

    FLUJO DE EJECUCIÓN (Pipeline):
    1. Parseo: Llama a `parse_args()` para capturar lo que el usuario escribió en la terminal.
    2. Carga: Utiliza `load_config` para leer el archivo YAML/JSON estructural del proyecto.
    3. Delegación: Invoca a `run_offline_path_planner` (el orquestador maestro que vimos en los 
       módulos anteriores), inyectándole tanto la configuración base como las variables sobreescritas 
       por la terminal.
    4. Notificación y Cierre: Al finalizar el procesamiento profundo, extrae del diccionario de metadatos 
       las rutas exactas de los archivos generados y las imprime (print) en consola, informando al 
       usuario dónde puede ver el mapa dibujado y dónde están las coordenadas matemáticas del camino final.
    """
    args = parse_args()
    config, config_dir = load_config(args.config)
    # config tiene las configuraciones provenientes del archivo 
    # YAML/JSON especificado en la línea de comandos
    # y config_dir es el directorio donde se encuentra ese archivo. 
    # Esto permite que el pipeline de planificación de ruta tenga 
    # acceso a todas las configuraciones necesarias para ejecutar 
    # correctamente el proceso de planificación de ruta basado en 
    # imágenes y otros datos opcionales proporcionados por el usuario.S
    # Corre el pipeline de planificación de ruta con las opciones proporcionadas desde la línea de comandos
    meta = run_offline_path_planner(
        config=config, # archivo yaml o json
        config_dir=config_dir,
        image=args.image,
        depth=args.depth,
        score_map=args.score_map,
        score_map_type=args.score_map_type,
        camera_k=args.camera_k,
        camera_pose=args.camera_pose,
        robot_pose_xy_yaw=args.robot_pose_xy_yaw,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        mode=args.mode,
        output_dir=args.output_dir,
    )
    files = meta["output_files"]
    print(f"[GENIE-PLAN] visualization: {files['visualization']}")
    print(f"[GENIE-PLAN] final path xy: {files['final_path_xy_m_npy']}")
    print(f"[GENIE-PLAN] metadata: {files['metadata_json']}")


if __name__ == "__main__":
    main()