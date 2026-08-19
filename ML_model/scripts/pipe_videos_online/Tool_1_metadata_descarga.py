from pathlib import Path
from huggingface_hub import list_repo_files, snapshot_download
import pandas as pd


def descargar_metadatos_frodobots(
    repo_id: str = "BitRobot/berkeley-frodobots-lerobot-7k",
    local_dir: str = "./frodobots_metadata",
):
  """Lista los archivos de metadatos disponibles en el repositorio y descarga

  únicamente los archivos ligeros (configs, jsons, parquets) evitando videos.
  """
  print("1. Listando archivos del repositorio...")
  files = list_repo_files(repo_id=repo_id, repo_type="dataset")

  # Filtramos solo los que pertenecen a la carpeta meta/ o son parquets de control
  metadata_files = [f for f in files if "meta/" in f or f.endswith(".parquet")]

  for f in metadata_files:
    print(f)

  print(
      f"\nTotal de archivos de metadatos/configuración encontrados:"
      f" {len(metadata_files)}"
  )

  print(f"\n2. Descargando metadatos de {repo_id}...")
  snapshot_download(
      repo_id=repo_id,
      repo_type="dataset",
      allow_patterns=["meta/*", "*.json", "*.parquet"],
      ignore_patterns=["videos/*", "chunk-*/*"],
      local_dir=local_dir,
  )
  print(f"¡Metadatos descargados con éxito en la carpeta '{local_dir}'!")


# ==========================================
# Ejemplo de uso:
# ==========================================
if __name__ == "__main__":
  # Paso 1: Descargar metadatos (descomenta si aún no los descargaste)
  # descargar_metadatos_frodobots()

  # Paso 2: Explorar un archivo parquet específico de forma segura
  archivo_ejemplo = (
      Path("frodobots_metadata") / "data" / "chunk-000" / "file-000.parquet"
  )

  if archivo_ejemplo.exists():
    df_chunk = explorar_parquet(archivo_ejemplo)
  else:
    print(
        f"El archivo {archivo_ejemplo} no se encuentra localmente. Asegúrate de"
        " descargarlo primero."
    )