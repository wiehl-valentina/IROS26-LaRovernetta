
import pandas as pd 
from pathlib import Path

def explorar_parquet(
    ruta_archivo: str | Path,
) -> pd.DataFrame:
  """Carga un archivo parquet de forma segura y muestra sus dimensiones,

  columnas y las primeras filas. Retorna el DataFrame de Pandas.
  """
  ruta_parquet = Path(ruta_archivo)

  print(f"Cargando archivo parquet desde: {ruta_parquet}...")
  df = pd.read_parquet(ruta_parquet)

  print(f"\nDimensiones del dataset (Filas, Columnas): {df.shape}")
  print("\nPrimeras columnas disponibles:")
  print(list(df.columns))

  print("\nPrimeras 5 filas:")
  print(df.head())

  return df
