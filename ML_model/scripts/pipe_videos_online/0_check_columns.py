from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(repo_id="BitRobot/FrodoBots-Mini-4K", filename="metadata.parquet", repo_type="dataset")
meta = pd.read_parquet(path)
print("columnas:", meta.columns.tolist())
print(meta.head(3).to_string())
