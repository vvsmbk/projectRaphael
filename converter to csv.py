import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

in_path = Path("./Desktop/influx backup/saw.backup.parquet")
df = pq.read_table(in_path).to_pandas()

df.to_csv("./Desktop/influx backup/saw.csv", index=False) 
