from influxdb_client_3 import InfluxDBClient3
import pandas as pd  
import pyarrow.parquet as pq
from pathlib import Path 
from dataclasses import dataclass 
from typing import Optional, Dict, List, Tuple  

LIMIT = 1_000_000

client = InfluxDBClient3(
    token=" ", #YOUR TOKEN HERE
    host="http://10.65.1.77:10102", #YOUR LINK TO DB HERE
    database="spp1-main" #YOUR NAME OF DB HERE
)

TABLE_NAME = "saw"
OUTPUT_FILE = "influxDB Backup/saw.backup.parquet"

offset = 0
all_frames = [] # List to store each chunk in memory

print(f"Starting export of '{TABLE_NAME}'...")

while True:
    print(f"Fetching rows {offset} to {offset + LIMIT}...", end="\r")

    # Order by time is critical to ensure pages don't shuffle or overlap
    query = f"""
        SELECT * FROM "{TABLE_NAME}" 
        ORDER BY time ASC
        LIMIT {LIMIT} OFFSET {offset}
    """

    try:
        # Fetch the chunk
        df_chunk = client.query(query=query, mode="pandas")

        # If chunk is empty, we are done
        if df_chunk.empty:
            print(f"\nNo more data found after offset {offset}.")
            break
        
        # Append to our in-memory list
        all_frames.append(df_chunk)

        # Check if this was the last page (less rows than limit)
        if len(df_chunk) < LIMIT:
            print(f"\nReached end of dataset at offset {offset + len(df_chunk)}.")
            break

        # Prepare for next iteration
        offset += LIMIT

    except Exception as e:
        print(f"\nCRITICAL ERROR at offset {offset}: {e}")
        # Depending on importance, you might want to 'break' or 'raise' here
        break

# --- Final Consolidation ---
if all_frames:
    print("Concatenating data...")
    final_df = pd.concat(all_frames, ignore_index=True)

    print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print("Export successful.")
else:
    print("No data was retrieved.")