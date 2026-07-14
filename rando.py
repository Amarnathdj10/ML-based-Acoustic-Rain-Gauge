import pandas as pd
import numpy as np

master_path = r"D:\Coding journey\ML-based Acoustic Rain Gauge\audit_report\master_df_clean.parquet"
new_pickle = r"D:\Coding journey\ML-based Acoustic Rain Gauge\aligned_data\pickle files\september_2025_aligned_dataset.pkl"

master_df = pd.read_parquet(master_path)
new_df = pd.read_pickle(new_pickle)

new_df["source_pickle"] = "september_2025_aligned_dataset.pkl"

# optional: avoid duplicate rows if the same data is already present
combined = pd.concat([master_df, new_df], ignore_index=True)
combined = combined.drop_duplicates()

combined.to_parquet(master_path, index=False)