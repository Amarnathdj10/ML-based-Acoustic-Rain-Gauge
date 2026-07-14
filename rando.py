import pandas as pd

master_path = r"D:\Coding journey\ML-based Acoustic Rain Gauge\audit_report\master_df_clean.parquet"
new_path = r"D:\Coding journey\ML-based Acoustic Rain Gauge\aligned_data\pickle files\september_2025_aligned_dataset.pkl"

master_df = pd.read_parquet(master_path)
new_df = pd.read_pickle(new_path)

# ensure the new rows are labeled correctly
new_df["source_pickle"] = "september_2025_aligned_dataset.pkl"

# align columns with master_df
new_df = new_df.reindex(columns=master_df.columns)

combined = pd.concat([master_df, new_df], ignore_index=True)

combined.to_parquet(master_path, index=False)