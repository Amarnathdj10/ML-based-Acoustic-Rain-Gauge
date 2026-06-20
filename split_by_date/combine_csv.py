from pathlib import Path
import pandas as pd

# Folder containing the CSV files
folder = Path(
    "D:\Coding journey\ML-based Acoustic Rain Gauge\split_by_date\Oct_2025"
)

dfs = []

for csv_file in folder.glob("*.csv"):
    print(f"Reading {csv_file.name}")

    try:
        df = pd.read_csv(csv_file)

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        # Standardize column names
        df.columns = df.columns.str.strip()

        dfs.append(df)

    except Exception as e:
        print(f"Skipping {csv_file.name}: {e}")

# Combine all CSVs
combined_df = pd.concat(dfs, ignore_index=True)

# Convert time column
combined_df["time"] = pd.to_datetime(combined_df["time"])

# Sort chronologically
combined_df = combined_df.sort_values("time")

# Remove duplicate timestamps
combined_df = combined_df.drop_duplicates(subset=["time"])

# Reset index
combined_df = combined_df.reset_index(drop=True)

# Save
combined_df.to_csv("October_2025_mechanical_rainfall_data.csv", index=False)

print("\nDone!")
print(f"Rows: {len(combined_df)}")
print(f"Start: {combined_df['time'].min()}")
print(f"End: {combined_df['time'].max()}")