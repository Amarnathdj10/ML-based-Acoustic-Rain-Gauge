from pathlib import Path
import pandas as pd

folder = Path(
    r"D:\Coding journey\ML-based Acoustic Rain Gauge\merged_split_by_date\Nov_2024"
)

csv_files = list(folder.glob("*.csv"))

if not csv_files:
    raise Exception("No CSV files found")

dfs = []

for csv_file in csv_files:
    print(f"Reading {csv_file.name}")

    df = pd.read_csv(csv_file, low_memory=False)

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    dfs.append(df)

# Combine
combined_df = pd.concat(dfs, ignore_index=True)

# Convert timestamp column
combined_df["time"] = pd.to_datetime(
    combined_df["time"],
    errors="coerce"
)

# Remove invalid rows
combined_df = combined_df.dropna(subset=["time"])

# Sort chronologically
combined_df = combined_df.sort_values("time")

# Remove duplicate timestamps
combined_df = combined_df.drop_duplicates(
    subset=["time"]
)

# Reset index
combined_df = combined_df.reset_index(drop=True)

# Output file
output_file = (
    folder /
    "november_2024_mechanical_rainfall_data.csv"
)

combined_df.to_csv(
    output_file,
    index=False
)

print(f"\nCreated {output_file}")
print(f"Rows: {len(combined_df)}")

# Delete source CSVs
for csv_file in csv_files:

    if csv_file != output_file:
        csv_file.unlink()
        print(f"Deleted {csv_file.name}")

print("\nFinished.")