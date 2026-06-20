from pathlib import Path
import pandas as pd

folder = Path("merged_split_by_date")

dfs = []

for csv_file in folder.rglob("*.csv"):

    try:
        print(f"Reading {csv_file}")

        df = pd.read_csv(
            csv_file,
            low_memory=False
        )

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        dfs.append(df)

    except Exception as e:
        print(f"Skipping {csv_file}: {e}")

# Combine all CSVs
combined_df = pd.concat(
    dfs,
    ignore_index=True
)

# Standardize column names
combined_df.columns = (
    combined_df.columns
    .str.strip()
)

# Find timestamp column
time_col = None

for col in combined_df.columns:
    if col.lower() in ["time", "timestamp"]:
        time_col = col
        break

if time_col is None:
    raise ValueError(
        "No time/timestamp column found"
    )

# Convert timestamps
combined_df[time_col] = pd.to_datetime(
    combined_df[time_col],
    errors="coerce"
)

# Remove invalid rows
combined_df = combined_df.dropna(
    subset=[time_col]
)

# Sort chronologically
combined_df = combined_df.sort_values(
    time_col
)

# Remove duplicate timestamps
combined_df = combined_df.drop_duplicates(
    subset=[time_col]
)

# Reset index
combined_df = combined_df.reset_index(
    drop=True
)

# Save
combined_df.to_csv(
    "MECH_DATA_MASTER.csv",
    index=False
)

print("\nDone!")
print(f"Rows: {len(combined_df)}")
print(
    f"Start: {combined_df[time_col].min()}"
)
print(
    f"End: {combined_df[time_col].max()}"
)