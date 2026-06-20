from pathlib import Path
import pandas as pd

# Load CSV
df = pd.read_csv("D:\Coding journey\ML-based Acoustic Rain Gauge\mechanical-data-2026-06-11 12_26_37.csv")

# Convert timestamp column to datetime
df["time"] = pd.to_datetime(df["time"])

# Output folder
output_folder = Path("split_by_date_2")
output_folder.mkdir(exist_ok=True)

# Group by date
for date, group in df.groupby(df["time"].dt.date):

    filename = pd.Timestamp(date).strftime("%d-%m-%y") + ".csv"

    group.to_csv(
        output_folder / filename,
        index=False
    )

    print(f"Created {filename} ({len(group)} rows)")

print("Done.")