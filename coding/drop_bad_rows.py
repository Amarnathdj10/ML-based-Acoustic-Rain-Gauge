from pathlib import Path
import pandas as pd

root = Path("merged_split_by_date")

for csv_file in root.rglob("*.csv"):

    try:
        df = pd.read_csv(csv_file)

        # Find timestamp column
        time_col = None
        for col in df.columns:
            if col.strip().lower() in ["time", "timestamp"]:
                time_col = col
                break

        if time_col is None:
            print(f"Skipping {csv_file}: no timestamp column")
            continue

        before = len(df)

        df[time_col] = pd.to_datetime(
            df[time_col],
            errors="coerce"
        )

        df = df.dropna(subset=[time_col])

        df = (
            df.sort_values(time_col)
              .drop_duplicates(subset=[time_col])
              .reset_index(drop=True)
        )

        after = len(df)

        df.to_csv(csv_file, index=False)

        print(
            f"{csv_file.name}: removed {before-after} duplicates"
        )

    except Exception as e:
        print(f"Error in {csv_file}: {e}")

print("Finished.")