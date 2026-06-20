from pathlib import Path
import pandas as pd

root = Path("merged_split_by_date")

for month_folder in root.iterdir():

    if not month_folder.is_dir():
        continue

    print(f"\nProcessing {month_folder.name}")

    csv_files = list(month_folder.glob("*.csv"))

    if not csv_files:
        print("No CSV files found")
        continue

    dfs = []

    for csv_file in csv_files:

        try:
            df = pd.read_csv(csv_file)

            # Remove empty columns
            df = df.dropna(axis=1, how="all")

            dfs.append(df)

        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")

    if not dfs:
        continue

    combined_df = pd.concat(dfs, ignore_index=True)

    # Find timestamp column
    time_col = None

    for col in combined_df.columns:
        if col.strip().lower() in ["time", "timestamp"]:
            time_col = col
            break

    if time_col is None:
        print(f"No timestamp column found in {month_folder.name}")
        continue

    # Clean timestamps
    combined_df[time_col] = pd.to_datetime(
        combined_df[time_col],
        errors="coerce"
    )

    combined_df = combined_df.dropna(subset=[time_col])

    # Sort and deduplicate
    combined_df = (
        combined_df
        .sort_values(time_col)
        .drop_duplicates(subset=[time_col])
        .reset_index(drop=True)
    )

    # Generate filename
    output_name = (
        month_folder.name.lower()
        + "_mechanical_rainfall_data.csv"
    )

    output_file = month_folder / output_name

    combined_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Created {output_name} "
        f"({len(combined_df)} rows)"
    )

    # Delete original CSVs
    for csv_file in csv_files:
        if csv_file != output_file:
            csv_file.unlink()

    print(f"Deleted {len(csv_files)} source CSV files")

print("\nFinished all folders.")