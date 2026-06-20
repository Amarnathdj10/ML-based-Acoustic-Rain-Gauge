from pathlib import Path
import shutil
from datetime import datetime

root = Path("split_by_date_2")

for csv_file in root.glob("*.csv"):

    try:
        # Parse filename like 01-12-24.csv
        date = datetime.strptime(
            csv_file.stem,
            "%d-%m-%y"
        )

        month_folder = (
            root /
            f"{date.strftime('%b')}_{date.year}"
        )

        month_folder.mkdir(exist_ok=True)

        shutil.move(
            str(csv_file),
            str(month_folder / csv_file.name)
        )

        print(f"Moved {csv_file.name} -> {month_folder.name}")

    except ValueError:
        print(f"Skipped: {csv_file.name}")