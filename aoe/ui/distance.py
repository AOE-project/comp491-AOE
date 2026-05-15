import csv
import json
from pathlib import Path


def _parse_distance(value: str):
    value = value.strip()
    if value == "":
        return 0
    try:
        number = float(value)
    except ValueError:
        return 0
    return int(number) if number.is_integer() else number


def convert_csv_to_json(input_csv: str | Path, output_json: str | Path):
    distances_dict = {}

    with open(input_csv, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f, delimiter=","))

    if not reader or not reader[0]:
        raise ValueError("CSV dosyasi bos veya gecersiz.")

    buildings = [name.strip() for name in reader[0][1:] if name.strip()]

    for row in reader[1:]:
        if not row:
            continue
        origin = row[0].strip()
        if not origin:
            continue

        distances_dict[origin] = {}
        for i, destination in enumerate(buildings, start=1):
            raw_value = row[i] if i < len(row) else ""
            distances_dict[origin][destination] = _parse_distance(raw_value)

    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(distances_dict, jf, ensure_ascii=False, indent=4)

    return buildings


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / "campus_matrix.csv"
    output_json = base_dir / "campus_matrix.json"
    convert_csv_to_json(input_csv, output_json)
    print(f"JSON olusturuldu: {output_json}")