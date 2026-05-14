from pathlib import Path

from aoe.ui.distance import convert_csv_to_json


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "aoe" / "ui"
    input_csv = base_dir / "campus_matrix.csv"
    output_json = base_dir / "campus_matrix.json"
    convert_csv_to_json(input_csv, output_json)
    print(f"JSON olusturuldu: {output_json}")


if __name__ == "__main__":
    main()
