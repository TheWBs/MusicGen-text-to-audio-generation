from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    dataset_dir = Path("datasets/custom_music")
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise SystemExit(f"Missing {metadata_path}")

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("metadata.csv is empty")

    missing = []
    for row in rows:
        file_name = row.get("file_name", "")
        caption = row.get("text", "")
        if not file_name or not caption:
            raise SystemExit("Each row must include file_name and text")
        if not (dataset_dir / file_name).exists():
            missing.append(file_name)

    if missing:
        raise SystemExit(f"Missing audio files: {', '.join(missing[:10])}")

    print(f"Dataset looks valid: {len(rows)} captioned audio files.")


if __name__ == "__main__":
    main()
