from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return clean or "subset"


def matches(row: dict[str, Any], keywords: list[str]) -> bool:
    haystack = " ".join(
        [
            normalize_text(row.get("caption")),
            normalize_text(row.get("aspect_list")),
        ]
    ).lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small CLAPv2/MusicCaps subset.")
    parser.add_argument("--keywords", nargs="+", default=["trumpet"])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="datasets/custom_music")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("CLAPv2/MusicCaps", split=args.split)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    metadata_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    keyword_slug = slug("_".join(args.keywords))
    for row in dataset:
        if not matches(row, args.keywords):
            continue

        number = len(metadata_rows) + 1
        filename = f"{keyword_slug}_{number:03d}.wav"
        relative_audio_path = f"audio/{filename}"
        audio_path = audio_dir / filename

        audio = row["audio"]
        if audio.get("bytes") is not None:
            array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        elif audio.get("path"):
            array, sampling_rate = sf.read(audio["path"], dtype="float32")
        else:
            raise RuntimeError(f"Audio payload is missing for row {row.get('index')}")

        array = np.asarray(array, dtype=np.float32)
        sf.write(audio_path, array, int(sampling_rate))

        caption = normalize_text(row.get("caption")).strip()
        metadata_rows.append(
            {
                "file_name": relative_audio_path,
                "text": caption,
            }
        )
        manifest_rows.append(
            {
                "file_name": relative_audio_path,
                "caption": caption,
                "aspect_list": normalize_text(row.get("aspect_list")),
                "index": normalize_text(row.get("index")),
                "ytid": normalize_text(row.get("ytid")),
                "start_s": normalize_text(row.get("start_s")),
                "end_s": normalize_text(row.get("end_s")),
            }
        )

        print(f"[{number:02d}/{args.limit}] {filename}: {caption[:90]}")
        if len(metadata_rows) >= args.limit:
            break

    if not metadata_rows:
        raise SystemExit(f"No rows matched keywords: {', '.join(args.keywords)}")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    manifest_path = output_dir / "subset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "caption", "aspect_list", "index", "ytid", "start_s", "end_s"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"Created {len(metadata_rows)} examples in {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
