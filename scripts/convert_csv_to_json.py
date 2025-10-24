#!/usr/bin/env python3
"""Convert tabular CSV data into a JSON array persisted on disk.

This helper script is intended for preparing static datasets (e.g. fire stations,
emergency hospitals) that are consumed by the Streamlit application.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV data to a JSON array file."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        help="Optional output path. Defaults to the CSV path with a .json suffix.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indentation level for the generated JSON (default: 4).",
    )
    return parser.parse_args()


def normalize_value(value: str | None) -> Any:
    if value is None:
        return None

    trimmed = value.strip()
    if trimmed == "":
        return None

    lowered = trimmed.lower()
    if lowered in {"null", "none"}:
        return None

    if _looks_like_float(trimmed):
        try:
            return float(trimmed)
        except ValueError:
            return trimmed

    return trimmed


def _looks_like_float(value: str) -> bool:
    if value.count(".") != 1:
        return False
    left, right = value.split(".", maxsplit=1)
    if not left or not right:
        return False
    left = left.lstrip("+-")
    return left.isdigit() and right.isdigit()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header row detected in {path}")

        rows: List[Dict[str, Any]] = []
        for row in reader:
            normalized = {key: normalize_value(value) for key, value in row.items()}
            rows.append(normalized)
        return rows


def write_json(path: Path, data: Iterable[Dict[str, Any]], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(list(data), json_file, indent=indent, ensure_ascii=False)
        json_file.write("\n")


def main() -> int:
    args = parse_args()
    csv_path: Path = args.csv_path
    json_path: Path = args.json_path or csv_path.with_suffix(".json")

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        payload = read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001 - surface any parsing errors
        print(f"Failed to read CSV: {exc}", file=sys.stderr)
        return 1

    try:
        write_json(json_path, payload, indent=args.indent)
    except Exception as exc:  # noqa: BLE001 - surface IO issues
        print(f"Failed to write JSON: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(payload)} records to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
