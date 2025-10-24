#!/usr/bin/env python3
"""Geocode addresses from a CSV file using Google Maps Geocoding API."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import parse, request

# Add project src directory to path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lookup coordinates from Google Maps Geocoding API and output a JSON array."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the source CSV file containing the address column.",
    )
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        help="Optional output path. Defaults to the CSV path with a .json suffix.",
    )
    parser.add_argument(
        "--address-column",
        default="地址",
        help="Name of the CSV column that stores the full address.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to wait between API calls to avoid hitting rate limits.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indentation for the generated JSON output (default: 4).",
    )
    return parser.parse_args()


def validate_configuration() -> None:
    if not config.GOOGLE_MAPS_API_KEY:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not configured. "
            "Set it in your environment or .env file."
        )


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header row detected in {path}")
        return list(reader)


def build_geocode_url(address: str) -> str:
    params = {
        "address": address,
        "key": config.GOOGLE_MAPS_API_KEY,
    }
    return f"{config.GOOGLE_GEOCODE_ENDPOINT}?{parse.urlencode(params)}"


def fetch_coordinates(address: str) -> Optional[Dict[str, float]]:
    url = build_geocode_url(address)
    try:
        with request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed request for address '{address}': {exc}", file=sys.stderr)
        return None

    status = payload.get("status")
    if status != "OK":
        print(
            f"[warn] Geocoding failed for '{address}' - status: {status}",
            file=sys.stderr,
        )
        return None

    results = payload.get("results") or []
    if not results:
        return None

    location = (
        results[0]
        .get("geometry", {})
        .get("location", {})
    )
    lat = location.get("lat")
    lng = location.get("lng")
    if lat is None or lng is None:
        return None

    return {"緯度": float(lat), "經度": float(lng)}


def merge_coordinates(
    rows: Iterable[Dict[str, Any]], address_column: str, delay: float
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        address = row.get(address_column)
        if not address:
            print(f"[warn] Missing address in row {index}", file=sys.stderr)
            enriched.append(row)
            continue

        coordinates = fetch_coordinates(str(address))
        if coordinates:
            row.update(coordinates)
            print(
                f"[info] {index:03d} {address} -> "
                f"{coordinates['緯度']}, {coordinates['經度']}"
            )
        else:
            print(f"[warn] Unable to geocode row {index}: {address}", file=sys.stderr)

        enriched.append(row)

        if delay > 0 and index < len(rows):
            time.sleep(delay)

    return enriched


def write_json(path: Path, data: Iterable[Dict[str, Any]], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(list(data), json_file, ensure_ascii=False, indent=indent)
        json_file.write("\n")


def main() -> int:
    args = parse_args()
    validate_configuration()

    csv_path: Path = args.csv_path
    json_path: Path = args.json_path or csv_path.with_suffix(".json")
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        rows = read_csv_rows(csv_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to read CSV: {exc}", file=sys.stderr)
        return 1

    enriched = merge_coordinates(rows, args.address_column, args.sleep)

    try:
        write_json(json_path, enriched, indent=args.indent)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to write JSON: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(enriched)} records to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
