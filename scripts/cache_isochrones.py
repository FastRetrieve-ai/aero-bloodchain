#!/usr/bin/env python3
"""
Fetch and cache OpenRouteService isochrones for emergency responsibility hospitals.

This script reads hospital metadata from ``data/hospital.json`` and requests
isochrone polygons for each hospital using the OpenRouteService Isochrones API.
The merged GeoJSON output is written to ``data/hospital_isochrones.geojson`` so
that the Streamlit app can render travel-time contours without making live API calls.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib import error, request

# Add project src directory to path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (
    DATA_DIR,
    OPENROUTESERVICE_API_KEY,
    OPENROUTESERVICE_BASE_URL,
)


HOSPITAL_DATA_PATH = DATA_DIR / "hospital.json"
ISOCHRONE_OUTPUT_PATH = DATA_DIR / "hospital_isochrones.geojson"
DEFAULT_MINUTES: Sequence[int] = (30, 60)
DEFAULT_PROFILE = "driving-car"


def parse_args() -> argparse.Namespace:
    """Configure and return CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch OpenRouteService isochrones for hospitals listed in data/hospital.json "
            "and cache the merged GeoJSON to disk."
        )
    )
    parser.add_argument(
        "--minutes",
        type=int,
        nargs="+",
        default=list(DEFAULT_MINUTES),
        metavar="MIN",
        help="Travel-time buckets to request (minutes, default: 30 60 120).",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="OpenRouteService routing profile (default: driving-car).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ISOCHRONE_OUTPUT_PATH,
        help="Destination GeoJSON file (default: data/hospital_isochrones.geojson).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Delay between API calls to avoid rate limits (default: 5.0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file even if it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N hospitals (for testing).",
    )
    return parser.parse_args()


def slugify(name: str) -> str:
    """Create a filesystem-friendly slug for the hospital name."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", name.strip()).strip("-").lower()
    return slug or "hospital"


def normalize_minutes(raw_minutes: Iterable[int]) -> List[int]:
    """Filter minutes to positive unique integers in ascending order."""
    normalized: List[int] = sorted({int(m) for m in raw_minutes if int(m) > 0})
    return normalized


def load_hospitals(path: Path) -> List[Dict[str, Any]]:
    """Load hospital metadata from JSON file."""
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except OSError as exc:
        raise RuntimeError(f"Cannot read hospital data: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list in {path}, got {type(data)}")
    return data


def fetch_isochrones(
    *,
    lat: float,
    lon: float,
    minutes: Sequence[int],
    profile: str,
    base_url: str,
    api_key: str,
) -> Dict[str, Any]:
    """Call OpenRouteService Isochrones API and return parsed JSON."""
    url = f"{base_url}/v2/isochrones/{profile}"
    payload = {
        "locations": [[float(lon), float(lat)]],
        "range": [int(m) * 60 for m in minutes],
        "range_type": "time",
        "location_type": "start",
        "attributes": ["area"],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=30) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"OpenRouteService returned status {resp.status}")
            content = resp.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"OpenRouteService error {exc.code}: {exc.reason}; response={detail or 'n/a'}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouteService request failed: {exc.reason}") from exc

    try:
        return json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenRouteService response is not valid JSON") from exc


def assign_minutes_to_features(
    features: List[Dict[str, Any]],
    requested_minutes: Sequence[int],
) -> List[Dict[str, Any]]:
    """Attach the requested minute bucket to each feature."""
    if not features:
        return []

    sorted_minutes = list(requested_minutes)
    sorted_features = sorted(
        features,
        key=lambda feature: feature.get("properties", {}).get("value", 0),
    )
    minute_count = len(sorted_minutes)

    attached: List[Dict[str, Any]] = []
    for idx, feature in enumerate(sorted_features):
        minute_idx = idx if idx < minute_count else minute_count - 1
        minute_value = int(sorted_minutes[minute_idx])

        props = feature.setdefault("properties", {})
        # Preserve raw seconds while adding derived metadata.
        seconds = props.get("value")
        props["minutes"] = minute_value
        props["seconds"] = seconds
        props["requested_minutes"] = sorted_minutes
        attached.append(feature)
    return attached


def main() -> int:
    args = parse_args()

    if not OPENROUTESERVICE_API_KEY:
        print(
            "ERROR: OPENROUTESERVICE_API_KEY is not configured. Set it in .env or Streamlit secrets.",
            file=sys.stderr,
        )
        return 1

    MAX_MINUTES = 120
    minutes = normalize_minutes(args.minutes)
    if not minutes:
        print("ERROR: At least one positive minute bucket must be provided.", file=sys.stderr)
        return 1
    
    if any(minute > MAX_MINUTES for minute in minutes):
        print(f"ERROR: The maximum minutes is {MAX_MINUTES} minutes.", file=sys.stderr)
        return 1

    hospitals = load_hospitals(HOSPITAL_DATA_PATH)
    if args.limit is not None:
        hospitals = hospitals[: args.limit]

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        print(
            f"Skipping fetch because {output_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 0

    total = len(hospitals)
    if total == 0:
        print("No hospitals found in data/hospital.json; nothing to fetch.", file=sys.stderr)
        return 0

    print(f"Fetching isochrones for {total} hospitals ({minutes} minutes)...")
    features: List[Dict[str, Any]] = []
    success_count = 0
    skipped_count = 0

    for index, hospital in enumerate(hospitals, start=1):
        name = str(hospital.get("醫院全名") or hospital.get("name") or "").strip()
        level = str(hospital.get("緊急醫療能力分級") or "").strip()
        city = str(hospital.get("縣市別") or "").strip()
        region = str(hospital.get("區域別") or "").strip()

        try:
            lat = float(hospital["緯度"])
            lon = float(hospital["經度"])
        except (KeyError, TypeError, ValueError):
            print(f"[{index}/{total}] Skipping {name or '未命名醫院'}: missing coordinates.", file=sys.stderr)
            skipped_count += 1
            continue

        slug = slugify(name or f"hospital-{index}")

        print(f"[{index}/{total}] {name} ({level or '未分級'}) ...", end=" ", flush=True)
        try:

            response = fetch_isochrones(
                lat=lat,
                lon=lon,
                minutes=minutes,
                profile=args.profile,
                base_url=OPENROUTESERVICE_BASE_URL.rstrip("/"),
                api_key=OPENROUTESERVICE_API_KEY,
            )
        except RuntimeError as exc:
            print(f"FAILED ({exc})")
            skipped_count += 1
            continue

        raw_features = response.get("features", [])
        if not raw_features:
            print("FAILED (no features returned)")
            skipped_count += 1
            continue

        enriched = assign_minutes_to_features(raw_features, minutes)
        for feature in enriched:
            props = feature.setdefault("properties", {})
            props["hospital_name"] = name
            props["hospital_slug"] = slug
            props["hospital_level"] = level
            props["hospital_city"] = city
            props["hospital_region"] = region
            props["profile"] = args.profile
        features.extend(enriched)
        success_count += 1
        print("OK")

        if index < total and args.sleep > 0:
            time.sleep(args.sleep)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "minutes": minutes,
        "hospital_count": total,
        "succeeded": success_count,
        "skipped": skipped_count,
    }
    geojson_payload = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": metadata,
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(geojson_payload, fp, ensure_ascii=False, indent=2)

    display_path: Path = output_path
    try:
        display_path = output_path.relative_to(Path.cwd())
    except ValueError:
        display_path = output_path

    print(f"Saved {len(features)} isochrone features to {display_path}")
    if skipped_count > 0:
        print(f"Completed with {skipped_count} skipped hospital(s). See logs above for details.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
