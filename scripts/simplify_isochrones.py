#!/usr/bin/env python3
"""
Combine and simplify hospital isochrone GeoJSON features.

Reads an existing `hospital_isochrones.geojson`, unions polygons by minute bucket,
optionally simplifies them, and writes a compact GeoJSON for downstream rendering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from shapely.geometry import GeometryCollection, mapping, shape
from shapely.ops import unary_union


DEFAULT_TOLERANCE = 0.0015  # roughly 150m at Taipei latitude


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Union and simplify hospital isochrone polygons."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source hospital_isochrones.geojson file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Destination GeoJSON file. Defaults to <input> with '.simplified.geojson' suffix.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Simplification tolerance in degrees (default: %(default)s). Use 0 to skip simplification.",
    )
    parser.add_argument(
        "--difference-ring",
        action="store_true",
        help="Output the 60-minute layer as an outer ring (difference between 60 and 30 minutes).",
    )
    return parser.parse_args()


def flatten_geometry(geom) -> List[Any]:
    """Yield simple polygon-like geometries from any collection."""
    if geom.is_empty:
        return []
    if isinstance(geom, GeometryCollection):
        geoms: List[Any] = []
        for sub in geom:
            geoms.extend(flatten_geometry(sub))
        return geoms
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return [geom]


def main() -> int:
    args = parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_path: Path
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.with_suffix(".simplified.geojson")

    with input_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    features: Iterable[Dict[str, Any]] = payload.get("features", [])
    grouped: Dict[int, List[Any]] = {}
    hospitals_per_minute: Dict[int, set[str]] = {}

    for feature in features:
        props = feature.get("properties", {}) or {}
        minutes = props.get("minutes")
        if minutes is None and "value" in props:
            minutes = int(round(float(props["value"]) / 60.0))
        if minutes is None:
            continue
        minutes = int(minutes)
        geom = feature.get("geometry")
        if not geom:
            continue

        try:
            shaped = shape(geom)
        except Exception:
            continue

        grouped.setdefault(minutes, []).append(shaped)

        hospital_slug = props.get("hospital_slug") or props.get("hospital_name")
        if hospital_slug:
            hospitals_per_minute.setdefault(minutes, set()).add(str(hospital_slug))

    if not grouped:
        print("No geometries found to process.")
        return 1

    simplified_features: List[Dict[str, Any]] = []
    minutes_sorted = sorted(grouped.keys())
    unions: Dict[int, Any] = {}

    for minute in minutes_sorted:
        geoms = grouped[minute]
        merged = unary_union(geoms)
        if args.tolerance > 0:
            merged = merged.simplify(args.tolerance, preserve_topology=True)
        unions[minute] = merged

    # Optionally compute 60-minute ring difference
    if args.difference_ring and 60 in unions and 30 in unions:
        outer = unions[60]
        inner = unions[30]
        unions[60] = outer.difference(inner)

    for minute in minutes_sorted:
        merged = unions[minute]
        for geom in flatten_geometry(merged):
            simplified_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {
                        "minutes": minute,
                        "seconds": minute * 60,
                        "hospital_count": len(hospitals_per_minute.get(minute, [])),
                        "simplify_tolerance": args.tolerance,
                        "difference_mode": bool(args.difference_ring),
                    },
                }
            )

    metadata = {
        "source": str(input_path),
        "generated": True,
        "tolerance": args.tolerance,
        "difference_ring": bool(args.difference_ring),
        "minutes": minutes_sorted,
    }

    counts_by_minute = {minute: len(hospitals_per_minute.get(minute, [])) for minute in minutes_sorted}

    output = {
        "type": "FeatureCollection",
        "features": simplified_features,
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, separators=(",", ":"))

    print(f"Wrote {len(simplified_features)} simplified isochrone feature(s) to {output_path}")
    print(f"Minutes processed: {minutes_sorted} · Tolerance: {args.tolerance} · Difference ring: {args.difference_ring}")
    print(f"Hospital counts by minute: {counts_by_minute}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
