"""Utilities for sampling UAV elevation profiles from the 20 m DTM."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod, Transformer

from config import DTM_SAMPLE_INTERVAL_M, DTM_TIF_PATH

WGS84_CRS = "EPSG:4326"


@lru_cache(maxsize=1)
def _get_geod() -> Geod:
    """Return a cached WGS84 geodesic helper."""

    return Geod(ellps="WGS84")


def _resolve_dtm_path(dtm_path: str | Path | None = None) -> Path:
    """Resolve the DTM GeoTIFF path, defaulting to configuration."""

    path = Path(dtm_path) if dtm_path is not None else Path(DTM_TIF_PATH)
    return path.expanduser()


def _build_sample_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    sample_interval_m: float,
) -> Tuple[List[Tuple[float, float]], np.ndarray, float]:
    """Construct geodesic sample points and cumulative distances."""

    geod = _get_geod()
    start_lat, start_lon = start
    end_lat, end_lon = end

    _, _, total_distance = geod.inv(start_lon, start_lat, end_lon, end_lat)
    # Degenerate case: identical points
    if total_distance <= 0:
        return [(start_lat, start_lon)], np.array([0.0], dtype=float), 0.0

    interval = max(sample_interval_m, 1.0)
    # Number of segments needed to respect interval; ensure at least two points
    segment_count = max(int(math.ceil(total_distance / interval)), 1)
    intermediate_count = max(segment_count - 1, 0)

    intermediate_lonlat = []
    if intermediate_count:
        intermediate_lonlat = geod.npts(
            start_lon,
            start_lat,
            end_lon,
            end_lat,
            intermediate_count,
        )

    latlon_points: List[Tuple[float, float]] = [(start_lat, start_lon)]
    latlon_points.extend((lat, lon) for lon, lat in intermediate_lonlat)
    latlon_points.append((end_lat, end_lon))

    cumulative = [0.0]
    for i in range(len(latlon_points) - 1):
        _, _, segment_distance = geod.inv(
            latlon_points[i][1],
            latlon_points[i][0],
            latlon_points[i + 1][1],
            latlon_points[i + 1][0],
        )
        cumulative.append(cumulative[-1] + segment_distance)

    return latlon_points, np.array(cumulative, dtype=float), float(total_distance)


def _transform_points(
    points: Sequence[Tuple[float, float]],
    transformer: Transformer,
) -> List[Tuple[float, float]]:
    """Project latitude/longitude points using a transformer."""

    projected: List[Tuple[float, float]] = []
    for lat, lon in points:
        x, y = transformer.transform(lon, lat)
        projected.append((x, y))
    return projected


def _sample_elevations(
    dataset: rasterio.io.DatasetReader,
    projected_points: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, List[bool]]:
    """Sample elevations for the provided dataset coordinates."""

    samples = list(dataset.sample(projected_points))
    nodata = None
    if dataset.nodatavals:
        nodata = dataset.nodatavals[0]

    bounds = dataset.bounds
    elevations: List[float] = []
    valid_flags: List[bool] = []

    for (x, y), value in zip(projected_points, samples):
        inside = bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
        elevation = np.nan
        if inside and value.size:
            if np.ma.isMaskedArray(value) and np.any(value.mask):
                elevation = np.nan
            else:
                current = float(value[0])
                if nodata is not None:
                    if (np.isnan(nodata) and np.isnan(current)) or (
                        np.isfinite(nodata) and current == nodata
                    ):
                        current = np.nan
                elevation = current

        elevations.append(elevation)
        valid_flags.append(np.isfinite(elevation))

    return np.array(elevations, dtype=float), valid_flags


def sample_profile(
    start: Tuple[float, float],
    end: Tuple[float, float],
    sample_interval_m: float | None = None,
    dtm_path: str | Path | None = None,
) -> pd.DataFrame:
    """Sample an elevation profile between two WGS84 points.

    Args:
        start: Starting point as (lat, lon).
        end: Ending point as (lat, lon).
        sample_interval_m: Maximum spacing between samples in metres.
        dtm_path: Optional override path for the DTM GeoTIFF.

    Returns:
        DataFrame with columns ["lat", "lon", "distance_m", "elevation_m", "is_valid"].

    Raises:
        FileNotFoundError: If the DTM GeoTIFF cannot be located.
        ValueError: If the DTM lacks an assigned CRS.
    """

    interval = float(sample_interval_m or DTM_SAMPLE_INTERVAL_M)
    points, cumulative_dist, total_distance = _build_sample_points(start, end, interval)

    dtm_file = _resolve_dtm_path(dtm_path)
    if not dtm_file.exists():
        raise FileNotFoundError(
            f"DTM 檔案不存在：{dtm_file}. 請更新環境變數 DTM_TIF_PATH 或設定檔。"
        )

    with rasterio.open(dtm_file) as dataset:
        if dataset.crs is None:
            raise ValueError("DTM GeoTIFF 缺少座標系 (CRS)，請先使用 gdal_edit.py 設定。")

        transformer = Transformer.from_crs(
            WGS84_CRS,
            dataset.crs,
            always_xy=True,
        )
        projected_points = _transform_points(points, transformer)
        elevations, valid_flags = _sample_elevations(dataset, projected_points)

    profile = pd.DataFrame(
        {
            "lat": [lat for lat, _ in points],
            "lon": [lon for _, lon in points],
            "distance_m": cumulative_dist,
            "elevation_m": elevations,
            "is_valid": valid_flags,
        }
    )

    profile.attrs.update(
        {
            "horizontal_distance_m": float(cumulative_dist[-1]) if len(cumulative_dist) else 0.0,
            "sample_interval_m": interval,
            "total_distance_m": total_distance,
            "dtm_path": str(dtm_file),
        }
    )

    return profile


def compute_metrics(
    profile: pd.DataFrame,
    speed_value: float,
    speed_unit: str = "km/h",
) -> dict:
    """Compute derived distance, elevation, and time metrics."""

    if profile.empty:
        base = {
            "horizontal_distance_m": 0.0,
            "vertical_gain_m": 0.0,
            "vertical_loss_m": 0.0,
            "path_length_3d_m": 0.0,
            "additive_path_length_m": 0.0,
            "eta_seconds": math.inf,
            "max_elevation_m": None,
            "min_elevation_m": None,
        }
        base["profile_resolution_m"] = None
        return base

    distances = profile["distance_m"].to_numpy(dtype=float)
    elevations = profile["elevation_m"].to_numpy(dtype=float)

    horizontal_distance = float(distances[-1]) if distances.size else 0.0
    vertical_gain = 0.0
    vertical_loss = 0.0
    path_length_3d = 0.0

    for i in range(len(distances) - 1):
        dh = distances[i + 1] - distances[i]
        if dh < 0:
            continue
        z1 = elevations[i]
        z2 = elevations[i + 1]
        if np.isfinite(z1) and np.isfinite(z2):
            dz = z2 - z1
            if dz > 0:
                vertical_gain += dz
            elif dz < 0:
                vertical_loss += abs(dz)
            path_length_3d += math.sqrt(dh * dh + dz * dz)
        else:
            path_length_3d += dh

    additive_path_length = horizontal_distance + vertical_gain

    speed = float(speed_value)
    unit = (speed_unit or "km/h").strip().lower()
    if unit in {"km/h", "kmh", "kph"}:
        speed_m_s = speed / 3.6
    elif unit in {"m/s", "mps"}:
        speed_m_s = speed
    else:
        speed_m_s = speed / 3.6

    if speed_m_s <= 0:
        eta = math.inf
    else:
        eta = additive_path_length / speed_m_s

    resolution = None
    if len(distances) > 1:
        resolution = float(np.diff(distances).mean())

    max_elevation = float(np.nanmax(elevations)) if np.isfinite(np.nanmax(elevations)) else None
    min_elevation = float(np.nanmin(elevations)) if np.isfinite(np.nanmin(elevations)) else None

    return {
        "horizontal_distance_m": horizontal_distance,
        "vertical_gain_m": vertical_gain,
        "vertical_loss_m": vertical_loss,
        "path_length_3d_m": path_length_3d,
        "additive_path_length_m": additive_path_length,
        "eta_seconds": eta,
        "profile_resolution_m": resolution,
        "max_elevation_m": max_elevation,
        "min_elevation_m": min_elevation,
    }
