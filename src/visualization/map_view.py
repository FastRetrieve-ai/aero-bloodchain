"""
Interactive map visualizations for emergency cases
"""
import json
import math
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import folium
import pandas as pd
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen, HeatMap, MiniMap
import plotly.express as px
import plotly.graph_objects as go

from config import DATA_DIR, DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM

try:
    from shapely.geometry import GeometryCollection, shape, mapping
    from shapely.ops import unary_union
    from shapely.geometry.base import BaseGeometry
    from shapely import geometry as shapely_geometry
    SHAPELY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SHAPELY_AVAILABLE = False
    GeometryCollection = None  # type: ignore
    BaseGeometry = None  # type: ignore
    shapely_geometry = None  # type: ignore
    shape = None  # type: ignore
    mapping = None  # type: ignore
    unary_union = None  # type: ignore


# District coordinates for New Taipei City
DISTRICT_COORDINATES = {
    "板橋區": [25.0116, 121.4625],
    "新莊區": [25.0372, 121.4325],
    "中和區": [24.9994, 121.4991],
    "永和區": [25.0039, 121.5156],
    "土城區": [24.9733, 121.4420],
    "樹林區": [24.9906, 121.4201],
    "三峽區": [24.9342, 121.3697],
    "鶯歌區": [24.9545, 121.3538],
    "三重區": [25.0619, 121.4885],
    "蘆洲區": [25.0847, 121.4741],
    "五股區": [25.0829, 121.4384],
    "泰山區": [25.0572, 121.4301],
    "林口區": [25.0770, 121.3926],
    "淡水區": [25.1688, 121.4406],
    "金山區": [25.2217, 121.6370],
    "萬里區": [25.1797, 121.6891],
    "汐止區": [25.0672, 121.6423],
    "瑞芳區": [25.1089, 121.8058],
    "貢寮區": [25.0202, 121.9086],
    "平溪區": [25.0258, 121.7391],
    "雙溪區": [25.0347, 121.8656],
    "石碇區": [24.9959, 121.6582],
    "深坑區": [25.0023, 121.6165],
    "石門區": [25.2906, 121.5685],
    "八里區": [25.1448, 121.3967],
    "坪林區": [24.9360, 121.7108],
    "烏來區": [24.8656, 121.5498],
    "三芝區": [25.2598, 121.5008],
}

DISTRICT_COORDINATES.update({
    # 基隆市
    "中山區": [25.1501, 121.7329],   # Keelung Zhongshan Dist.
    "中正區": [25.1425, 121.7747],   # Keelung Zhongzheng Dist.
    "暖暖區": [25.1014, 121.7377],   # Nuannuan Station vicinity
    # 新北市
    "新店區": [24.9678, 121.5414],   # Xindian District Office Station
    # 桃園市
    "八德區": [24.9546, 121.2926],   # Bade District
    "大溪區": [24.8806, 121.2871],   # Daxi District
    "蘆竹區": [25.0333, 121.2833],   # Luzhu District 
    "龜山區": [24.9950, 121.3381],   # Guishan District (區公所附近)  
    # 南投縣
    "集集鎮": [23.8286, 120.7864],   # Jiji Township
})

HOSPITAL_DATA_PATH = DATA_DIR / "hospital.json"
FIRE_STATION_DATA_PATH = DATA_DIR / "fire-station.json"
ISOCHRONE_CACHE_FILE = DATA_DIR / "hospital_isochrones.geojson"
ISOCHRONE_SIMPLIFIED_FILE = DATA_DIR / "hospital_isochrones.simplified.geojson"

HOSPITAL_LEVEL_STYLES: Dict[str, Dict[str, str]] = {
    "重度": {"color": "#d73027", "emoji": "🏥"},
    "中度": {"color": "#fc8d59", "emoji": "🏥"},
    "一般": {"color": "#7fc97f", "emoji": "🏨"},
}
DEFAULT_HOSPITAL_STYLE = {"color": "#636363", "emoji": "🏥"}

FIRE_STATION_EMOJI = "🚒"
FIRE_STATION_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]

ISOCHRONE_SUPPORTED_MINUTES: Tuple[int, int] = (30, 60)
ISOCHRONE_DEFAULT_MINUTES: Tuple[int, int] = ISOCHRONE_SUPPORTED_MINUTES
ISOCHRONE_STYLE_BY_MINUTE: Dict[int, Dict[str, Any]] = {
    30: {"color": "#1f78b4", "fillColor": "#1f78b4", "fillOpacity": 0.35},
    60: {"color": "#fdae61", "fillColor": "#fdae61", "fillOpacity": 0.25},
}

_SLUG_CLEAN_PATTERN = re.compile(r"[^0-9a-zA-Z]+")


def _create_base_map() -> folium.Map:
    """Create a Folium base map with shared tiles and mini-map controls."""
    base_map = folium.Map(
        location=DEFAULT_MAP_CENTER,
        zoom_start=DEFAULT_MAP_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    MiniMap(toggle_display=True).add_to(base_map)
    Fullscreen(position='topleft').add_to(base_map)
    return base_map


def _slugify_hospital_name(name: str) -> str:
    """Create a deterministic slug for hospital names."""
    slug = _SLUG_CLEAN_PATTERN.sub("-", str(name).strip()).strip("-").lower()
    return slug or "hospital"


def _union_geojson_features(
    features: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union a collection of GeoJSON features if Shapely is available."""
    feature_list = [feature for feature in features if feature.get("geometry")]
    if not feature_list:
        return []

    if not SHAPELY_AVAILABLE:
        return feature_list

    geometries: List[BaseGeometry] = []
    for feature in feature_list:
        try:
            geometries.append(shape(feature["geometry"]))
        except Exception:
            continue

    if not geometries:
        return []

    try:
        unioned = unary_union(geometries)
    except Exception:
        return feature_list

    if unioned.is_empty:
        return []

    unified_features: List[Dict[str, Any]] = []

    def append_geometry(geom: BaseGeometry) -> None:
        if geom.is_empty:
            return
        if isinstance(geom, GeometryCollection):
            for sub_geom in geom:
                append_geometry(sub_geom)
            return
        if geom.geom_type == "MultiPolygon":
            for sub_geom in geom.geoms:
                append_geometry(sub_geom)
            return
        unified_features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {},
            }
        )

    append_geometry(unioned)
    return unified_features or feature_list


@lru_cache(maxsize=1)
def _load_cached_isochrones() -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Load cached isochrone GeoJSON into a lookup keyed by hospital slug and minute."""
    if not ISOCHRONE_CACHE_FILE.exists():
        return {}

    try:
        with ISOCHRONE_CACHE_FILE.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return {}

    features = payload.get("features", [])
    cache: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for feature in features:
        props = feature.get("properties", {})
        slug = props.get("hospital_slug")
        if not slug:
            name = props.get("hospital_name")
            if not isinstance(name, str):
                continue
            slug = _slugify_hospital_name(name)
        try:
            minute_value = int(props.get("minutes"))
        except (TypeError, ValueError):
            continue
        cache.setdefault(slug, {}).setdefault(minute_value, []).append(feature)
    return cache


@lru_cache(maxsize=1)
def _load_unionized_isochrones() -> Dict[int, List[Dict[str, Any]]]:
    """
    Load unionized isochrone features grouped by minutes.

    Prefers the simplified cache if available; otherwise falls back to
    runtime union across hospital-specific geometries.
    """
    if ISOCHRONE_SIMPLIFIED_FILE.exists():
        try:
            with ISOCHRONE_SIMPLIFIED_FILE.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, json.JSONDecodeError):
            pass
        else:
            simplified_features: Dict[int, List[Dict[str, Any]]] = {}
            for feature in payload.get("features", []):
                props = feature.get("properties", {}) or {}
                minutes = props.get("minutes")
                if minutes is None and "value" in props:
                    minutes = int(round(float(props["value"]) / 60.0))
                if minutes is None:
                    continue
                minutes = int(minutes)
                simplified_features.setdefault(minutes, []).append(feature)
            if simplified_features:
                return simplified_features

    # Fallback to aggregating the raw cache, unioning when possible
    hospital_isochrones = _load_cached_isochrones()
    if not hospital_isochrones:
        return {}

    aggregated: Dict[int, List[Dict[str, Any]]] = {}
    for minute_map in hospital_isochrones.values():
        for minute, features in minute_map.items():
            aggregated.setdefault(minute, []).extend(features)

    unionized: Dict[int, List[Dict[str, Any]]] = {}
    for minute, features in aggregated.items():
        if SHAPELY_AVAILABLE:
            union_features = _union_geojson_features(features)
        else:
            union_features = features
        processed: List[Dict[str, Any]] = []
        for feature in union_features:
            geom = feature.get("geometry")
            if not geom:
                continue
            processed.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"minutes": minute},
                }
            )
        unionized[minute] = processed

    return unionized


def _prepare_hospital_dataframe(
    hospital_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, List[str]]:
    """Load and standardize the hospital dataset."""
    warnings: List[str] = []
    if hospital_df is None:
        try:
            if not HOSPITAL_DATA_PATH.exists():
                warnings.append("找不到 data/hospital.json，無法載入急救責任醫院資料。")
                return pd.DataFrame(), warnings
            with HOSPITAL_DATA_PATH.open("r", encoding="utf-8") as fp:
                raw_data = json.load(fp)
            hospital_df_local = pd.DataFrame(raw_data)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"讀取 hospital.json 時發生錯誤：{exc}")
            return pd.DataFrame(), warnings
    else:
        hospital_df_local = hospital_df.copy()

    if hospital_df_local.empty:
        warnings.append("沒有可顯示的醫院資料。")
        return pd.DataFrame(), warnings

    hospital_df_local = hospital_df_local.rename(
        columns={
            "醫院全名": "name",
            "緊急醫療能力分級": "level",
            "經度": "lon",
            "緯度": "lat",
            "縣市別": "city",
            "區域別": "region",
        }
    )
    text_columns = ["name", "level", "city", "region"]
    for col in text_columns:
        if col not in hospital_df_local.columns:
            hospital_df_local[col] = ""
        hospital_df_local[col] = (
            hospital_df_local[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    hospital_df_local["lat"] = pd.to_numeric(
        hospital_df_local.get("lat"), errors="coerce"
    )
    hospital_df_local["lon"] = pd.to_numeric(
        hospital_df_local.get("lon"), errors="coerce"
    )
    hospital_df_local = hospital_df_local.dropna(
        subset=["name", "lat", "lon"]
    ).reset_index(drop=True)

    if hospital_df_local.empty:
        warnings.append("沒有可顯示的醫院資料。")
        return pd.DataFrame(), warnings

    expected_cols = ["name", "level", "lat", "lon", "city", "region"]
    for col in expected_cols:
        if col not in hospital_df_local.columns:
            hospital_df_local[col] = ""
    hospital_df_local = hospital_df_local[expected_cols]
    return hospital_df_local, warnings


def _prepare_fire_station_dataframe(
    station_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, List[str]]:
    """Load and standardize the fire station dataset."""
    warnings: List[str] = []
    if station_df is None:
        try:
            if not FIRE_STATION_DATA_PATH.exists():
                warnings.append("找不到 data/fire-station.json，無法載入消防隊資料。")
                return pd.DataFrame(), warnings
            with FIRE_STATION_DATA_PATH.open("r", encoding="utf-8") as fp:
                raw_data = json.load(fp)
            station_df_local = pd.DataFrame(raw_data)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"讀取 fire-station.json 時發生錯誤：{exc}")
            return pd.DataFrame(), warnings
    else:
        station_df_local = station_df.copy()

    if station_df_local.empty:
        warnings.append("沒有可顯示的消防隊資料。")
        return pd.DataFrame(), warnings

    station_df_local = station_df_local.rename(
        columns={
            "分隊": "division",
            "消防單位": "unit",
            "地址": "address",
            "電話": "phone",
            "緯度": "lat",
            "經度": "lon",
        }
    )

    text_columns = ["division", "unit", "address", "phone"]
    for col in text_columns:
        if col not in station_df_local.columns:
            station_df_local[col] = ""
        station_df_local[col] = (
            station_df_local[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    station_df_local["lat"] = pd.to_numeric(
        station_df_local.get("lat"), errors="coerce"
    )
    station_df_local["lon"] = pd.to_numeric(
        station_df_local.get("lon"), errors="coerce"
    )
    station_df_local = station_df_local.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if station_df_local.empty:
        warnings.append("沒有可顯示的消防隊資料。")
        return pd.DataFrame(), warnings

    expected_cols = ["division", "unit", "address", "phone", "lat", "lon"]
    for col in expected_cols:
        if col not in station_df_local.columns:
            station_df_local[col] = ""
    station_df_local = station_df_local[expected_cols]
    return station_df_local, warnings


def _build_hospital_layers(
    *,
    hospital_df: pd.DataFrame | None = None,
    include_hospitals: bool = True,
    include_isochrones: bool = True,
    isochrone_minutes: Sequence[int] = ISOCHRONE_DEFAULT_MINUTES,
) -> tuple[MacroElement | None, List[folium.FeatureGroup], Dict[int, folium.FeatureGroup], List[str]]:
    """Build hospital marker and isochrone feature groups without adding them to the map."""
    if not include_hospitals and not include_isochrones:
        return None, [], {}, []

    hospital_df_local, warnings = _prepare_hospital_dataframe(hospital_df)
    if hospital_df_local.empty:
        return None, [], {}, warnings

    valid_minutes: set[int] = set()
    for minute in isochrone_minutes:
        try:
            minute_value = int(minute)
        except (TypeError, ValueError):
            continue
        if minute_value > 0:
            valid_minutes.add(minute_value)
    valid_minutes &= set(ISOCHRONE_SUPPORTED_MINUTES)
    normalized_minutes = tuple(sorted(valid_minutes))
    if not normalized_minutes:
        include_isochrones = False
        normalized_minutes = ISOCHRONE_DEFAULT_MINUTES
    render_minutes = tuple(sorted(set(normalized_minutes), reverse=True))

    legend = _build_hospital_legend() if include_hospitals else None

    hospital_groups: Dict[str, folium.FeatureGroup] = {}
    hospital_group_counts: Dict[str, int] = {}
    if include_hospitals:
        for level in HOSPITAL_LEVEL_STYLES:
            hospital_groups[level] = folium.FeatureGroup(name=f"{level}等級責任醫院", show=True)
            hospital_group_counts[level] = 0
        hospital_groups["__default__"] = folium.FeatureGroup(name="其他責任醫院", show=True)
        hospital_group_counts["__default__"] = 0

    iso_layers: Dict[int, folium.FeatureGroup] = {}
    unionized_isochrones: Dict[int, List[Dict[str, Any]]] = {}
    if include_isochrones:
        union_data = _load_unionized_isochrones()
        if not union_data:
            warnings.append(
                "尚未找到預先建立的等時線資料，請先執行 `scripts/cache_isochrones.py` 與 "
                "`scripts/simplify_isochrones.py` 產生快取。"
            )
            include_isochrones = False
        else:
            unionized_isochrones = {
                minute: union_data.get(minute, [])
                for minute in render_minutes
                if union_data.get(minute)
            }

    for _, row in hospital_df_local.iterrows():
        level = row["level"] or ""
        style = HOSPITAL_LEVEL_STYLES.get(level, DEFAULT_HOSPITAL_STYLE)

        if include_hospitals:
            group_key = level if level in hospital_groups else "__default__"
            target_group = hospital_groups[group_key]

            icon_html = (
                '<div style="display:flex;align-items:center;justify-content:center;'
                f'width:32px;height:32px;border-radius:50%;background:{style["color"]};'
                'border:2px solid #ffffff;box-shadow:0 0 6px rgba(0,0,0,0.25);">'
                f'<span style="font-size:18px;line-height:1;">{style["emoji"]}</span>'
                "</div>"
            )
            popup_html = f"""
                <div style="width:220px;">
                    <h4 style="margin:0 0 6px 0;font-size:15px;">{row['name']}</h4>
                    <table style="width:100%;font-size:13px;">
                        <tr><td style="width:72px;"><b>分級</b></td><td>{row['level'] or '—'}</td></tr>
                        <tr><td><b>縣市</b></td><td>{row['city'] or '—'}</td></tr>
                        <tr><td><b>區域</b></td><td>{row['region'] or '—'}</td></tr>
                        <tr><td><b>座標</b></td><td>{row['lat']:.5f}, {row['lon']:.5f}</td></tr>
                    </table>
                </div>
            """

            tooltip_level_indicator_dict = {
                "重度": "🔴",
                "中度": "🟡",
                "一般": "🟢",
            }
            

            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(html=icon_html, icon_size=(32, 32), icon_anchor=(16, 16)),
                tooltip=f"{tooltip_level_indicator_dict.get(row['level'], '🔘')} {row['name']}",
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(target_group)
            hospital_group_counts[group_key] += 1

    if include_isochrones:
        for minute, minute_features in unionized_isochrones.items():
            if not minute_features:
                continue
            layer = folium.FeatureGroup(
                name=f"{minute} 分等時線",
                show=True,
            )
            style_cfg = ISOCHRONE_STYLE_BY_MINUTE.get(
                minute,
                {"color": "#3182bd", "fillColor": "#9ecae1", "fillOpacity": 0.18},
            )
            for feature in minute_features:
                if "geometry" not in feature:
                    continue
                geojson = folium.GeoJson(
                    feature,
                    style_function=_geojson_style(style_cfg),
                    name=f"{minute} 分涵蓋區域",
                )
                geojson.add_child(folium.Tooltip(f"{minute} 分涵蓋區域"))
                geojson.add_to(layer)
            iso_layers[minute] = layer

    hospital_groups_list: List[folium.FeatureGroup] = []
    if include_hospitals:
        for level_key, group in hospital_groups.items():
            if hospital_group_counts.get(level_key, 0) > 0:
                hospital_groups_list.append(group)

    return legend, hospital_groups_list, iso_layers, warnings


def _build_fire_station_layers(
    *,
    station_df: pd.DataFrame | None = None,
    include_fire_stations: bool = True,
) -> tuple[List[folium.FeatureGroup], List[str]]:
    """Build fire station feature groups."""
    if not include_fire_stations:
        return [], []

    station_df_local, warnings = _prepare_fire_station_dataframe(station_df)
    if station_df_local.empty:
        return [], warnings

    divisions = list(dict.fromkeys(station_df_local["division"]))
    division_styles: Dict[str, str] = {}
    for idx, division in enumerate(divisions):
        division_styles[division] = FIRE_STATION_COLORS[idx % len(FIRE_STATION_COLORS)]

    division_groups: Dict[str, folium.FeatureGroup] = {}
    division_counts: Dict[str, int] = {}
    for division in divisions:
        group = folium.FeatureGroup(name=f"{division}消防隊", show=True)
        division_groups[division] = group
        division_counts[division] = 0

    for _, row in station_df_local.iterrows():
        division = row["division"] or "消防隊"
        color = division_styles.get(division, "#d95f0e")
        group = division_groups.setdefault(division, folium.FeatureGroup(name=f"{division}消防隊", show=True))
        division_counts[division] = division_counts.get(division, 0) + 1

        icon_html = (
            '<div style="display:flex;align-items:center;justify-content:center;'
            f'width:28px;height:28px;border-radius:50%;background:{color};'
            'border:2px solid #ffffff;box-shadow:0 0 6px rgba(0,0,0,0.25);">'
            f'<span style="font-size:17px;line-height:1;">{FIRE_STATION_EMOJI}</span>'
            "</div>"
        )
        popup_html = f"""
            <div style="width:220px;">
                <h4 style="margin:0 0 6px 0;font-size:15px;">{row['unit'] or row['division']}</h4>
                <table style="width:100%;font-size:13px;">
                    <tr><td style="width:72px;"><b>分隊</b></td><td>{row['division'] or '—'}</td></tr>
                    <tr><td><b>地址</b></td><td>{row['address'] or '—'}</td></tr>
                    <tr><td><b>電話</b></td><td>{row['phone'] or '—'}</td></tr>
                    <tr><td><b>座標</b></td><td>{row['lat']:.5f}, {row['lon']:.5f}</td></tr>
                </table>
            </div>
        """

        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(html=icon_html, icon_size=(28, 28), icon_anchor=(14, 14)),
            tooltip=f"{row['unit'] or row['division']}",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(group)

    fire_groups: List[folium.FeatureGroup] = []
    for division, group in division_groups.items():
        if division_counts.get(division, 0) > 0:
            fire_groups.append(group)

    return fire_groups, warnings

    # hospital_df_local, prep_warnings = _prepare_hospital_dataframe(hospital_df)
    # warnings.extend(prep_warnings)
    # if hospital_df_local.empty:
    #     return warnings

    # # Normalize minutes input
    # valid_minutes: set[int] = set()
    # for minute in isochrone_minutes:
    #     try:
    #         minute_value = int(minute)
    #     except (TypeError, ValueError):
    #         continue
    #     if minute_value > 0:
    #         valid_minutes.add(minute_value)
    # valid_minutes &= set(ISOCHRONE_SUPPORTED_MINUTES)
    # normalized_minutes = tuple(sorted(valid_minutes))
    # if not normalized_minutes:
    #     include_isochrones = False
    #     normalized_minutes = ISOCHRONE_DEFAULT_MINUTES
    # render_minutes = tuple(sorted(set(normalized_minutes), reverse=True))

    # if include_hospitals:
    #     map_obj.get_root().add_child(_build_hospital_legend())

    # level_groups: Dict[str, folium.FeatureGroup] = {}
    # level_counts: Dict[str, int] = {}
    # if include_hospitals:
    #     for level in HOSPITAL_LEVEL_STYLES:
    #         group = folium.FeatureGroup(name=f"{level}等級責任醫院", show=True)
    #         level_groups[level] = group
    #         level_counts[level] = 0
    #     default_group = folium.FeatureGroup(name="其他責任醫院", show=True)
    #     level_groups["__default__"] = default_group
    #     level_counts["__default__"] = 0

def _build_hospital_legend() -> MacroElement:
    """Create a legend element describing hospital level color coding."""
    legend_rows = []
    for level, style in HOSPITAL_LEVEL_STYLES.items():
        legend_rows.append(
            f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <div style="width:22px;height:22px;border-radius:50%;background:{style['color']};
                            display:flex;align-items:center;justify-content:center;
                            color:#ffffff;font-size:14px;">{style['emoji']}</div>
                <span style="font-size:12px;color:#222222;">{level}</span>
            </div>
            """
        )

    legend_rows.append(
        f"""
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:22px;height:22px;border-radius:50%;background:{DEFAULT_HOSPITAL_STYLE['color']};
                        display:flex;align-items:center;justify-content:center;
                        color:#ffffff;font-size:14px;">{DEFAULT_HOSPITAL_STYLE['emoji']}</div>
            <span style="font-size:12px;color:#222222;">其他</span>
        </div>
        """
    )

    template = Template(
        f"""
        {{% macro html(this, kwargs) %}}
        <div style="
            position: fixed;
            bottom: 28px;
            left: 28px;
            z-index: 9999;
            background: rgba(255, 255, 255, 0.92);
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            font-family: 'Arial', sans-serif;
        ">
            <div style="font-weight: 600; margin-bottom: 8px; font-size: 13px;">
                急救責任醫院分級
            </div>
            {''.join(legend_rows)}
        </div>
        {{% endmacro %}}
        """
    )
    macro = MacroElement()
    macro._template = template
    return macro


def _geojson_style(style: Dict[str, Any]):
    """Return a function compatible with Folium GeoJson style_function."""

    def fn(_: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "color": style.get("color", "#2c7fb8"),
            "weight": style.get("weight", 2),
            "fillOpacity": style.get("fillOpacity", 0.2),
            "fillColor": style.get("fillColor", style.get("color", "#2c7fb8")),
        }

    return fn


def geocode_address(address: str) -> tuple:
    """
    Geocode an address to get coordinates
    Returns (latitude, longitude) or None if failed
    """
    if not address or pd.isna(address):
        return None
    
    # Check if address contains a known district
    for district, coords in DISTRICT_COORDINATES.items():
        if district in str(address):
            return tuple(coords)
    
    # Try geocoding (commented out for performance, uncomment if needed)
    # try:
    #     geolocator = Nominatim(user_agent="bloodchain_app")
    #     location = geolocator.geocode(address, timeout=10)
    #     if location:
    #         return (location.latitude, location.longitude)
    # except (GeocoderTimedOut, Exception):
    #     pass
    
    return None


def create_heatmap(
    district_df: pd.DataFrame,
    *,
    hospital_df: pd.DataFrame | None = None,
    include_hospitals: bool = True,
    include_isochrones: bool = True,
    include_fire_stations: bool = False,
    isochrone_minutes: Sequence[int] = ISOCHRONE_DEFAULT_MINUTES,
) -> tuple[folium.Map, List[str]]:
    """
    Create an interactive heatmap with markers
    
    Args:
        district_df: Aggregated DataFrame with columns:
            - incident_district
            - case_count
            - critical_count
            - avg_response_seconds
    
    Returns:
        Tuple[folium.Map, List[str]]: map object and warnings encountered while building overlays.
    """
    warnings: List[str] = []

    # Create base map
    m = _create_base_map()

    required_columns = {'incident_district', 'case_count'}
    if not required_columns.issubset(district_df.columns):
        folium.LayerControl(collapsed=False).add_to(m)
        return m, warnings

    # Only consider districts with known coordinates
    df_geo = district_df[
        district_df['incident_district'].isin(DISTRICT_COORDINATES)
    ].copy()
    if df_geo.empty:
        folium.LayerControl(collapsed=False).add_to(m)
        return m, warnings

    df_geo['lat'] = df_geo['incident_district'].map(
        lambda d: DISTRICT_COORDINATES[d][0]
    )
    df_geo['lon'] = df_geo['incident_district'].map(
        lambda d: DISTRICT_COORDINATES[d][1]
    )
    df_geo['critical_ratio'] = df_geo.apply(
        lambda row: (row['critical_count'] / row['case_count'])
        if row['case_count']
        else 0.0,
        axis=1,
    )
    df_geo['avg_response_min'] = df_geo['avg_response_seconds'].apply(
        lambda v: (v / 60.0) if pd.notna(v) else None
    )

    heat_layer = None
    color_scale: LinearColormap | None = None
    circle_rows: List[Dict[str, Any]] = []
    max_count_root = 1.0
    heat_gradient = {0.3: '#74add1', 0.6: '#fdae61', 0.8: '#f46d43', 1.0: '#d73027'}

    heat_data = df_geo[['lat', 'lon', 'case_count']].values.tolist()
    if heat_data:
        heat_layer = HeatMap(
            heat_data,
            control=False,
            radius=35,
            blur=28,
            max_zoom=13,
            gradient=heat_gradient,
        )
        min_count = float(df_geo['case_count'].min())
        max_count = float(df_geo['case_count'].max())
        if min_count == max_count:
            min_count -= 1
            max_count += 1
        color_scale = LinearColormap(
            colors=["#31a354", "#006837", "#ffffb2", "#fe9929", "#d95f0e"],
            vmin=min_count,
            vmax=max_count,
            caption="行政區案件數",
        )
        circle_rows = df_geo.to_dict("records")
        max_count_root = math.sqrt(float(df_geo['case_count'].max())) if max_count > 0 else 1.0

    legend = None
    hospital_groups: List[folium.FeatureGroup] = []
    iso_layers: Dict[int, folium.FeatureGroup] = {}
    fire_station_groups: List[folium.FeatureGroup] = []
    if include_hospitals or include_isochrones:
        legend, hospital_groups, iso_layers, overlay_warnings = _build_hospital_layers(
            hospital_df=hospital_df,
            include_hospitals=include_hospitals,
            include_isochrones=include_isochrones,
            isochrone_minutes=isochrone_minutes,
        )
        warnings.extend(overlay_warnings)
        if legend:
            m.get_root().add_child(legend)
        for minute in sorted(iso_layers.keys(), reverse=True):
            iso_layers[minute].add_to(m)
    if include_fire_stations:
        fire_station_groups, fire_warnings = _build_fire_station_layers()
        warnings.extend(fire_warnings)

    if heat_layer:
        heat_layer.add_to(m)
    if color_scale:
        color_scale.add_to(m)
    if circle_rows and color_scale:
        for row in circle_rows:
            popup_html = """
            <div style=\"width: 260px;\">
                <h4 style=\"margin-bottom:6px;\">{district}</h4>
                <table style=\"width:100%;font-size:13px;\">
                    <tr><td><b>案件數：</b></td><td>{count:,}</td></tr>
                    <tr><td><b>危急案件比：</b></td><td>{critical_ratio:.1%}</td></tr>
                    <tr><td><b>平均反應時間：</b></td><td>{avg_response}</td></tr>
                </table>
            </div>
            """
            avg_response_disp = f"{row['avg_response_min']:.1f} 分" if row['avg_response_min'] is not None else "—"
            popup_html = popup_html.format(
                district=row['incident_district'],
                count=int(row['case_count']),
                critical_ratio=row['critical_ratio'],
                avg_response=avg_response_disp,
            )

            radius = (
                8 + 20 * (math.sqrt(row['case_count']) / max_count_root)
                if max_count_root
                else 10
            )
            fill_color = color_scale(float(row['case_count']))

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                color='#444444',
                weight=1.0,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.75,
                tooltip=f"{row['incident_district']}：{int(row['case_count']):,} 件",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(m)

            font_size = 12
            label_html = (
                '<div style="display:flex;align-items:center;justify-content:center;'
                'transform: translate(0%, 0%);'
                'width:100%;height:100%;">'
                '<span style="font-size:{fs}px;font-style:bold;color:#000000;'
                'padding:2px 6px;border-radius:12px;'
                '">{count}</span></div>'
            ).format(fs=int(font_size), count=f"{int(row['case_count']):,}")
            folium.map.Marker(
                location=[row['lat'], row['lon']],
                icon=folium.DivIcon(
                    html=label_html,
                    icon_size=(int(radius * 2), int(radius * 2)),
                    icon_anchor=(int(radius), int(radius)),
                ),
            ).add_to(m)

    for group in fire_station_groups:
        group.add_to(m)

    for group in hospital_groups:
        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m, warnings


def create_hospital_response_map(
    hospital_df: pd.DataFrame | None = None,
    *,
    include_isochrones: bool = True,
    isochrone_minutes: Sequence[int] = ISOCHRONE_DEFAULT_MINUTES,
) -> Tuple[folium.Map, List[str]]:
    """
    Create a map visualising emergency responsibility hospitals and travel-time isochrones.

    Returns the constructed Folium map and a list of warning messages (if any).
    """
    hospital_map = _create_base_map()
    legend, hospital_groups, iso_layers, warnings = _build_hospital_layers(
        hospital_df=hospital_df,
        include_hospitals=True,
        include_isochrones=include_isochrones,
        isochrone_minutes=isochrone_minutes,
    )
    if legend:
        hospital_map.get_root().add_child(legend)
    for minute in sorted(iso_layers.keys(), reverse=True):
        iso_layers[minute].add_to(hospital_map)
    for group in hospital_groups:
        group.add_to(hospital_map)
    folium.LayerControl(collapsed=False).add_to(hospital_map)
    return hospital_map, warnings


def create_time_animation_map(daily_df: pd.DataFrame) -> go.Figure:
    """
    Animated map where bubble size encodes case count by day+district and
    color encodes critical ratio.
    """
    df_anim = daily_df.copy()
    required_columns = {'date', 'incident_district', 'case_count', 'critical_count'}
    if not required_columns.issubset(df_anim.columns):
        fig = go.Figure()
        fig.update_layout(title="缺少日期或行政區欄位", height=600)
        return fig

    df_anim['date'] = pd.to_datetime(df_anim['date'], errors='coerce')
    df_anim = df_anim[df_anim['date'].notna()]
    if df_anim.empty:
        fig = go.Figure()
        fig.update_layout(title="無可用的地理位置資料", height=600)
        return fig

    df_anim['date_str'] = df_anim['date'].dt.date.astype(str)
    base_counts = df_anim['case_count'].replace({0: pd.NA})
    df_anim['critical_ratio'] = (
        df_anim['critical_count'] / base_counts
    ).fillna(0.0)
    df_anim['lat'] = df_anim['incident_district'].map(
        lambda d: DISTRICT_COORDINATES.get(d, [None, None])[0]
    )
    df_anim['lon'] = df_anim['incident_district'].map(
        lambda d: DISTRICT_COORDINATES.get(d, [None, None])[1]
    )
    map_df = (
        df_anim.dropna(subset=['lat', 'lon'])
        .copy()
        .sort_values('date')
    )
    if map_df.empty:
        fig = go.Figure()
        fig.update_layout(title="無可用的地理位置資料", height=600)
        return fig

    fig = px.scatter_mapbox(
        map_df,
        lat='lat',
        lon='lon',
        size='case_count',
        size_max=48,
        color='critical_ratio',
        color_continuous_scale='YlOrRd',
        range_color=(0, 1),
        hover_name='incident_district',
        hover_data={'case_count': True, 'critical_ratio': ':.2f', 'lat': False, 'lon': False},
        animation_frame='date_str',
        zoom=10,
        center={'lat': DEFAULT_MAP_CENTER[0], 'lon': DEFAULT_MAP_CENTER[1]},
        mapbox_style='carto-positron',
        height=600,
        title='急救案件時間序列動畫（氣泡=案件數，顏色=危急比率）',
    )

    fig.update_layout(hovermode='closest', coloraxis_colorbar=dict(title='危急比率'))

    if fig.layout.updatemenus and fig.layout.updatemenus[0].buttons:
        button_args = fig.layout.updatemenus[0].buttons[0].args
        if len(button_args) > 1 and 'frame' in button_args[1] and 'transition' in button_args[1]:
            button_args[1]['frame']['duration'] = 500
            button_args[1]['transition']['duration'] = 300

    return fig


def create_hex_density_map(
    df: pd.DataFrame,
    *,
    resolution: int = 8,
    show_3d: bool = True,
):
    """Create a hexagon density map using pydeck + H3.

    Aggregates points into H3 hexagons to dramatically reduce payload size.
    Returns a pydeck.Deck object if dependencies are available; otherwise None.
    """
    try:
        import pydeck as pdk  # type: ignore
        import h3  # type: ignore
        import numpy as np  # noqa: F401
    except Exception:
        return None

    # Prefer explicit lat/lon if available, otherwise map by administrative district
    if {'lat', 'lon'}.issubset(df.columns):
        lat = pd.to_numeric(df['lat'], errors='coerce')
        lon = pd.to_numeric(df['lon'], errors='coerce')
    elif {'latitude', 'longitude'}.issubset(df.columns):
        lat = pd.to_numeric(df['latitude'], errors='coerce')
        lon = pd.to_numeric(df['longitude'], errors='coerce')
    else:
        coords_series = df.get('incident_district', pd.Series(dtype=object)).map(DISTRICT_COORDINATES)
        if coords_series is None or coords_series.empty:
            return None
        lat = coords_series.apply(lambda v: float(v[0]) if isinstance(v, (list, tuple)) else float('nan'))
        lon = coords_series.apply(lambda v: float(v[1]) if isinstance(v, (list, tuple)) else float('nan'))
    points = pd.DataFrame({'lat': lat, 'lon': lon}).dropna()
    if points.empty:
        return None

    points = points.astype({'lat': 'float32', 'lon': 'float32'})
    points['hex'] = points.apply(lambda r: h3.geo_to_h3(float(r.lat), float(r.lon), resolution), axis=1)

    agg = points.groupby('hex').size().reset_index(name='count')
    max_cnt = max(1, int(agg['count'].max()))
    agg['norm'] = (agg['count'] / float(max_cnt)).clip(0.0, 1.0)

    def to_color(p: float) -> list[int]:
        # clamp 0..1 and map to YlOrRd-like gradient with alpha
        p = 0 if p < 0 else (1 if p > 1 else p)
        if p < 0.5:
            t = p / 0.5
            r = int(255 * (1 - t) + 254 * t)
            g = int(255 * (1 - t) + 178 * t)
            b = int(178 * (1 - t) + 76 * t)
        else:
            t = (p - 0.5) / 0.5
            r = int(254 * (1 - t) + 240 * t)
            g = int(178 * (1 - t) + 59 * t)
            b = int(76 * (1 - t) + 32 * t)
        return [r, g, b, 200]

    agg['color'] = agg['norm'].apply(to_color)

    elev_ref = float(agg['count'].quantile(0.95)) if not agg.empty else 1.0
    elev_scale = 40.0 / elev_ref if elev_ref > 0 else 10.0

    layer = pdk.Layer(
        'H3HexagonLayer',
        data=agg,
        get_hexagon='hex',
        get_fill_color='color',
        get_elevation='count',
        elevation_scale=elev_scale,
        opacity=0.9,
        coverage=0.95,
        extruded=bool(show_3d),
        pickable=True,
        get_line_color=[60, 60, 60],
        line_width_min_pixels=1,
    )

    tooltip = {
        'html': '<b>案件數:</b> {count}',
        'style': {'backgroundColor': 'steelblue', 'color': 'white'},
    }

    view_state = pdk.ViewState(
        latitude=float(DEFAULT_MAP_CENTER[0]),
        longitude=float(DEFAULT_MAP_CENTER[1]),
        zoom=float(DEFAULT_MAP_ZOOM),
        pitch=40 if show_3d else 0,
        bearing=0,
    )

    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='carto-positron', tooltip=tooltip)
    return deck


def create_deck_heatmap(
    df: pd.DataFrame,
    *,
    radius_pixels: int = 60,
    intensity: float = 1.0,
):
    """Full-resolution heatmap using pydeck's HeatmapLayer.

    Optimized for large datasets; renders all points without server-side sampling.
    Returns pydeck.Deck or None if dependencies missing.
    """
    try:
        import pydeck as pdk  # type: ignore
    except Exception:
        return None

    # Prefer explicit lat/lon if available, otherwise map by administrative district
    if {'lat', 'lon'}.issubset(df.columns):
        lat = pd.to_numeric(df['lat'], errors='coerce')
        lon = pd.to_numeric(df['lon'], errors='coerce')
    elif {'latitude', 'longitude'}.issubset(df.columns):
        lat = pd.to_numeric(df['latitude'], errors='coerce')
        lon = pd.to_numeric(df['longitude'], errors='coerce')
    else:
        coords_series = df.get('incident_district', pd.Series(dtype=object)).map(DISTRICT_COORDINATES)
        if coords_series is None or coords_series.empty:
            return None
        lat = coords_series.apply(lambda v: float(v[0]) if isinstance(v, (list, tuple)) else float('nan'))
        lon = coords_series.apply(lambda v: float(v[1]) if isinstance(v, (list, tuple)) else float('nan'))
    points = pd.DataFrame({'lat': lat, 'lon': lon}).dropna()
    if points.empty:
        return None

    layer = pdk.Layer(
        'HeatmapLayer',
        data=points,
        get_position='[lon, lat]',
        aggregation='SUM',
        radius_pixels=radius_pixels,
        intensity=intensity,
        color_range=[
            [255, 255, 204],
            [255, 237, 160],
            [254, 217, 118],
            [254, 178, 76],
            [253, 141, 60],
            [240, 59, 32],
        ],
    )

    view_state = pdk.ViewState(
        latitude=float(DEFAULT_MAP_CENTER[0]),
        longitude=float(DEFAULT_MAP_CENTER[1]),
        zoom=float(DEFAULT_MAP_ZOOM),
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='carto-positron')
