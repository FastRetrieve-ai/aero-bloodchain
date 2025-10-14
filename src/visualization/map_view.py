"""
Interactive map visualizations for emergency cases
"""
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import math

from config import DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM


# District coordinates for New Taipei City
DISTRICT_COORDINATES = {
    '板橋區': [25.0116, 121.4625],
    '新莊區': [25.0372, 121.4325],
    '中和區': [24.9994, 121.4991],
    '永和區': [25.0039, 121.5156],
    '土城區': [24.9733, 121.4420],
    '樹林區': [24.9906, 121.4201],
    '三峽區': [24.9342, 121.3697],
    '鶯歌區': [24.9545, 121.3538],
    '三重區': [25.0619, 121.4885],
    '蘆洲區': [25.0847, 121.4741],
    '五股區': [25.0829, 121.4384],
    '泰山區': [25.0572, 121.4301],
    '林口區': [25.0770, 121.3926],
    '淡水區': [25.1688, 121.4406],
    '金山區': [25.2217, 121.6370],
    '萬里區': [25.1797, 121.6891],
    '汐止區': [25.0672, 121.6423],
    '瑞芳區': [25.1089, 121.8058],
    '貢寮區': [25.0202, 121.9086],
    '平溪區': [25.0258, 121.7391],
    '雙溪區': [25.0347, 121.8656],
    '石碇區': [24.9959, 121.6582],
    '深坑區': [25.0023, 121.6165],
    '石門區': [25.2906, 121.5685],
    '八里區': [25.1448, 121.3967],
    '坪林區': [24.9360, 121.7108],
    '烏來區': [24.8656, 121.5498],
}



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
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    *,
    max_heatmap_points: Optional[int] = None,
) -> folium.Map:
    """
    Create an interactive heatmap with markers
    
    Args:
        df: DataFrame with emergency cases
        filters: Optional filters dict
    
    Returns:
        folium.Map object
    """
    # Parameter retained for backward compatibility (aggregation removes need for sampling)
    _ = max_heatmap_points
    # Apply filters if provided
    if filters:
        if 'start_date' in filters and 'date' in df.columns:
            df = df[df['date'] >= pd.to_datetime(filters['start_date'])]
        if 'end_date' in filters and 'date' in df.columns:
            df = df[df['date'] <= pd.to_datetime(filters['end_date'])]
        if 'district' in filters and filters['district'] and 'incident_district' in df.columns:
            df = df[df['incident_district'] == filters['district']]
        if 'dispatch_reasons' in filters and filters['dispatch_reasons'] and 'dispatch_reason' in df.columns:
            df = df[df['dispatch_reason'].isin(filters['dispatch_reasons'])]
        if 'triage_levels' in filters and filters['triage_levels'] and 'triage_level' in df.columns:
            df = df[df['triage_level'].isin(filters['triage_levels'])]
        if 'critical_only' in filters and filters['critical_only'] and 'critical_case' in df.columns:
            df = df[df['critical_case'] == True]
    
    # Create base map
    m = folium.Map(
        location=DEFAULT_MAP_CENTER,
        zoom_start=DEFAULT_MAP_ZOOM,
        tiles=None,
        control_scale=True
    )

    # Additional base layers for aesthetics
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Positron',
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    ).add_to(m)
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
        name='Stamen Toner',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, under ODbL.'
    ).add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen(position='topleft').add_to(m)
    
    # Prepare aggregated statistics by district to minimize payload
    if 'incident_district' not in df.columns:
        folium.LayerControl(collapsed=False).add_to(m)
        return m

    # Only consider districts with known coordinates
    df_geo = df[df['incident_district'].isin(DISTRICT_COORDINATES)].copy()
    if df_geo.empty:
        folium.LayerControl(collapsed=False).add_to(m)
        return m

    agg_df = df_geo.groupby('incident_district').agg(
        count=('incident_district', 'size'),
        critical_count=('critical_case', lambda s: int(pd.Series(s).fillna(False).astype(int).sum()) if s is not None else 0),
        avg_response=('response_time_seconds', 'mean'),
    ).reset_index()

    agg_df['lat'] = agg_df['incident_district'].map(lambda d: DISTRICT_COORDINATES[d][0])
    agg_df['lon'] = agg_df['incident_district'].map(lambda d: DISTRICT_COORDINATES[d][1])
    agg_df['critical_ratio'] = agg_df.apply(
        lambda row: (row['critical_count'] / row['count']) if row['count'] else 0.0,
        axis=1
    )
    agg_df['avg_response_min'] = agg_df['avg_response'].apply(lambda v: (v / 60.0) if pd.notna(v) else None)

    # Heatmap data uses weight per district instead of every case
    heat_data = agg_df[['lat', 'lon', 'count']].values.tolist()
    if heat_data:
        HeatMap(
            heat_data,
            radius=35,
            blur=28,
            max_zoom=13,
            gradient={0.3: '#74add1', 0.6: '#fdae61', 0.8: '#f46d43', 1.0: '#d73027'}
        ).add_to(m)

        # Add one marker per district summarizing key metrics
        max_count = agg_df['count'].max()
        for _, row in agg_df.iterrows():
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
                count=int(row['count']),
                critical_ratio=row['critical_ratio'],
                avg_response=avg_response_disp,
            )

            # Scale circle radius by square root to avoid huge values
            radius = 8 + 20 * math.sqrt(row['count'] / max_count) if max_count else 10

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                color='#2b8cbe',
                fill=True,
                fill_color='#2b8cbe',
                fill_opacity=0.65,
                tooltip=f"{row['incident_district']}：{int(row['count']):,} 件",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def create_time_animation_map(df: pd.DataFrame) -> go.Figure:
    """
    Animated map where bubble size encodes case count by day+district and
    color encodes critical ratio.
    """
    df_anim = df.copy()
    if 'date' not in df_anim.columns or 'incident_district' not in df_anim.columns:
        fig = go.Figure()
        fig.update_layout(title="缺少日期或行政區欄位", height=600)
        return fig

    df_anim['date'] = pd.to_datetime(df_anim['date'], errors='coerce')
    df_anim = df_anim[df_anim['date'].notna()]
    if df_anim.empty:
        fig = go.Figure()
        fig.update_layout(title="無可用的地理位置資料", height=600)
        return fig

    df_anim['date_only'] = df_anim['date'].dt.date
    df_anim['critical_case'] = df_anim.get('critical_case', False).fillna(False).astype(int)
    grouped = df_anim.groupby(['date_only', 'incident_district']).agg(
        count=('incident_district', 'size'),
        critical_count=('critical_case', 'sum'),
    ).reset_index()
    grouped['date_str'] = grouped['date_only'].astype(str)
    grouped['critical_ratio'] = (grouped['critical_count'] / grouped['count']).fillna(0.0)
    grouped['lat'] = grouped['incident_district'].map(lambda d: DISTRICT_COORDINATES.get(d, [None, None])[0])
    grouped['lon'] = grouped['incident_district'].map(lambda d: DISTRICT_COORDINATES.get(d, [None, None])[1])
    map_df = grouped.dropna(subset=['lat', 'lon']).copy().sort_values('date_only')
    if map_df.empty:
        fig = go.Figure()
        fig.update_layout(title="無可用的地理位置資料", height=600)
        return fig

    fig = px.scatter_mapbox(
        map_df,
        lat='lat',
        lon='lon',
        size='count',
        size_max=48,
        color='critical_ratio',
        color_continuous_scale='YlOrRd',
        range_color=(0, 1),
        hover_name='incident_district',
        hover_data={'count': True, 'critical_ratio': ':.2f', 'lat': False, 'lon': False},
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
