"""
Interactive map visualizations for emergency cases
"""
import pandas as pd
import folium
from folium.plugins import HeatMap, MiniMap, Fullscreen
from branca.colormap import LinearColormap
import plotly.express as px
import plotly.graph_objects as go
import math

from config import DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM


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
) -> folium.Map:
    """
    Create an interactive heatmap with markers
    
    Args:
        district_df: Aggregated DataFrame with columns:
            - incident_district
            - case_count
            - critical_count
            - avg_response_seconds
    
    Returns:
        folium.Map object
    """
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

    required_columns = {'incident_district', 'case_count'}
    if not required_columns.issubset(district_df.columns):
        folium.LayerControl(collapsed=False).add_to(m)
        return m

    # Only consider districts with known coordinates
    df_geo = district_df[
        district_df['incident_district'].isin(DISTRICT_COORDINATES)
    ].copy()
    if df_geo.empty:
        folium.LayerControl(collapsed=False).add_to(m)
        return m

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

    # Heatmap data uses weight per district instead of every case
    heat_data = df_geo[['lat', 'lon', 'case_count']].values.tolist()
    if heat_data:
        HeatMap(
            heat_data,
            radius=35,
            blur=28,
            max_zoom=13,
            gradient={0.3: '#74add1', 0.6: '#fdae61', 0.8: '#f46d43', 1.0: '#d73027'}
        ).add_to(m)

        # Color scale for circle markers (lighter for少量, darker for大量)
        min_count = float(df_geo['case_count'].min())
        max_count = float(df_geo['case_count'].max())
        if min_count == max_count:
            # Expand range slightly to avoid degenerate colormap
            min_count -= 1
            max_count += 1
        color_scale = LinearColormap(
            # colors=['#2b8cbe', '#7fcdbb', '#edf8b1', '#fc8d59', '#d7301f'],
            colors=["#31a354", "#006837", "#ffffb2", "#fe9929", "#d95f0e"],
            vmin=min_count,
            vmax=max_count,
            caption="行政區案件數",
        )
        color_scale.add_to(m)

        # Add one marker per district summarizing key metrics
        max_count_root = (
            math.sqrt(float(df_geo['case_count'].max())) if max_count > 0 else 1.0
        )
        for _, row in df_geo.iterrows():
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

            # Scale circle radius by square root to avoid極端差距
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

            # Add count label at circle center
            # font_size = max(11, min(22, radius * 1.1))
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

    folium.LayerControl(collapsed=False).add_to(m)
    return m


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
