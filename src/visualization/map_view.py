"""
Interactive map visualizations for emergency cases
"""
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import random

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

# Performance guardrails
# Limit the number of markers (popups are heavy) and
# sample heatmap points to keep payload under Streamlit limits
MAX_MARKERS: int = 5_000
HEATMAP_SAMPLE: int = 150_000
RANDOM_SEED: int = 42


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


def create_heatmap(df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> folium.Map:
    """
    Create an interactive heatmap with markers
    
    Args:
        df: DataFrame with emergency cases
        filters: Optional filters dict
    
    Returns:
        folium.Map object
    """
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
    
    # Prepare location data
    locations: list[list[float]] = []
    marker_data: list[Dict[str, Any]] = []
    
    random.seed(RANDOM_SEED)
    for idx, row in df.iterrows():
        # Try to get coordinates
        coords = None
        
        # First try from district
        district = row.get('incident_district')
        if district and district in DISTRICT_COORDINATES:
            coords = DISTRICT_COORDINATES[district]
        else:
            # Try geocoding actual address
            address = row.get('actual_address') or row.get('case_address')
            coords = geocode_address(address)
        
        if coords:
            # Append coordinates for heatmap (full set; will sample below)
            locations.append([float(coords[0]), float(coords[1])])

            # Only accumulate marker info up to MAX_MARKERS
            if len(marker_data) < MAX_MARKERS:
                marker_data.append({
                    'coords': [float(coords[0]), float(coords[1])],
                    'case_number': row.get('case_number', 'N/A'),
                    'date': str(row.get('date', 'N/A')),
                    'dispatch_reason': row.get('dispatch_reason', 'N/A'),
                    'district': row.get('incident_district', 'N/A'),
                    'triage_level': row.get('triage_level', 'N/A'),
                    'hospital': row.get('destination_hospital', 'N/A'),
                    'critical': bool(row.get('critical_case', False))
                })
    
    # Add heatmap layer
    if locations:
        # Sample heatmap points to cap payload size
        if len(locations) > HEATMAP_SAMPLE:
            locations = random.sample(locations, HEATMAP_SAMPLE)

        HeatMap(
            locations,
            radius=18,
            blur=28,
            max_zoom=13,
            gradient={0.4: '#74add1', 0.6: '#fdae61', 0.8: '#f46d43', 1.0: '#d73027'}
        ).add_to(m)

        # Add marker cluster only when under threshold
        if marker_data:
            marker_cluster = MarkerCluster(name="案件標記").add_to(m)

            for data in marker_data:
                icon_color = 'red' if data['critical'] else 'blue'
                popup_html = f"""
                <div style=\"width: 260px;\">\n                    <h4 style=\"margin-bottom:6px;\">案件 {data['case_number']}</h4>\n                    <table style=\"width:100%;font-size:13px;\">\n                        <tr><td><b>日期：</b></td><td>{data['date']}</td></tr>\n                        <tr><td><b>派遣原因：</b></td><td>{data['dispatch_reason']}</td></tr>\n                        <tr><td><b>行政區：</b></td><td>{data['district']}</td></tr>\n                        <tr><td><b>檢傷分級：</b></td><td>{data['triage_level']}</td></tr>\n                        <tr><td><b>後送醫院：</b></td><td>{data['hospital']}</td></tr>\n                        <tr><td><b>危急個案：</b></td><td>{'是' if data['critical'] else '否'}</td></tr>\n                    </table>\n                </div>
                """

                folium.Marker(
                    location=data['coords'],
                    popup=folium.Popup(popup_html, max_width=320),
                    icon=folium.Icon(color=icon_color, icon='ambulance', prefix='fa')
                ).add_to(marker_cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def create_time_animation_map(df: pd.DataFrame) -> go.Figure:
    """
    Create an animated map showing cases over time with clustering
    
    Args:
        df: DataFrame with emergency cases
    
    Returns:
        plotly Figure object with animation
    """
    # Prepare data with coordinates
    map_data = []
    
    for idx, row in df.iterrows():
        district = row.get('incident_district')
        if district and district in DISTRICT_COORDINATES:
            coords = DISTRICT_COORDINATES[district]
            
            # Parse date
            date = row.get('date')
            if pd.notna(date):
                if isinstance(date, str):
                    try:
                        date = pd.to_datetime(date)
                    except:
                        continue
                
                map_data.append({
                    'lat': coords[0],
                    'lon': coords[1],
                    'date': date,
                    'date_str': date.strftime('%Y-%m-%d'),
                    'case_number': row.get('case_number', 'N/A'),
                    'dispatch_reason': row.get('dispatch_reason', 'N/A'),
                    'district': district,
                    'triage_level': row.get('triage_level', 'N/A'),
                    'critical': row.get('critical_case', False)
                })
    
    if not map_data:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="無可用的地理位置資料",
            height=600
        )
        return fig
    
    # Create DataFrame
    map_df = pd.DataFrame(map_data)
    map_df = map_df.sort_values('date')
    
    # Create animated scatter mapbox
    fig = px.scatter_mapbox(
        map_df,
        lat='lat',
        lon='lon',
        animation_frame='date_str',
        hover_name='case_number',
        hover_data={
            'dispatch_reason': True,
            'district': True,
            'triage_level': True,
            'lat': False,
            'lon': False,
            'date_str': False
        },
        color='critical',
        color_discrete_map={True: '#d73027', False: '#4575b4'},
        labels={'critical': '危急個案'},
        zoom=10,
        center={'lat': DEFAULT_MAP_CENTER[0], 'lon': DEFAULT_MAP_CENTER[1]},
        mapbox_style='carto-positron',
        height=600,
        title='急救案件時間序列動畫'
    )
    
    # Update layout
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update animation speed
    if fig.layout.updatemenus and fig.layout.updatemenus[0].buttons:
        button_args = fig.layout.updatemenus[0].buttons[0].args
        if len(button_args) > 1 and "frame" in button_args[1] and "transition" in button_args[1]:
            button_args[1]["frame"]["duration"] = 500
            button_args[1]["transition"]["duration"] = 300
    
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

    # Build coordinates from district centers
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
        # Blue -> Red gradient
        r = int(69 + (215 - 69) * p)
        g = int(117 + (53 - 117) * p)
        b = int(180 + (39 - 180) * p)
        return [r, g, b]

    agg['color'] = agg['norm'].apply(to_color)

    layer = pdk.Layer(
        'H3HexagonLayer',
        data=agg,
        get_hexagon='hex',
        get_fill_color='color',
        get_elevation='count',
        elevation_scale=10,
        extruded=bool(show_3d),
        pickable=True,
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
