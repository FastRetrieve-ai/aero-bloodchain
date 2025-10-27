"""Streamlit page for UAV straight-line elevation profile analysis."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error, parse, request

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_MAP_CENTER,
    DEFAULT_MAP_ZOOM,
    DTM_SAMPLE_INTERVAL_M,
    DTM_TIF_PATH,
    GEOCODER_RATE_LIMIT_SECONDS,
    GEOCODER_TIMEOUT,
    GOOGLE_GEOCODE_ENDPOINT,
    GOOGLE_MAPS_API_KEY,
)
from geo.elevation_profile import compute_metrics, sample_profile

try:
    from streamlit_folium import st_folium
except ImportError:  # pragma: no cover - handled at runtime
    st.error("缺少 streamlit-folium 套件，請先安裝後再試一次。")
    st.stop()


MAP_POINTS_KEY = "uav_profile_points"
RESULT_KEY = "uav_profile_result"
LAST_GEOCODE_TS_KEY = "uav_last_geocode_ts"
COORD_SELECTION_KEY = "uav_coord_selection"
ADDRESS_SELECTION_KEY = "uav_address_selection"


def _get_dtm_path() -> Path:
    return Path(DTM_TIF_PATH)


def _build_geocode_url(address: str) -> str:
    params = {
        "address": address,
        "key": GOOGLE_MAPS_API_KEY,
    }
    return f"{GOOGLE_GEOCODE_ENDPOINT}?{parse.urlencode(params)}"


def _geocode(address: str) -> Optional[Tuple[float, float]]:
    cleaned = address.strip()
    if not cleaned:
        return None

    if not GOOGLE_MAPS_API_KEY:
        st.error("尚未設定 GOOGLE_MAPS_API_KEY，無法使用地址地理編碼功能。")
        return None

    now = time.time()
    last_ts = st.session_state.get(LAST_GEOCODE_TS_KEY)
    if last_ts and now - last_ts < GEOCODER_RATE_LIMIT_SECONDS:
        st.warning("請稍候片刻再進行下一次地理編碼，以避免觸發服務限制。")
        return None

    url = _build_geocode_url(cleaned)
    try:
        with request.urlopen(url, timeout=float(GEOCODER_TIMEOUT or 5)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        st.error(f"Google 地理編碼 API 回傳錯誤 {exc.code}: {exc.reason}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"呼叫 Google 地理編碼 API 失敗：{exc}")
        return None

    status = payload.get("status")
    if status != "OK":
        st.warning(f"無法定位地址：{cleaned}（狀態：{status}）")
        return None

    results = payload.get("results") or []
    if not results:
        st.warning(f"找不到地址：{cleaned}")
        return None

    location = results[0].get("geometry", {}).get("location", {})
    lat = location.get("lat")
    lng = location.get("lng")
    if lat is None or lng is None:
        st.warning("地理編碼回傳結果缺少經緯度，請重新嘗試。")
        return None

    st.session_state[LAST_GEOCODE_TS_KEY] = time.time()
    return float(lat), float(lng)


def _parse_coordinate(value: str, *, min_value: float, max_value: float) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (min_value <= number <= max_value):
        return None
    return number


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        return "—"
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} 小時 {minutes:02d} 分 {sec:02d} 秒"
    if minutes:
        return f"{minutes} 分 {sec:02d} 秒"
    return f"{sec} 秒"


def _display_results(profile: pd.DataFrame, metrics: dict) -> None:
    if profile.empty:
        st.info("沒有可用的剖面資料。")
        return

    distance_km = metrics.get("horizontal_distance_m", 0.0) / 1000
    vertical_gain = metrics.get("vertical_gain_m", 0.0)
    vertical_loss = metrics.get("vertical_loss_m", 0.0)
    eta_seconds = metrics.get("eta_seconds", math.inf)
    path_length = metrics.get("additive_path_length_m", 0.0) / 1000
    path_length_3d = metrics.get("path_length_3d_m", 0.0) / 1000

    metric_cols = st.columns(4)
    metric_cols[0].metric("水平距離", f"{distance_km:.2f} 公里")
    metric_cols[1].metric("總爬升", f"{vertical_gain:.0f} 公尺")
    metric_cols[2].metric("總下降", f"{vertical_loss:.0f} 公尺")
    metric_cols[3].metric("預估飛行時間", _format_duration(eta_seconds))

    st.caption(
        f"3D 路徑長度 ≈ {path_length_3d:.2f} 公里；水平距離 + 爬升 ≈ {path_length:.2f} 公里"
    )

    x_values = profile["distance_m"].to_numpy() / 1000
    y_values = profile["elevation_m"].to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name="地形剖面",
            line=dict(color="#D62728", width=3),
            hovertemplate="距離 %{x:.2f} km<br>高度 %{y:.1f} m<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.2)",
        )
    )

    valid_profile = profile[np.isfinite(profile["elevation_m"])]
    if not valid_profile.empty:
        idx_max = valid_profile["elevation_m"].idxmax()
        idx_min = valid_profile["elevation_m"].idxmin()
        max_row = valid_profile.loc[idx_max]
        min_row = valid_profile.loc[idx_min]
        fig.add_trace(
            go.Scatter(
                x=[max_row["distance_m"] / 1000],
                y=[max_row["elevation_m"]],
                mode="markers",
                marker=dict(color="#2CA02C", size=10),
                name="最高點",
                hovertemplate="距離 %{x:.2f} km<br>高度 %{y:.1f} m<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_row["distance_m"] / 1000],
                y=[min_row["elevation_m"]],
                mode="markers",
                marker=dict(color="#1F77B4", size=10),
                name="最低點",
                hovertemplate="距離 %{x:.2f} km<br>高度 %{y:.1f} m<extra></extra>",
            )
        )

    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="距離 (公里)",
        yaxis_title="地面高度 (公尺)",
        hovermode="x unified",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    if not bool(profile["is_valid"].all()):
        st.warning("路徑部分點超出 DTM 範圍或為缺值，已以 NaN 顯示。")

    display_df = profile.copy()
    display_df["distance_km"] = display_df["distance_m"] / 1000
    display_df = display_df[["lat", "lon", "distance_km", "elevation_m", "is_valid"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = profile.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="下載剖面 CSV",
        data=csv_bytes,
        file_name="uav_elevation_profile.csv",
        mime="text/csv",
    )


def _render_preview_map(
    points: List[Tuple[float, float]],
    *,
    map_key: str,
    caption: Optional[str] = None,
) -> None:
    valid_points = [point for point in points if point]
    if not valid_points:
        return

    if caption:
        st.markdown(f"**{caption}**")

    if len(valid_points) == 1:
        center = valid_points[0]
    else:
        center = (
            sum(lat for lat, _ in valid_points) / len(valid_points),
            sum(lon for _, lon in valid_points) / len(valid_points),
        )

    preview_map = folium.Map(location=center, zoom_start=DEFAULT_MAP_ZOOM)
    for idx, (lat, lon) in enumerate(valid_points, start=1):
        folium.Marker(
            location=(lat, lon),
            tooltip=f"點 {idx}",
            icon=folium.Icon(color="red" if idx == 1 else "blue"),
        ).add_to(preview_map)

    if len(valid_points) >= 2:
        folium.PolyLine(valid_points[:2], color="orange", weight=4).add_to(preview_map)

    st_folium(
        preview_map,
        height=360,
        use_container_width=True,
        returned_objects=[],
        key=map_key,
    )


def _render_map_picker() -> List[Tuple[float, float]]:
    points = list(st.session_state.get(MAP_POINTS_KEY, []))

    map_col, control_col = st.columns([3, 1])
    with control_col:
        st.markdown("### 地圖操作")
        st.caption("在地圖上點擊兩個位置以設定起、終點。")
        if st.button("清除選點", use_container_width=True):
            st.session_state[MAP_POINTS_KEY] = []
            st.rerun()

        if points:
            st.markdown("#### 已選座標")
            for idx, (lat, lon) in enumerate(points, start=1):
                st.code(f"點 {idx}: {lat:.6f}, {lon:.6f}")

    with map_col:
        folium_map = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM)
        if points:
            for idx, (lat, lon) in enumerate(points, start=1):
                folium.Marker(
                    location=(lat, lon),
                    tooltip=f"點 {idx}",
                    icon=folium.Icon(color="red" if idx == 1 else "blue"),
                ).add_to(folium_map)
            if len(points) == 2:
                folium.PolyLine(points, color="orange", weight=4).add_to(folium_map)

        folium.LatLngPopup().add_to(folium_map)
        map_event = st_folium(
            folium_map,
            height=520,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="uav_elevation_profile_map",
        )

    if map_event and map_event.get("last_clicked"):
        lat = map_event["last_clicked"].get("lat")
        lon = map_event["last_clicked"].get("lng")
        if lat is not None and lon is not None:
            new_point = (float(lat), float(lon))
            if not points or (abs(points[-1][0] - new_point[0]) > 1e-6 or abs(points[-1][1] - new_point[1]) > 1e-6):
                points.append(new_point)
                if len(points) > 2:
                    points = points[-2:]
                st.session_state[MAP_POINTS_KEY] = points
                st.rerun()

    return list(st.session_state.get(MAP_POINTS_KEY, []))


def render_uav_elevation_profile_page() -> None:
    """Render the UAV elevation profile analysis page."""

    st.title("✈️ 無人機直線飛行剖面分析")
    st.markdown("利用 20 公尺 DTM 估算無人機直線航線的高度變化、距離與預估飛行時間。")

    dtm_path = _get_dtm_path()
    if not dtm_path.exists():
        st.error(
            f"找不到 DTM GeoTIFF：{dtm_path}. 請確認 `data/` 目錄是否包含 2024 年 20m DTM。"
        )
        return

    input_mode = st.radio(
        "選擇輸入模式",
        options=["輸入座標", "輸入地址", "地圖選擇"],
        horizontal=True,
        key="uav_profile_input_mode",
    )

    start_point: Optional[Tuple[float, float]] = None
    end_point: Optional[Tuple[float, float]] = None
    errors: List[str] = []

    sample_interval = st.number_input(
        "取樣間距 (公尺)",
        min_value=10.0,
        max_value=200.0,
        value=float(DTM_SAMPLE_INTERVAL_M),
        step=10.0,
    )

    if input_mode == "輸入座標":
        st.markdown("#### 輸入起終點座標 (WGS84)")

        stored_coords: Dict[str, Tuple[float, float]] | None = st.session_state.get(
            COORD_SELECTION_KEY
        )
        if stored_coords:
            st.session_state.setdefault(
                "uav_coord_start_lat",
                f"{stored_coords['start'][0]:.6f}",
            )
            st.session_state.setdefault(
                "uav_coord_start_lon",
                f"{stored_coords['start'][1]:.6f}",
            )
            st.session_state.setdefault(
                "uav_coord_end_lat",
                f"{stored_coords['end'][0]:.6f}",
            )
            st.session_state.setdefault(
                "uav_coord_end_lon",
                f"{stored_coords['end'][1]:.6f}",
            )

        st.markdown("##### 🏁 起點座標")
        start_lat_text = st.text_input(
            "🌐 緯度",
            placeholder="25.060833",
            key="uav_coord_start_lat",
        )
        start_lon_text = st.text_input(
            "🌍 經度",
            placeholder="121.490556",
            key="uav_coord_start_lon",
        )

        st.markdown("##### 🎯 終點座標")
        end_lat_text = st.text_input(
            "🌐 緯度",
            placeholder="25.171944",
            key="uav_coord_end_lat",
        )
        end_lon_text = st.text_input(
            "🌍 經度",
            placeholder="121.679444",
            key="uav_coord_end_lon",
        )

        if st.button("🌐 套用座標", key="apply_coords", type="secondary", use_container_width=True):
            start_lat = _parse_coordinate(start_lat_text, min_value=-90, max_value=90)
            start_lon = _parse_coordinate(start_lon_text, min_value=-180, max_value=180)
            end_lat = _parse_coordinate(end_lat_text, min_value=-90, max_value=90)
            end_lon = _parse_coordinate(end_lon_text, min_value=-180, max_value=180)

            if start_lat is None or start_lon is None:
                errors.append("起點座標格式錯誤，請輸入有效經緯度。")
            if end_lat is None or end_lon is None:
                errors.append("終點座標格式錯誤，請輸入有效經緯度。")
            if not errors:
                start_point = (start_lat, start_lon)
                end_point = (end_lat, end_lon)
                st.session_state[COORD_SELECTION_KEY] = {
                    "start": start_point,
                    "end": end_point,
                }
                st.session_state[MAP_POINTS_KEY] = [start_point, end_point]
                st.session_state[RESULT_KEY] = None
                st.success("座標已更新，請點選下方生成剖面。")
        stored_coords = st.session_state.get(COORD_SELECTION_KEY)
        if stored_coords:
            st.caption(
                f"目前選定：起點 {stored_coords['start'][0]:.5f}, {stored_coords['start'][1]:.5f} · "
                f"終點 {stored_coords['end'][0]:.5f}, {stored_coords['end'][1]:.5f}"
            )
            _render_preview_map(
                [stored_coords["start"], stored_coords["end"]],
                map_key="uav_coord_preview_map",
                caption="輸入座標預覽",
            )

    elif input_mode == "輸入地址":
        st.markdown("#### 📇 輸入地址並自動地理編碼")
        stored_addresses = st.session_state.get(ADDRESS_SELECTION_KEY)
        if stored_addresses:
            st.session_state.setdefault("uav_address_start", stored_addresses.get("start_text", ""))
            st.session_state.setdefault("uav_address_end", stored_addresses.get("end_text", ""))

        start_address = st.text_input(
            "起點地址",
            placeholder="例：新北市三重區新北大道一段3號",
            key="uav_address_start",
        )
        end_address = st.text_input(
            "終點地址",
            placeholder="例：新北市萬里區中福路",
            key="uav_address_end",
        )

        if st.button("📍 地理編碼地址", key="geocode_addresses", type="secondary", use_container_width=True):
            start_point = _geocode(start_address)
            if start_point and GEOCODER_RATE_LIMIT_SECONDS > 0:
                time.sleep(GEOCODER_RATE_LIMIT_SECONDS)
            end_point = _geocode(end_address)
            if start_point and end_point:
                st.session_state[ADDRESS_SELECTION_KEY] = {
                    "start": start_point,
                    "end": end_point,
                    "start_text": start_address,
                    "end_text": end_address,
                }
                st.session_state[MAP_POINTS_KEY] = [start_point, end_point]
                st.session_state[RESULT_KEY] = None
                st.success("地址已成功地理編碼，請點選下方生成剖面。")
            else:
                errors.append("無法解析其中一個地址，請確認輸入是否正確。")
        stored_addresses = st.session_state.get(ADDRESS_SELECTION_KEY)
        if stored_addresses:
            st.caption(
                "目前選定的地理座標："
                f"起點 {stored_addresses['start'][0]:.5f}, {stored_addresses['start'][1]:.5f} · "
                f"終點 {stored_addresses['end'][0]:.5f}, {stored_addresses['end'][1]:.5f}"
            )
            _render_preview_map(
                [stored_addresses["start"], stored_addresses["end"]],
                map_key="uav_address_preview_map",
                caption="地址地理編碼預覽",
            )

    else:
        st.markdown("#### 地圖選點")
        selected_points = _render_map_picker()
        if len(selected_points) == 2:
            start_point, end_point = selected_points

    speed_col, unit_col = st.columns([2, 1])
    with speed_col:
        speed_value = st.number_input("無人機平均速度", min_value=0.0, value=80.0, step=1.0)
    with unit_col:
        speed_unit = st.selectbox("速度單位", ["km/h", "m/s"], index=0)

    result_state = st.session_state.get(RESULT_KEY)

    if st.button("🚀 生成剖面分析", type="primary", use_container_width=True):
        if input_mode == "輸入座標":
            stored = st.session_state.get(COORD_SELECTION_KEY)
            if stored:
                start_point = stored.get("start")
                end_point = stored.get("end")
        elif input_mode == "輸入地址":
            stored = st.session_state.get(ADDRESS_SELECTION_KEY)
            if stored:
                start_point = stored.get("start")
                end_point = stored.get("end")

        if start_point is None or end_point is None:
            errors.append("請先設定起點與終點。")
        if start_point and end_point and start_point == end_point:
            errors.append("起點與終點相同，無法產生剖面。")

        if not errors:
            try:
                profile_df = sample_profile(
                    start_point,
                    end_point,
                    sample_interval_m=sample_interval,
                    dtm_path=dtm_path,
                )
                metrics = compute_metrics(profile_df, speed_value=speed_value, speed_unit=speed_unit)
                st.session_state[RESULT_KEY] = {
                    "profile": profile_df,
                    "metrics": metrics,
                }
                result_state = st.session_state[RESULT_KEY]
                st.success("剖面計算完成！")
            except FileNotFoundError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(f"DTM 設定錯誤：{exc}")
            except Exception as exc:  # pragma: no cover - unexpected runtime errors
                st.error(f"計算剖面時發生未知錯誤：{exc}")

    for err in errors:
        st.error(err)

    if result_state and isinstance(result_state, dict):
        profile_df = result_state.get("profile")
        metrics = result_state.get("metrics")
        if isinstance(profile_df, pd.DataFrame) and isinstance(metrics, dict):
            st.divider()
            _display_results(profile_df, metrics)
            st.caption(f"資料來源：地政司 / 2024年版全臺灣20公尺網格數值地形模型DTM資料")
