"""
Statistical charts for emergency case data
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional
import numpy as np

from database.db_manager import DatabaseManager


def _build_histogram(
    values: pd.Series,
    *,
    title: str,
    x_label: str,
    color: str,
    min_bin_size: float,
    target_bins: int = 60,
) -> go.Figure | None:
    """Helper to create histogram with explicit bins and custom hover."""
    if values is None or values.empty:
        return None

    cleaned = values.astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    cleaned = cleaned[cleaned >= 0]
    if cleaned.empty:
        return None

    # Determine upper bound using 99th percentile to ignore outliers
    if len(cleaned) > 50:
        upper_bound = float(cleaned.quantile(0.99))
    else:
        upper_bound = float(cleaned.max())
    upper_bound = max(upper_bound, min_bin_size)

    # Compute bin size and edges
    bin_size = max(upper_bound / target_bins, min_bin_size)
    max_edge = np.ceil(upper_bound / bin_size) * bin_size
    edges = np.arange(0, max_edge + bin_size, bin_size)
    if len(edges) < 2:
        edges = np.array([0.0, bin_size])

    counts, edges = np.histogram(cleaned, bins=edges)
    if not counts.any():
        return None

    centers = edges[:-1] + np.diff(edges) / 2
    widths = np.diff(edges)
    custom = np.stack([edges[:-1], edges[1:]], axis=-1)

    fig = go.Figure(
        data=[
            go.Bar(
                x=centers,
                y=counts,
                width=widths * 0.9,
                marker_color=color,
                customdata=custom,
                hovertemplate=(
                    "範圍 %{customdata[0]:.1f}–%{customdata[1]:.1f} 分鐘"  # bin range
                    "<br>案件數 %{y:,}"  # formatted count
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="案件數",
        height=400,
        bargap=0.05,
    )
    fig.update_xaxes(range=[0, edges[-1]])

    return fig


def create_statistics_charts(
    db_manager: DatabaseManager,
    *,
    filters: Optional[Dict[str, object]] = None,
    district_summary: Optional[pd.DataFrame] = None,
) -> Dict[str, go.Figure]:
    """
    Create various statistical charts by querying the database on demand.

    Args:
        db_manager: Database access layer.
        filters: Optional filter dictionary shared with the map view.
        district_summary: Optional pre-computed district aggregation to avoid
            duplicate queries.

    Returns:
        Dictionary mapping chart keys to Plotly Figure objects.
    """
    charts: Dict[str, go.Figure] = {}

    # 1. Cases by District (Bar Chart)
    district_df = district_summary
    if district_df is None:
        district_df = db_manager.get_district_aggregates(filters)
    if district_df is not None and not district_df.empty:
        district_chart = (
            district_df[['incident_district', 'case_count']]
            .sort_values('case_count', ascending=False)
            .head(15)
            .rename(columns={'incident_district': '行政區', 'case_count': '案件數'})
        )
        fig_district = px.bar(
            district_chart,
            x='行政區',
            y='案件數',
            title='各行政區急救案件數量',
            labels={'行政區': '行政區', '案件數': '案件數'},
            color='案件數',
            color_continuous_scale='Reds',
        )
        fig_district.update_layout(xaxis_tickangle=-45, height=400)
        charts['district_bar'] = fig_district

    # Helper to safely extract hour component
    def _extract_hour(raw: object) -> Optional[int]:
        if pd.isna(raw):
            return None
        text = str(raw)
        if not text:
            return None
        hour_text = text.split(':', 1)[0]
        try:
            hour_value = int(hour_text)
        except ValueError:
            return None
        return hour_value if 0 <= hour_value <= 23 else None

    # 2. Cases by Time of Day (Line Chart)
    dispatch_df = db_manager.get_cases_dataframe(
        filters, columns=['dispatch_time']
    )
    if not dispatch_df.empty and 'dispatch_time' in dispatch_df.columns:
        hours = dispatch_df['dispatch_time'].apply(_extract_hour).dropna()
        if not hours.empty:
            hour_counts = (
                hours.astype(int)
                .value_counts()
                .sort_index()
                .reset_index()
            )
            hour_counts.columns = ['小時', '案件數']
            fig_time = px.line(
                hour_counts,
                x='小時',
                y='案件數',
                title='24小時急救案件分布',
                labels={'小時': '時段 (小時)', '案件數': '案件數'},
                markers=True,
            )
            fig_time.update_layout(height=400)
            charts['time_line'] = fig_time

    # 3. Response Time Distribution (Histogram)
    response_df = db_manager.get_cases_dataframe(
        filters, columns=['response_time_seconds']
    )
    if 'response_time_seconds' in response_df.columns:
        response_series = response_df['response_time_seconds'] / 60.0
        fig_response = _build_histogram(
            response_series,
            title='反應時間分布',
            x_label='反應時間 (分鐘)',
            color='#636EFA',
            min_bin_size=0.5,
        )
        if fig_response:
            charts['response_histogram'] = fig_response

    # 3b. Transport Time Distribution (送醫時間)
    transport_df = db_manager.get_cases_dataframe(
        filters, columns=['transport_time_seconds']
    )
    if 'transport_time_seconds' in transport_df.columns:
        transport_series = transport_df['transport_time_seconds'] / 60.0
        fig_transport = _build_histogram(
            transport_series,
            title='送醫時間分布',
            x_label='送醫時間 (分鐘)',
            color='#EF553B',
            min_bin_size=1.0,
        )
        if fig_transport:
            charts['transport_histogram'] = fig_transport

    # 4. Triage Level Distribution (Pie Chart)
    triage_df = db_manager.get_counts_by_field(
        'triage_level', filters=filters
    )
    if not triage_df.empty and 'triage_level' in triage_df.columns:
        triage_chart = (
            triage_df.dropna(subset=['triage_level'])
            .loc[lambda df_: df_['triage_level'] != ""]
            .rename(columns={'triage_level': '檢傷分級', 'case_count': '案件數'})
        )
        if not triage_chart.empty:
            fig_triage = px.pie(
                triage_chart,
                values='案件數',
                names='檢傷分級',
                title='檢傷分級分布',
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_triage.update_traces(textposition='inside', textinfo='percent+label')
            fig_triage.update_layout(height=400)
            charts['triage_pie'] = fig_triage

    # 5. Dispatch Reason Distribution (Bar Chart)
    reason_df = db_manager.get_counts_by_field(
        'dispatch_reason', filters=filters, limit=10
    )
    if not reason_df.empty and 'dispatch_reason' in reason_df.columns:
        reason_chart = (
            reason_df.dropna(subset=['dispatch_reason'])
            .loc[lambda df_: df_['dispatch_reason'] != ""]
            .rename(columns={'dispatch_reason': '派遣原因', 'case_count': '案件數'})
        )
        if not reason_chart.empty:
            fig_reason = px.bar(
                reason_chart,
                y='派遣原因',
                x='案件數',
                orientation='h',
                title='派遣原因統計（前10）',
                labels={'派遣原因': '派遣原因', '案件數': '案件數'},
                color='案件數',
                color_continuous_scale='Blues',
            )
            fig_reason.update_layout(height=450)
            charts['reason_bar'] = fig_reason

    # 6. Critical Cases Trend (Line Chart over time)
    critical_df = db_manager.get_daily_critical_counts(filters)
    if not critical_df.empty and 'critical_count' in critical_df.columns:
        critical_df['date'] = pd.to_datetime(
            critical_df['date'], errors='coerce'
        )
        critical_chart = (
            critical_df[critical_df['date'].notna()]
            .sort_values('date')
            .assign(日期=lambda df_: df_['date'].dt.date,
                    危急案件數=lambda df_: df_['critical_count'].astype(int))
            [['日期', '危急案件數']]
        )
        if not critical_chart.empty:
            fig_critical = px.line(
                critical_chart,
                x='日期',
                y='危急案件數',
                title='危急案件時間趨勢',
                labels={'日期': '日期', '危急案件數': '危急案件數'},
                markers=True,
            )
            fig_critical.update_layout(height=400)
            charts['critical_trend'] = fig_critical

    # 7. Hospital Distribution (Bar Chart)
    hospital_df = db_manager.get_counts_by_field(
        'destination_hospital', filters=filters, limit=10
    )
    if not hospital_df.empty and 'destination_hospital' in hospital_df.columns:
        hospital_chart = (
            hospital_df.dropna(subset=['destination_hospital'])
            .loc[lambda df_: df_['destination_hospital'] != ""]
            .rename(columns={'destination_hospital': '醫院', 'case_count': '案件數'})
        )
        if not hospital_chart.empty:
            fig_hospital = px.bar(
                hospital_chart,
                x='醫院',
                y='案件數',
                title='後送醫院統計（前10）',
                labels={'醫院': '醫院', '案件數': '案件數'},
                color='案件數',
                color_continuous_scale='Greens',
            )
            fig_hospital.update_layout(xaxis_tickangle=-45, height=400)
            charts['hospital_bar'] = fig_hospital

    return charts


def create_custom_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None) -> go.Figure:
    """
    Create a custom chart based on specified parameters
    
    Args:
        df: DataFrame with data
        chart_type: Type of chart ('bar', 'line', 'scatter', 'pie', 'histogram')
        x_col: Column for x-axis
        y_col: Column for y-axis (optional for some chart types)
    
    Returns:
        Plotly Figure object
    """
    if chart_type == 'bar':
        if y_col:
            fig = px.bar(df, x=x_col, y=y_col)
        else:
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, '數量']
            fig = px.bar(counts, x=x_col, y='數量')
    
    elif chart_type == 'line':
        fig = px.line(df, x=x_col, y=y_col, markers=True)
    
    elif chart_type == 'scatter':
        fig = px.scatter(df, x=x_col, y=y_col)
    
    elif chart_type == 'pie':
        counts = df[x_col].value_counts().reset_index()
        counts.columns = [x_col, '數量']
        fig = px.pie(counts, names=x_col, values='數量')
    
    elif chart_type == 'histogram':
        fig = px.histogram(df, x=x_col)
    
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    return fig
