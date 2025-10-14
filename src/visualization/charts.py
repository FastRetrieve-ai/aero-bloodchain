"""
Statistical charts for emergency case data
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any


def create_statistics_charts(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create various statistical charts
    
    Returns:
        Dictionary of chart names to plotly Figure objects
    """
    charts = {}
    
    # 1. Cases by District (Bar Chart)
    if 'incident_district' in df.columns:
        district_counts = df['incident_district'].value_counts().reset_index()
        district_counts.columns = ['行政區', '案件數']
        
        fig_district = px.bar(
            district_counts.head(15),
            x='行政區',
            y='案件數',
            title='各行政區急救案件數量',
            labels={'行政區': '行政區', '案件數': '案件數'},
            color='案件數',
            color_continuous_scale='Reds'
        )
        fig_district.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        charts['district_bar'] = fig_district
    
    # 2. Cases by Time of Day (Line Chart)
    if 'dispatch_time' in df.columns:
        # Extract hour from dispatch time
        df_time = df.copy()
        df_time['hour'] = df_time['dispatch_time'].apply(
            lambda x: int(str(x).split(':')[0]) if pd.notna(x) and ':' in str(x) else None
        )
        
        if df_time['hour'].notna().any():
            hour_counts = df_time['hour'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['小時', '案件數']
            
            fig_time = px.line(
                hour_counts,
                x='小時',
                y='案件數',
                title='24小時急救案件分布',
                labels={'小時': '時段 (小時)', '案件數': '案件數'},
                markers=True
            )
            fig_time.update_layout(height=400)
            charts['time_line'] = fig_time
    
    # 3. Response Time Distribution (Histogram)
    if 'response_time_seconds' in df.columns:
        # Convert to minutes; clip negatives to 0; limit extreme outliers
        response_series = (df['response_time_seconds'].astype('float64') / 60.0).dropna()
        response_series = response_series.clip(lower=0)
        if len(response_series) > 0:
            ub = float(response_series.quantile(0.99)) if len(response_series) > 50 else float(response_series.max())
            ub = max(ub, 1.0)
            # Use go.Histogram with explicit bin size to avoid plotly auto-binning artifacts
            import plotly.graph_objects as go
            target_bins = 60
            bin_size = max(ub / target_bins, 0.5)  # at least 0.5 minute per bin
            fig_response = go.Figure(data=[
                go.Histogram(
                    x=response_series,
                    xbins=dict(start=0, end=ub, size=bin_size),
                    marker_color='#636EFA',
                    hovertemplate='範圍 %{xbins.start}–%{xbins.end}<br>案件數 %{y}<extra></extra>'
                )
            ])
            fig_response.update_layout(
                title='反應時間分布',
                xaxis_title='反應時間 (分鐘)',
                yaxis_title='案件數',
                height=400,
                bargap=0.05,
                xaxis_range=[0, ub]
            )
            charts['response_histogram'] = fig_response

    # 3b. Transport Time Distribution (送醫時間)
    if 'transport_time_seconds' in df.columns:
        transport_series = (df['transport_time_seconds'].astype('float64') / 60.0).dropna()
        transport_series = transport_series.clip(lower=0)
        if len(transport_series) > 0:
            ub_t = float(transport_series.quantile(0.99)) if len(transport_series) > 50 else float(transport_series.max())
            ub_t = max(ub_t, 1.0)
            target_bins = 60
            bin_size_t = max(ub_t / target_bins, 1.0)
            fig_transport = go.Figure(data=[
                go.Histogram(
                    x=transport_series,
                    xbins=dict(start=0, end=ub_t, size=bin_size_t),
                    marker_color='#EF553B',
                    hovertemplate='範圍 %{xbins.start}–%{xbins.end}<br>案件數 %{y}<extra></extra>'
                )
            ])
            fig_transport.update_layout(
                title='送醫時間分布',
                xaxis_title='送醫時間 (分鐘)',
                yaxis_title='案件數',
                height=400,
                bargap=0.05,
                xaxis_range=[0, ub_t]
            )
            charts['transport_histogram'] = fig_transport
    
    # 4. Triage Level Distribution (Pie Chart)
    if 'triage_level' in df.columns:
        triage_counts = df['triage_level'].value_counts().reset_index()
        triage_counts.columns = ['檢傷分級', '案件數']
        
        fig_triage = px.pie(
            triage_counts,
            values='案件數',
            names='檢傷分級',
            title='檢傷分級分布',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_triage.update_traces(textposition='inside', textinfo='percent+label')
        fig_triage.update_layout(height=400)
        charts['triage_pie'] = fig_triage
    
    # 5. Dispatch Reason Distribution (Bar Chart)
    if 'dispatch_reason' in df.columns:
        reason_counts = df['dispatch_reason'].value_counts().head(10).reset_index()
        reason_counts.columns = ['派遣原因', '案件數']
        
        fig_reason = px.bar(
            reason_counts,
            y='派遣原因',
            x='案件數',
            orientation='h',
            title='派遣原因統計（前10）',
            labels={'派遣原因': '派遣原因', '案件數': '案件數'},
            color='案件數',
            color_continuous_scale='Blues'
        )
        fig_reason.update_layout(height=450)
        charts['reason_bar'] = fig_reason
    
    # 6. Critical Cases Trend (Line Chart over time)
    if 'date' in df.columns and 'critical_case' in df.columns:
        df_critical = df.copy()
        df_critical['date'] = pd.to_datetime(df_critical['date'], errors='coerce')
        df_critical = df_critical[df_critical['date'].notna()]
        
        if len(df_critical) > 0:
            df_critical['date_only'] = df_critical['date'].dt.date
            daily_critical = df_critical.groupby('date_only')['critical_case'].sum().reset_index()
            daily_critical.columns = ['日期', '危急案件數']
            
            fig_critical = px.line(
                daily_critical,
                x='日期',
                y='危急案件數',
                title='危急案件時間趨勢',
                labels={'日期': '日期', '危急案件數': '危急案件數'},
                markers=True
            )
            fig_critical.update_layout(height=400)
            charts['critical_trend'] = fig_critical
    
    # 7. Hospital Distribution (Bar Chart)
    if 'destination_hospital' in df.columns:
        hospital_counts = df['destination_hospital'].value_counts().head(10).reset_index()
        hospital_counts.columns = ['醫院', '案件數']
        
        fig_hospital = px.bar(
            hospital_counts,
            x='醫院',
            y='案件數',
            title='後送醫院統計（前10）',
            labels={'醫院': '醫院', '案件數': '案件數'},
            color='案件數',
            color_continuous_scale='Greens'
        )
        fig_hospital.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
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
