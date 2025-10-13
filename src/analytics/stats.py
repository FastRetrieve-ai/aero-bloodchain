"""
Statistical analysis functions
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics from dataframe"""
    stats = {
        'total_records': len(df),
        'columns': list(df.columns),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        stats['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:10]:  # Limit to first 10
        value_counts = df[col].value_counts()
        stats['categorical_stats'][col] = {
            'unique_values': len(value_counts),
            'top_values': value_counts.head(5).to_dict()
        }
    
    return stats


def infer_chart_type(question: str, data: pd.DataFrame) -> Optional[str]:
    """
    Infer the best chart type based on the question and data
    
    Returns:
        Chart type string: 'bar', 'line', 'pie', 'scatter', 'histogram', or None
    """
    question_lower = question.lower()
    
    # Check data shape
    if data.empty or len(data.columns) < 1:
        return None
    
    # Keywords for different chart types
    if any(word in question_lower for word in ['趨勢', '變化', '時間', '日期', 'trend', 'time']):
        return 'line'
    
    if any(word in question_lower for word in ['比較', '對比', '排名', 'compare', 'rank']):
        return 'bar'
    
    if any(word in question_lower for word in ['箱型', '箱形', '箱線圖', 'box plot', '箱型圖']):
        if len(data.columns) >= 2:
            return 'box'
        return 'histogram'
    
    if any(word in question_lower for word in ['分布', '比例', '佔比', 'distribution', 'proportion', 'percentage']):
        if len(data) <= 10:
            return 'pie'
        else:
            return 'bar'
    
    if any(word in question_lower for word in ['相關', '關係', 'correlation', 'relationship']):
        return 'scatter'
    
    # Default based on data structure
    if len(data.columns) == 2:
        if data[data.columns[1]].dtype in ['int64', 'float64']:
            if len(data) <= 10:
                return 'bar'
            else:
                return 'line'
    
    return 'bar'  # Default fallback


def create_chart_from_data(
    data: pd.DataFrame, 
    chart_type: str, 
    title: str = "數據視覺化"
) -> go.Figure:
    """
    Create a chart from data
    
    Args:
        data: DataFrame with data
        chart_type: Type of chart to create
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title="無資料可顯示")
        return fig
    
    # Get first two columns by default
    cols = data.columns.tolist()
    x_col = cols[0]
    y_col = cols[1] if len(cols) > 1 else None
    
    try:
        if chart_type == 'bar':
            if y_col:
                fig = px.bar(data, x=x_col, y=y_col, title=title)
            else:
                fig = px.bar(data, x=x_col, title=title)
        
        elif chart_type == 'line':
            fig = px.line(data, x=x_col, y=y_col, title=title, markers=True)
        
        elif chart_type == 'pie':
            fig = px.pie(data, names=x_col, values=y_col, title=title)
        
        elif chart_type == 'scatter':
            fig = px.scatter(data, x=x_col, y=y_col, title=title)
        
        elif chart_type == 'histogram':
            fig = px.histogram(data, x=x_col, title=title)
        
        elif chart_type == 'box':
            if y_col:
                fig = px.box(data, x=x_col, y=y_col, title=title, points="outliers")
            else:
                fig = px.box(data, y=x_col, title=title, points="outliers")
        
        else:
            fig = px.bar(data, x=x_col, y=y_col, title=title)
        
        fig.update_layout(height=400)
        return fig
    
    except Exception as e:
        print(f"Error creating chart: {e}")
        fig = go.Figure()
        fig.update_layout(title=f"建立圖表時發生錯誤: {str(e)}")
        return fig
