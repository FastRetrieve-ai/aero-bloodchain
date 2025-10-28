"""
Statistical analysis helpers for analytics Q&A.

Adds optional LLM-backed chart inference while preserving heuristic fallbacks.
"""
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, CHART_LLM_MODEL


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
    Infer an appropriate chart type using an LLM first, with safe fallbacks.

    Returns one of: 'bar', 'line', 'pie', 'scatter', 'histogram', 'box'.
    """
    # Quick checks
    if data is None or data.empty or len(data.columns) < 1:
        return None

    allowed = ["bar", "line", "pie", "scatter", "histogram", "box"]

    # Attempt LLM inference
    try:
        llm = ChatOpenAI(model=CHART_LLM_MODEL, openai_api_key=OPENAI_API_KEY)
        cols = ", ".join([f"{c}({str(data[c].dtype)})" for c in data.columns])
        sample = data.head(10).to_csv(index=False)
        prompt = (
            "[系統]\n你是資料視覺化助理。從使用者問題和資料結構判斷最佳圖表類型。\n\n"
            "[任務]\n"
            + "使用者問題:\n" + (question or "") + "\n\n"
            + f"可用圖表: {allowed}\n"
            + f"欄位與型別: {cols}\n\n"
            + "資料預覽(CSV, 前10列):\n" + sample + "\n\n"
            + "只輸出一個關鍵字，不要解釋。"
        )
        resp = llm.invoke(prompt)
        text = (getattr(resp, "content", "") or "").strip().lower()
        # Extract first allowed keyword present
        for kind in allowed:
            if re.search(rf"\b{re.escape(kind)}\b", text):
                return kind
        # Some models may answer in Chinese; map common terms
        zh_map = {"長條圖": "bar", "折線圖": "line", "圓餅圖": "pie", "散點圖": "scatter", "直方圖": "histogram", "箱型圖": "box"}
        if text in zh_map:
            return zh_map[text]
    except Exception:
        pass

    # Heuristic fallback
    question_lower = (question or "").lower()
    if any(w in question_lower for w in ["趨勢", "變化", "時間", "日期", "trend", "time"]):
        return "line"
    if any(w in question_lower for w in ["比較", "對比", "排名", "compare", "rank"]):
        return "bar"
    if any(w in question_lower for w in ["箱型", "箱形", "箱線圖", "box plot", "箱型圖"]):
        return "box" if len(data.columns) >= 2 else "histogram"
    if any(w in question_lower for w in ["分布", "比例", "佔比", "distribution", "proportion", "percentage"]):
        return "pie" if len(data) <= 10 else "bar"
    if any(w in question_lower for w in ["相關", "關係", "correlation", "relationship"]):
        return "scatter"

    # Default based on data structure
    if len(data.columns) == 2 and pd.api.types.is_numeric_dtype(data[data.columns[1]]):
        return "bar" if len(data) <= 10 else "line"
    return "bar"


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
            # If values column is missing, compute frequency of x_col
            if not y_col or y_col not in data.columns:
                counts = data[x_col].value_counts().reset_index()
                counts.columns = [x_col, "count"]
                fig = px.pie(counts, names=x_col, values="count", title=title)
            else:
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
