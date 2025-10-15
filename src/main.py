"""
Main Streamlit application for Blood Chain System
熱血飛騰：血品供應韌性系統
"""
import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    OPENAI_API_KEY,
    APP_LOGIN_USERNAME,
    APP_LOGIN_PASSWORD,
)
from database.db_manager import DatabaseManager
from qa_bot.manual_qa import ManualQABot
from visualization.map_view import (
    create_heatmap,
    create_time_animation_map,
    create_hex_density_map,
    create_deck_heatmap,
)
from visualization.charts import create_statistics_charts
from analytics.data_qa import DataQABot
from forms.generator import FormGenerator

# Import streamlit_folium for map rendering
try:
    from streamlit_folium import st_folium
except ImportError:
    st.error("請安裝 streamlit-folium: pip install streamlit-folium")


# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_api_key():
    """Check if OpenAI API key is configured"""
    if not OPENAI_API_KEY:
        st.sidebar.error("⚠️ 請設定 OpenAI API Key")
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if api_key:
            import os
            os.environ["OPENAI_API_KEY"] = api_key
            st.sidebar.success("✅ API Key 已設定")
            return True
        return False
    return True


def page_manual_qa():
    """Page 1: Emergency Manual Q&A Bot"""
    st.title("📋 緊急救護程序問答系統")
    st.markdown("根據新北市政府消防局緊急傷病患作業程序手冊回答問題")
    
    # Initialize QA bot
    if 'qa_bot' not in st.session_state:
        try:
            st.session_state.qa_bot = ManualQABot()
            with st.spinner("載入緊急救護手冊..."):
                st.session_state.qa_bot.load_manual()
            st.success("✅ 手冊載入完成")
        except Exception as e:
            st.error(f"載入手冊時發生錯誤：{e}")
            return
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history using Streamlit chat layout
    for entry in st.session_state.chat_history:
        if isinstance(entry, dict):
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            sources = entry.get("sources", [])
        else:
            # Backward compatibility for tuple-based history
            question, answer = entry
            sources = []

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                st.caption("參考章節")
                for source in sources:
                    section = source.get("section") or "未知章節"
                    similarity = source.get("similarity")
                    excerpt = source.get("content", "")
                    if similarity is not None:
                        st.markdown(f"- **{section}** · 相似度 {similarity:.2f}")
                    else:
                        st.markdown(f"- **{section}**")
                    if excerpt:
                        with st.expander(f"查看 {section} 節錄"):
                            st.write(excerpt)

    # Input form
    with st.form(key='qa_form'):
        question = st.text_input(
            "請輸入您的問題：",
            placeholder="例如：G1 通用流程包含哪些內容？"
        )
        submit = st.form_submit_button("🔍 詢問")

        if submit and question:
            with st.spinner("思考中..."):
                try:
                    result = st.session_state.qa_bot.ask(question)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result['answer'],
                        "sources": result.get('sources', [])
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"處理問題時發生錯誤：{e}")

    # Clear history button
    if st.session_state.chat_history and st.button("🗑️ 清除對話歷史"):
        st.session_state.chat_history = []
        st.session_state.qa_bot.clear_history()
        st.rerun()


def ensure_authenticated() -> None:
    """Enforce simple username/password authentication when configured."""
    if not APP_LOGIN_USERNAME or not APP_LOGIN_PASSWORD:
        return

    auth_box = st.sidebar.container()

    if st.session_state.get("auth_user"):
        auth_box.success(f"👤 已登入：{st.session_state['auth_user']}")
        if auth_box.button("登出", key="logout_button"):
            st.session_state.pop("auth_user", None)
            st.session_state.pop("auth_error", None)
            st.rerun()
        return

    auth_box.warning("請登入以使用系統")

    with auth_box.form("login_form"):
        username = st.text_input("帳號")
        password = st.text_input("密碼", type="password")
        submitted = st.form_submit_button("登入")

    if submitted:
        if username == APP_LOGIN_USERNAME and password == APP_LOGIN_PASSWORD:
            st.session_state["auth_user"] = username
            st.session_state.pop("auth_error", None)
            st.rerun()
        else:
            st.session_state["auth_error"] = "帳號或密碼錯誤，請再試一次。"

    error_message = st.session_state.get("auth_error")
    if error_message:
        auth_box.error(error_message)

    st.stop()


def page_maps():
    """Page 2: Interactive Case Maps"""
    st.title("🗺️ 急救案件地理視覺化")
    st.markdown("互動式地圖顯示急救案件分布與時間變化")

    # Initialize database
    db_manager = DatabaseManager()

    # Sidebar filters
    st.sidebar.subheader("地圖篩選條件")

    try:
        districts = sorted([d for d in db_manager.get_distinct_values('incident_district') if d])
    except Exception:
        districts = []
    district_options = ["全部"] + districts
    selected_districts = st.sidebar.multiselect(
        "行政區",
        district_options,
        default=["全部"] if district_options else [],
    )

    try:
        dispatch_options = sorted([d for d in db_manager.get_distinct_values('dispatch_reason') if d])
    except Exception:
        dispatch_options = []
    selected_dispatch = st.sidebar.multiselect("派遣原因", dispatch_options)

    try:
        triage_options = sorted([t for t in db_manager.get_distinct_values('triage_level') if t])
    except Exception:
        triage_options = []
    selected_triage = st.sidebar.multiselect("檢傷分級", triage_options)

    date_range = st.sidebar.date_input("日期範圍", [])
    critical_only = st.sidebar.checkbox("僅顯示危急個案")

    filters = {}
    if selected_districts:
        filtered_districts = [d for d in selected_districts if d != "全部"]
        if filtered_districts:
            filters['districts'] = filtered_districts
    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
        filters['start_date'] = start_date
        filters['end_date'] = end_date
    if selected_dispatch:
        filters['dispatch_reasons'] = selected_dispatch
    if selected_triage:
        filters['triage_levels'] = selected_triage
    if critical_only:
        filters['critical_only'] = True

    try:
        summary = db_manager.get_cases_summary(filters)

        total_cases = summary.get("total_cases", 0)
        if total_cases == 0:
            st.warning("沒有符合條件的資料")
            return

        st.info(f"符合條件的案件共 {total_cases:,} 筆")
        critical_cases = summary.get("critical_cases", 0)
        avg_response_minutes = None
        avg_seconds = summary.get("avg_response_seconds")
        if avg_seconds is not None:
            avg_response_minutes = avg_seconds / 60.0

        covered_districts = summary.get("covered_districts", 0)
        period_start = summary.get("period_start")
        period_end = summary.get("period_end")

        metric_cols = st.columns(4)
        metric_cols[0].metric("案件數", f"{total_cases:,}")
        metric_cols[1].metric("危急案件", f"{critical_cases:,}")
        if avg_response_minutes is not None:
            metric_cols[2].metric("平均反應時間 (分)", f"{avg_response_minutes:.1f}")
        else:
            metric_cols[2].metric("平均反應時間 (分)", "—")
        metric_cols[3].metric("涵蓋行政區", f"{covered_districts}")

        if period_start is not None and period_end is not None:
            st.caption(f"資料期間：{period_start:%Y-%m-%d} ～ {period_end:%Y-%m-%d}")

        tab_heatmap, tab_animation, tab_stats = st.tabs(
            ["📍 熱力圖與標記", "⏱️ 時間序列動畫", "📊 統計圖表"]
        )

        district_stats = None
        daily_district_counts = None

        with tab_heatmap:
            st.markdown("透過熱力圖快速掌握案件密度，並利用標記瀏覽案件細節。")
            map_mode = st.radio(
                "地圖模式",
                # ["Folium 熱力圖", "Hex 聚合地圖 (pydeck)", "Pydeck 熱力圖 (全量)"]
                ["Folium 熱力圖"],
                horizontal=True,
            )

            if map_mode == "Folium 熱力圖":
                if district_stats is None:
                    with st.spinner("載入地圖資料..."):
                        district_stats = db_manager.get_district_aggregates(filters)
                if district_stats is None or district_stats.empty:
                    st.warning("目前沒有可用的地圖資料。")
                else:
                    with st.spinner("生成熱力圖..."):
                        heatmap = create_heatmap(district_stats)
                        st_folium(
                            heatmap,
                            width=None,
                            height=800,
                            returned_objects=[],
                        )
                    st.caption(" Folium 熱力圖已改為依行政區聚合，避免大量資料傳輸。")
            elif map_mode == "Hex 聚合地圖 (pydeck)":
                with st.spinner("生成 Hex 聚合地圖..."):
                    hex_df = db_manager.get_cases_dataframe(
                        filters, columns=["incident_district"]
                    )
                    deck = create_hex_density_map(hex_df, resolution=8, show_3d=True)
                    if deck is None:
                        st.warning(
                            "缺少依賴：請在環境中安裝 pydeck 與 h3 後再試 (`poetry add pydeck h3`)"
                        )
                    else:
                        st.pydeck_chart(deck, use_container_width=True, height=520)
                st.caption("Hex 聚合能夠在 40–50 萬筆資料下保持流暢互動。")
            else:
                with st.spinner("生成 Pydeck 熱力圖 (全量)..."):
                    deck_df = db_manager.get_cases_dataframe(
                        filters, columns=["incident_district"]
                    )
                    deck = create_deck_heatmap(deck_df, radius_pixels=60, intensity=1.0)
                    if deck is None:
                        st.warning("缺少依賴：請安裝 pydeck 後再試 (`poetry add pydeck`) ")
                    else:
                        st.pydeck_chart(deck, use_container_width=True, height=520)
                st.caption("此模式會傳送所有點位，適合強機或生產部署環境。")

        with tab_animation:
            st.markdown("時間序列動畫呈現案件發生的累積趨勢與時空分布。")
            if daily_district_counts is None:
                with st.spinner("載入時間序列資料..."):
                    daily_district_counts = db_manager.get_daily_district_counts(filters)
            if daily_district_counts is None or daily_district_counts.empty:
                st.warning("目前沒有可用的時間序列資料。")
            else:
                with st.spinner("生成時間動畫..."):
                    animation_fig = create_time_animation_map(daily_district_counts)
                    st.plotly_chart(animation_fig, use_container_width=True)

        with tab_stats:
            st.markdown("多維統計視角幫助追蹤行政區、時段與檢傷等核心指標。")
            with st.spinner("生成統計圖表..."):
                if district_stats is None:
                    district_stats = db_manager.get_district_aggregates(filters)
                charts = create_statistics_charts(
                    db_manager,
                    filters=filters,
                    district_summary=district_stats,
                )

                chart_order = [
                    "time_line",
                    "critical_trend",
                    "response_histogram",
                    "transport_histogram",
                    "triage_pie",
                    "reason_bar",
                    "district_bar",
                    "hospital_bar",
                ]
                available_charts = [key for key in chart_order if key in charts]

                if not available_charts:
                    st.info("目前沒有可顯示的統計圖表。")
                else:
                    for i in range(0, len(available_charts), 2):
                        cols = st.columns(2)
                        for col, key in zip(cols, available_charts[i:i+2]):
                            with col:
                                st.plotly_chart(charts[key], use_container_width=True)

    except Exception as e:
        st.error(f"載入地圖時發生錯誤：{e}")


def page_analytics():
    """Page 3: Data Analytics Q&A"""
    st.title("📊 數據分析問答系統")
    st.markdown("使用自然語言查詢急救案件數據")
    
    # Initialize analytics bot
    if 'analytics_bot' not in st.session_state:
        try:
            st.session_state.analytics_bot = DataQABot()
            st.success("✅ 分析系統已就緒")
        except Exception as e:
            st.error(f"初始化分析系統時發生錯誤：{e}")
            return
    
    # Example questions
    with st.expander("💡 範例問題"):
        st.markdown("""
        - 每個行政區有多少急救案件？
        - 危急案件的平均反應時間是多少？
        - 哪個醫院接收最多案件？
        - 最常見的派遣原因是什麼？
        - 各檢傷分級的案件數量分布？
        """)
    
    # Question input
    question = st.text_input(
        "請輸入您的數據查詢問題：",
        placeholder="例如：各行政區的急救案件數量統計"
    )
    
    if st.button("🔍 查詢") and question:
        with st.spinner("分析中..."):
            try:
                result = st.session_state.analytics_bot.ask(question)
                
                # Display answer
                st.subheader("💬 回答")
                st.markdown(result['answer'])
                
                # Display SQL query if available
                if result['sql_query']:
                    with st.expander("📝 SQL查詢語句"):
                        st.code(result['sql_query'], language='sql')
                
                # Display data table if available
                if result['data'] is not None and not result['data'].empty:
                    st.subheader("📋 數據表格")
                    st.dataframe(result['data'], use_container_width=True)
                
                # Display chart if available
                if result['chart'] is not None:
                    st.subheader("📈 視覺化圖表")
                    st.plotly_chart(result['chart'], use_container_width=True)
            
            except Exception as e:
                st.error(f"查詢時發生錯誤：{e}")
    
    # Custom SQL query section
    with st.expander("🔧 進階：自訂 SQL 查詢"):
        custom_sql = st.text_area(
            "輸入 SQL 查詢語句：",
            placeholder="SELECT * FROM emergency_cases LIMIT 10"
        )
        if st.button("執行 SQL") and custom_sql:
            try:
                result_df = st.session_state.analytics_bot.execute_custom_query(custom_sql)
                st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"執行 SQL 時發生錯誤：{e}")


def page_forms():
    """Page 4: Administrative Forms Generator"""
    st.title("📄 行政表單產生器")
    st.markdown("快速產生電子行政表單（骨架實作）")
    
    st.info("⚠️ 此功能為骨架實作，需要根據實際需求客製化表單格式")
    
    # Initialize form generator
    form_gen = FormGenerator()
    
    # Form type selection
    form_type = st.selectbox(
        "選擇表單類型",
        [
            "案件摘要報告",
            "統計分析報告",
            "案件彙總表",
            "自訂表單（待實作）"
        ]
    )
    
    # Output format
    output_format = st.radio("輸出格式", ["PDF", "Excel"])
    
    if form_type == "案件摘要報告":
        st.subheader("案件摘要報告")
        
        # Input fields (placeholder)
        with st.form("case_summary_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                case_number = st.text_input("案件編號")
                patient_name = st.text_input("患者姓名")
                incident_district = st.text_input("發生地點行政區")
            
            with col2:
                date = st.date_input("日期")
                dispatch_reason = st.text_input("派遣原因")
                hospital = st.text_input("後送醫院")
            
            notes = st.text_area("備註")
            
            submit = st.form_submit_button("📥 產生表單")
            
            if submit:
                case_data = {
                    'case_number': case_number,
                    'patient_name': patient_name,
                    'incident_district': incident_district,
                    'date': date,
                    'dispatch_reason': dispatch_reason,
                    'destination_hospital': hospital,
                    'notes': notes
                }
                
                try:
                    if output_format == "PDF":
                        pdf_bytes = form_gen.generate_case_summary_pdf(case_data)
                        st.download_button(
                            label="📄 下載 PDF",
                            data=pdf_bytes,
                            file_name=f"case_summary_{case_number}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        excel_bytes = form_gen.generate_case_summary_excel([case_data])
                        st.download_button(
                            label="📊 下載 Excel",
                            data=excel_bytes,
                            file_name=f"case_summary_{case_number}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    st.success("✅ 表單產生成功！")
                except Exception as e:
                    st.error(f"產生表單時發生錯誤：{e}")
    
    elif form_type == "統計分析報告":
        st.subheader("統計分析報告")
        st.info("TODO: 實作統計分析報告表單輸入介面")
    
    elif form_type == "案件彙總表":
        st.subheader("案件彙總表")
        st.info("TODO: 實作案件彙總表表單輸入介面")
    
    else:
        st.subheader("自訂表單")
        st.info("TODO: 實作自訂表單功能")


def main():
    """Main application"""

    # Header
    st.title(f"🚁 {APP_TITLE}")
    st.caption(APP_SUBTITLE)

    # Authentication guard must run before exposing other controls
    ensure_authenticated()

    # Check API key
    if not check_api_key():
        st.stop()

    # Sidebar navigation
    with st.sidebar:
        st.image("images/logo.jpg", use_container_width=True)

    st.sidebar.title("📋 功能選單")
    page = st.sidebar.radio(
        "選擇功能",
        [
            "📋 緊急救護問答",
            "🗺️ 地理視覺化地圖",
            "📊 數據分析問答",
            "📄 行政表單產生"
        ]
    )

    st.sidebar.divider()

    # Display selected page
    if page == "📋 緊急救護問答":
        page_manual_qa()
    elif page == "🗺️ 地理視覺化地圖":
        page_maps()
    elif page == "📊 數據分析問答":
        page_analytics()
    elif page == "📄 行政表單產生":
        page_forms()

    # Footer
    st.sidebar.divider()

    # with st.sidebar:
    #     st.image("images/logo.jpg", use_container_width=True)
    st.sidebar.caption("© 2025 熱血飛騰：血品供應韌性系統")
    st.sidebar.caption("Emergency Blood Chain System")


if __name__ == "__main__":
    main()
