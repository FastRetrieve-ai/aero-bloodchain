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
    EMERGENCY_MANUAL_PDF,
)
from database.db_manager import DatabaseManager
from qa_bot.manual_qa import ManualQABot
from visualization.map_view import (
    create_heatmap,
    create_time_animation_map,
)
from visualization.charts import create_statistics_charts
from visualization.elevation_profile_page import (
    render_uav_elevation_profile_page,
)
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
    """Page 1: Emergency Manual Q&A Bot (chat UI aligned with analytics)."""
    st.title("📋 緊急救護程序問答系統")
    st.markdown("根據《113年緊急傷病患救護流程手冊》檢索回答，並附上可點擊引用頁碼。")

    if EMERGENCY_MANUAL_PDF.exists():
        try:
            with open(EMERGENCY_MANUAL_PDF, "rb") as f:
                st.download_button(
                    label="📄 下載完整手冊 PDF",
                    data=f.read(),
                    file_name=EMERGENCY_MANUAL_PDF.name,
                    mime="application/pdf",
                )
        except Exception:
            pass

    # Initialize bot and index
    if "manual_bot" not in st.session_state:
        try:
            st.session_state.manual_bot = ManualQABot()
            with st.spinner("檢查 / 建置手冊向量索引…"):
                chunks = st.session_state.manual_bot.build_or_load_index()
            st.success(f"✅ 手冊索引就緒（{chunks} 個片段）")
            st.session_state.manual_messages = []  # [{role, content, citations}]
        except Exception as e:
            st.error(f"初始化手冊問答系統時發生錯誤：{e}")
            return

    # Controls
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔁 重建向量索引"):
            try:
                with st.spinner("重建中…"):
                    chunks = st.session_state.manual_bot.build_or_load_index()
                st.success(f"已完成重建（{chunks} 個片段）")
            except Exception as e:
                st.error(f"重建索引失敗：{e}")
    with col_b:
        if st.button("🗑️ 清除對話"):
            st.session_state.manual_messages = []
            st.session_state.manual_bot.clear_history()
            st.rerun()

    # Helper: sort citations by numeric page number
    def _page_sort_key(c: dict) -> int:
        try:
            return int(str(c.get("page_number", "")).strip())
        except Exception:
            return 10**9

    # Render chat history
    for msg in st.session_state.manual_messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            if role == "assistant":
                citations = msg.get("citations") or []
                if citations:
                    st.caption("引用來源（點擊展開內容頁面）")
                    for c in sorted(citations, key=_page_sort_key):
                        pn = c.get("page_number")
                        label = f"p.{str(pn).zfill(3)}"
                        with st.expander(label):
                            # Show markdown page content as ground truth snippet
                            try:
                                page_md = st.session_state.manual_bot.vector_store.read_page_markdown(
                                    st.session_state.manual_bot.manual_dir, pn
                                )
                                st.markdown(page_md)
                            except Exception:
                                st.info("找不到該頁的 markdown 內容。")

    # Chat input
    prompt = st.chat_input("輸入要查詢的手冊內容…")
    if prompt:
        # Echo user
        st.session_state.manual_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ask bot
        with st.chat_message("assistant"):
            with st.spinner("檢索與作答中…"):
                try:
                    result = st.session_state.manual_bot.ask(prompt, k_chunks=20)
                    answer = result.get("answer") or ""
                    citations = result.get("citations") or []
                    st.markdown(answer)
                    if citations:
                        st.caption("引用來源（點擊展開頁面內容）")
                        for c in sorted(citations, key=_page_sort_key):
                            pn = c.get("page_number")
                            label = f"p.{str(pn).zfill(3)}"
                            with st.expander(label):
                                try:
                                    page_md = st.session_state.manual_bot.vector_store.read_page_markdown(
                                        st.session_state.manual_bot.manual_dir, pn
                                    )
                                    st.markdown(page_md)
                                except Exception:
                                    st.info("找不到該頁的 markdown 內容。")

                    # Persist assistant message
                    st.session_state.manual_messages.append(
                        {"role": "assistant", "content": answer, "citations": citations}
                    )
                except Exception as e:
                    error_msg = f"處理問題時發生錯誤：{e}"
                    st.error(error_msg)
                    st.session_state.manual_messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


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
            show_hospitals = st.checkbox(
                "🏥 顯示責任醫院位置",
                value=True,
                key="map_show_hospitals",
            )
            show_isochrones = st.checkbox(
                "🚕 顯示 30/60 分交通等時線",
                value=True,
                key="map_show_isochrones",
            )
            show_fire_stations = st.checkbox(
                "🚒 顯示消防分隊位置",
                value=True,
                key="map_show_fire_stations",
            )

            if district_stats is None:
                with st.spinner("載入地圖資料..."):
                    district_stats = db_manager.get_district_aggregates(filters)
            if district_stats is None or district_stats.empty:
                st.warning("目前沒有可用的地圖資料。")
            else:
                with st.spinner("生成熱力圖與責任醫院圖層..."):
                    heatmap_map, map_messages = create_heatmap(
                        district_stats,
                        include_hospitals=show_hospitals,
                        include_isochrones=show_isochrones,
                        include_fire_stations=show_fire_stations,
                    )
                    st_folium(
                        heatmap_map,
                        width=None,
                        height=800,
                        returned_objects=[],
                    )
                caption_parts = []
                if show_hospitals:
                    caption_parts.append("責任醫院資料來源：衛福部")
                if show_isochrones:
                    caption_parts.append("等時線來源：OpenRouteService")
                if show_fire_stations:
                    caption_parts.append("消防隊資料來源：新北市政府消防局")
                if caption_parts:
                    st.caption("；".join(caption_parts))
                else:
                    st.caption("Folium 熱力圖依行政區聚合，避免大量資料傳輸。")
                if map_messages:
                    for message in dict.fromkeys(map_messages):
                        st.warning(message)

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
    """Page 3: Data Analytics Q&A (chat UI)"""
    st.title("📊 數據分析問答系統")
    st.markdown("使用自然語言查詢急救案件數據，並以口語化說明搭配圖表。")

    # Initialize analytics bot and chat history
    if "analytics_bot" not in st.session_state:
        try:
            st.session_state.analytics_bot = DataQABot()
            st.session_state.analytics_messages = []  # [{role, content, sql_query, data, chart}]
            st.success("✅ 分析系統已就緒")
        except Exception as e:
            st.error(f"初始化分析系統時發生錯誤：{e}")
            return

    # Sample prompts
    with st.expander("💡 範例問題"):
        st.markdown(
            "- 每個行政區有多少急救案件？\n"
            "- 危急案件的平均反應時間是多少？\n"
            "- 哪個醫院接收最多案件？\n"
            "- 最常見的派遣原因是什麼？\n"
            "- 各檢傷分級的案件數量分布？"
        )

    # Custom SQL query section
    with st.expander("🔧 進階：自訂 SQL 查詢"):
        custom_sql = st.text_area(
            "輸入 SQL 查詢語句：",
            placeholder="SELECT * FROM emergency_cases LIMIT 10",
        )
        if st.button("執行 SQL", key="exec_custom_sql") and custom_sql:
            try:
                result_df = st.session_state.analytics_bot.execute_custom_query(
                    custom_sql
                )
                st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"執行 SQL 時發生錯誤：{e}")

    # Render chat history
    for msg in st.session_state.analytics_messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            if role == "assistant":
                sql_query = msg.get("sql_query")
                data = msg.get("data")
                chart = msg.get("chart")
                if sql_query:
                    with st.expander("📝 SQL 查詢"):
                        st.code(sql_query, language="sql")
                if data is not None:
                    with st.expander("📋 數據表格"):
                        st.dataframe(data, use_container_width=True)
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)

    # Chat input
    prompt = st.chat_input("輸入要查詢的數據問題…")
    if prompt:
        # Echo user message
        st.session_state.analytics_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query bot
        with st.chat_message("assistant"):
            with st.spinner("分析中…"):
                try:
                    result = st.session_state.analytics_bot.ask(prompt)
                    answer = result.get("answer") or "已完成分析。"
                    st.markdown(answer)

                    if result.get("sql_query"):
                        with st.expander("📝 SQL 查詢"):
                            st.code(result["sql_query"], language="sql")
                    if result.get("data") is not None:
                        with st.expander("📋 數據表格"):
                            st.dataframe(result["data"], use_container_width=True)
                    if result.get("chart") is not None:
                        st.plotly_chart(result["chart"], use_container_width=True)

                    # Persist assistant message with artifacts
                    st.session_state.analytics_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sql_query": result.get("sql_query"),
                            "data": result.get("data"),
                            "chart": result.get("chart"),
                        }
                    )
                except Exception as e:
                    error_msg = f"查詢時發生錯誤：{e}"
                    st.error(error_msg)
                    st.session_state.analytics_messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


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
            "✈️ 無人機直線剖面",
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
    elif page == "✈️ 無人機直線剖面":
        render_uav_elevation_profile_page()
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
