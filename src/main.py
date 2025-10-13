"""
Main Streamlit application for Blood Chain System
熱血飛騰：血品供應韌性系統
"""
import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_TITLE, APP_SUBTITLE, OPENAI_API_KEY
from database.db_manager import DatabaseManager
from qa_bot.manual_qa import ManualQABot
from visualization.map_view import create_heatmap, create_time_animation_map
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
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**👤 問題：** {question}")
            st.markdown(f"**🤖 回答：** {answer}")
            st.divider()
    
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
                    st.session_state.chat_history.append((question, result['answer']))
                    st.rerun()
                except Exception as e:
                    st.error(f"處理問題時發生錯誤：{e}")
    
    # Clear history button
    if st.session_state.chat_history and st.button("🗑️ 清除對話歷史"):
        st.session_state.chat_history = []
        st.session_state.qa_bot.clear_history()
        st.rerun()


def page_maps():
    """Page 2: Interactive Case Maps"""
    st.title("🗺️ 急救案件地理視覺化")
    st.markdown("互動式地圖顯示急救案件分布與時間變化")
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Sidebar filters
    st.sidebar.subheader("地圖篩選條件")
    
    # Get distinct districts
    try:
        districts = db_manager.get_distinct_values('incident_district')
        selected_district = st.sidebar.selectbox("行政區", ["全部"] + districts)
    except:
        selected_district = "全部"
    
    # Date range filter
    date_range = st.sidebar.date_input("日期範圍", [])
    
    # Critical cases only
    critical_only = st.sidebar.checkbox("僅顯示危急個案")
    
    # Prepare filters
    filters = {}
    if selected_district != "全部":
        filters['district'] = selected_district
    if len(date_range) == 2:
        filters['start_date'] = date_range[0]
        filters['end_date'] = date_range[1]
    if critical_only:
        filters['critical_only'] = True
    
    # Get data
    try:
        df = db_manager.get_cases_dataframe(filters)
        
        if df.empty:
            st.warning("沒有符合條件的資料")
            return
        
        st.info(f"共 {len(df)} 筆案件")
        
        # Row 1: Heatmap with markers
        st.subheader("📍 案件熱力圖與標記")
        with st.spinner("生成熱力圖..."):
            heatmap = create_heatmap(df, filters)
            st_folium(heatmap, width=1200, height=500)
        
        st.divider()
        
        # Row 2: Time animation
        st.subheader("⏱️ 時間序列動畫")
        with st.spinner("生成時間動畫..."):
            animation_fig = create_time_animation_map(df)
            st.plotly_chart(animation_fig, use_container_width=True)
        
        st.divider()
        
        # Statistics charts
        st.subheader("📊 統計圖表")
        with st.spinner("生成統計圖表..."):
            charts = create_statistics_charts(df)
            
            # Display charts in columns
            col1, col2 = st.columns(2)
            
            with col1:
                if 'district_bar' in charts:
                    st.plotly_chart(charts['district_bar'], use_container_width=True)
                if 'time_line' in charts:
                    st.plotly_chart(charts['time_line'], use_container_width=True)
            
            with col2:
                if 'triage_pie' in charts:
                    st.plotly_chart(charts['triage_pie'], use_container_width=True)
                if 'response_histogram' in charts:
                    st.plotly_chart(charts['response_histogram'], use_container_width=True)
    
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
    
    # Check API key
    if not check_api_key():
        st.stop()
    
    # Sidebar navigation
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
    st.sidebar.caption("© 2025 熱血飛騰：血品供應韌性系統")
    st.sidebar.caption("Emergency Blood Chain System")


if __name__ == "__main__":
    main()

