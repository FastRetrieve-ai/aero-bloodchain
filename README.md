# 熱血飛騰：血品供應韌性系統

# Emergency Blood Chain System

一個整合 AI、地理視覺化與數據分析的緊急救護管理系統

## 專案簡介

本系統提供以下四大核心功能：

1. **📋 緊急救護程序問答機器人** - 基於新北市政府消防局緊急傷病患作業程序手冊的 AI 問答系統
2. **🗺️ 地理視覺化地圖** - 互動式地圖展示急救案件分布、熱力圖與時間序列動畫
3. **📊 數據分析問答** - 使用自然語言查詢急救案件數據，自動產生圖表
4. **📄 行政表單產生器** - 快速產生 PDF/Excel 格式的行政文書（骨架實作）

> 新增：可透過環境變數設定簡易帳號密碼登入，保護 Streamlit 介面。

## 技術架構

- **前端框架**: Streamlit
- **後端語言**: Python 3.10+
- **AI/LLM**: OpenAI GPT-4 (準備支援 GPT-5)
- **向量資料庫**: SQLite-Vec (開發) / PGVector (生產)
- **關聯式資料庫**: SQLite (開發) / PostgreSQL (生產)
- **AI 框架**: LangChain
- **地圖視覺化**: Folium + Plotly
- **圖表視覺化**: Plotly, Matplotlib, Seaborn
- **文件生成**: ReportLab (PDF), OpenPyXL (Excel)
- **依賴管理**: Poetry

## 系統需求

- Python 3.10 或更高版本
- Poetry (Python 依賴管理工具)
- OpenAI API Key
- OpenRouteService API Key（可選，用於責任醫院等時線）

## 安裝步驟

### 1. 克隆專案

```bash
cd /path/to/aero-bloodchain
```

### 2. 安裝 Poetry

```bash
# macOS / Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 3. 安裝依賴

```bash
poetry install
```

### 4. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env` 檔案，填入您的 OpenAI API Key 與模型設定：

```env
OPENAI_API_KEY=your_openai_api_key_here
# 推薦：一般對話/SQL 皆可用 gpt-4o-mini；或填入 reasoning 模型（如 gpt-5/o-*）
OPENAI_MODEL=gpt-4o-mini
# 可選：數據分析問答專用模型，未設定時沿用 OPENAI_MODEL
SQL_QA_MODEL=
# 可選：啟用簡易登入保護（兩者皆需設定）
APP_LOGIN_USERNAME=
APP_LOGIN_PASSWORD=

# Optional: OpenRouteService 等時線 API
OPENROUTESERVICE_API_KEY=
# OPENROUTESERVICE_BASE_URL=https://api.openrouteservice.org
```

或者，使用 Streamlit secrets (推薦用於生產環境)：

```bash
mkdir -p .streamlit
```

建立 `.streamlit/secrets.toml`：

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
# Optional login guard
# APP_LOGIN_USERNAME = "admin"
# APP_LOGIN_PASSWORD = "changeme"
```

### 5. 預先建立 OpenRouteService 等時線快取（建議）

等時線圖層會在 Streamlit 地圖頁面載入快取檔案 `data/hospital_isochrones.geojson`。
首次部署或更新 `data/hospital.json` 後，請執行下列指令產生最新快取：

```bash
poetry run python scripts/cache_isochrones.py
```

指令會對每間責任醫院呼叫 OpenRouteService Isochrones API，生成 30/60/120 分鐘的範圍多邊形。
若沒有設置 `OPENROUTESERVICE_API_KEY`，腳本會直接提示錯誤並中止。產出檔案不應提交至版本控制，
但可放心放在本機 `data/` 目錄供應用程式讀取。

### 6. 載入數據到資料庫

在執行應用程式之前，需要先將 Excel/CSV 檔案的數據載入到 SQL 資料庫：

```bash
poetry run python scripts/load_data.py
```

此腳本會：

- 建立資料庫表格
- 讀取 `data/` 目錄下的所有 .xlsx 和 .csv 檔案
- 解析並驗證數據
- 將記錄插入資料庫
- 顯示載入統計資訊

## 執行應用程式

### 開發模式

```bash
poetry run streamlit run src/main.py
```

應用程式將在 `http://localhost:8501` 啟動

### 生產部署

```bash
poetry run streamlit run src/main.py --server.port 80 --server.address 0.0.0.0
```

## 功能說明

### 1. 緊急救護程序問答機器人

**技術實現**：

- 使用 LangChain RAG (Retrieval-Augmented Generation) 架構
- 將救護手冊切分成語意區塊，並建立向量嵌入
- 使用 OpenAI Embeddings + GPT-4 進行語意搜尋與回答
- 自動引用相關章節編號 (G1-G7, M1-M10, S1-S15 等)

**使用方式**：

1. 選擇「緊急救護問答」功能
2. 輸入關於救護程序的問題
3. 系統會參考手冊內容回答，並引用章節來源

**範例問題**：

- "G1 通用流程包含哪些內容？"
- "如何處理無脈搏的患者？"
- "腦中風的評估流程是什麼？"

### 2. 地理視覺化地圖

**技術實現**：

- 使用 Folium 建立互動式地圖
- HeatMap 顯示案件密度
- MarkerCluster 展示個別案件詳情
- Plotly 實現時間序列動畫

**效能優化重點**：

- Folium 熱力圖改以行政區聚合統計，直接從 SQL 取得彙總資料，避免一次載入全量點位。
- Hex 聚合地圖維持 pydeck + H3 作為大量資料選項，可於地圖分頁的「地圖模式」切換使用。
- 統計圖表與動畫改為即時查詢資料庫，只抓取繪圖所需欄位，減少記憶體占用。

**功能特色**：

- **熱力圖與標記**：顯示案件分布密度，點擊標記查看詳情
- **急救責任醫院地圖**：以顏色與 emoji 呈現醫院分級，可選擇疊加 30/60/120 分交通等時線（需設定 OpenRouteService API 並先執行 `scripts/cache_isochrones.py` 快取）
- **時間序列動畫**：播放案件隨時間的變化
- **統計圖表**：各行政區案件數、時段分布、反應時間等

**篩選條件**：

- 日期範圍
- 行政區（多選，預設為「全部」）
- 派遣原因（多選）
- 檢傷分級（多選）
- 僅顯示危急個案

### 3. 數據分析問答

**技術實現**：

- LangChain `SQLDatabaseChain`（啟用 `use_query_checker`、回傳 `intermediate_steps`）
- 自訂 Prompt（內含表結構與欄位語意）提升 SQL 正確率
- 自動清理模型輸出的 SQL：移除 ```sql/SQLQuery/SQLResult 標記、反引號→雙引號、聚合查詢移除 LIMIT、加入安全檢查
- 失敗時套用備援 SQL（常見的 GROUP BY 統計），並同時輸出圖表與表格

**功能特色**：

- 自然語言查詢資料庫
- 自動生成 SQL 語句
- 智慧選擇圖表類型（長條圖、折線圖、圓餅圖等）
- 支援自訂 SQL 查詢（進階功能）

**範例問題**：

- "每個行政區有多少急救案件？"
- "危急案件的平均反應時間是多少？"
- "最常見的派遣原因是什麼？"
- "各檢傷分級的案件數量分布？"

**模型選擇建議**：

- 一般建議 `OPENAI_MODEL=gpt-4o-mini`
- 若要使用 reasoning 模型（如 gpt-5/o-*），可保留 `OPENAI_MODEL`，或改設 `SQL_QA_MODEL` 專供 SQL 鏈使用；程式已內建相容層處理 stop 參數差異。

### 4. 行政表單產生器

**技術實現**：

- ReportLab 生成 PDF
- OpenPyXL 生成 Excel
- 骨架架構設計，便於客製化

**目前提供**：

- 案件摘要報告 (PDF/Excel)
- 統計分析報告 (PDF)
- 案件彙總表 (Excel)

**未來擴展**：

- 根據實際需求客製化表單格式
- 加入電子簽章功能
- 支援郵件寄送

## 專案結構

```
aero-bloodchain/
├── pyproject.toml              # Poetry 依賴配置
├── .env.example                # 環境變數範本
├── .gitignore                  # Git 忽略檔案
├── README.md                   # 專案說明
├── plan.md                     # 開發計畫
├── data/                       # 原始數據檔案
│   ├── 2024-cases.xlsx
│   ├── 2025-cases.xlsx
│   ├── sample-2025-cases.csv
│   └── emergency-patient-rescue-process.md
├── database/                   # SQLite 資料庫檔案 (自動生成)
│   └── bloodchain.db
├── vector_db/                  # 向量資料庫檔案 (自動生成)
│   └── manual_embeddings.db
├── scripts/                    # 工具腳本
│   └── load_data.py           # 數據載入腳本
└── src/                        # 原始碼
    ├── __init__.py
    ├── main.py                # Streamlit 主程式
    ├── config.py              # 配置管理
    ├── database/              # 資料庫模組
    │   ├── __init__.py
    │   ├── models.py          # SQLAlchemy 模型
    │   └── db_manager.py      # 資料庫管理器
    ├── qa_bot/                # 問答機器人模組
    │   ├── __init__.py
    │   ├── manual_qa.py       # 手冊問答
    │   └── embeddings.py      # 向量儲存
    ├── visualization/         # 視覺化模組
    │   ├── __init__.py
    │   ├── map_view.py        # 地圖視覺化
    │   └── charts.py          # 統計圖表
    ├── analytics/             # 分析模組
    │   ├── __init__.py
    │   ├── data_qa.py         # 數據問答
    │   └── stats.py           # 統計分析
    └── forms/                 # 表單生成模組
        ├── __init__.py
        ├── generator.py       # 表單生成器
        └── templates/         # 表單模板
```

## 資料庫架構

### EmergencyCase 表格

主要欄位包括：

- 基本資訊：案件編號、日期、大隊、分隊、車號
- 派遣資訊：派遣原因、派遣分類、發生地點行政區
- 患者資訊：姓名、身分證字號、年齡、性別、地址
- 時間記錄：派遣、抵達、接觸患者、離開現場、抵達醫院等時間
- 生命徵象：意識狀態、GCS、呼吸、脈搏、血壓、體溫、血氧
- 處置記錄：一般處置、呼吸處置、加護處置、創傷處置
- 醫院資訊：後送醫院、檢傷分級、病歷號

完整欄位定義請參考 `src/database/models.py`

## 常見問題

### Q: 如何更新數據？

A: 將新的 Excel/CSV 檔案放入 `data/` 目錄，然後重新執行：

```bash
poetry run python scripts/load_data.py
```

### Q: 如何切換到 PostgreSQL？

A: 修改 `.env` 中的 `DATABASE_URL`：

```env
DATABASE_URL=postgresql://user:password@localhost:5432/bloodchain
```

### Q: OpenAI API 費用大概多少？

A: 依使用量而定，建議：

- 設定 API usage limits
- 使用 GPT-3.5 降低成本（效果較差）
- 實作快取機制

### Q: 如何客製化表單格式？

A: 編輯 `src/forms/generator.py`，參考 ReportLab 文檔：
<https://www.reportlab.com/docs/reportlab-userguide.pdf>

### Q: 地圖沒有顯示標記？

A: 確認：

1. 數據中有 `incident_district` 欄位
2. 行政區名稱符合 `DISTRICT_COORDINATES` 字典
3. 如需更精確定位，可啟用 geocoding 功能

### Q: 地圖出現 Custom tiles must have an attribution？

A: 已改為手動註冊具 attribution 的圖磚層（CartoDB/OSM/Stamen）。請更新到本版程式或重新啟動。

### Q: 時間序列動畫報 tuple index out of range？

A: 已加入 Plotly 動畫控制的安全檢查，避免在沒有 updatemenus 按鈕時取值失敗。

### Q: SQL 執行時出現 near "```sql"？

A: 我們已在 SQL 執行前清理 Markdown code fence 與 SQLQuery/SQLResult 標記並正規化 SQL。若仍出現，請展開 UI 中的 SQL 查詢區塊貼上，我們可擴充清理規則。

### Q: 匯入腳本或主程式發生 attempted relative import beyond top-level package？

A: 專案已全面改為絕對匯入（`from config ...`），且 `scripts/load_data.py` 會自動將 `src/` 加入 `sys.path`。請確保從專案根目錄執行指令。

### Q: Poetry 安裝專案時顯示 No file/folder found for package？

A: 我們採用依賴管理模式，未將專案註冊為可安裝套件；`pyproject.toml` 已設為 package-mode=false。請直接 `poetry install` 使用依賴與指令。

## 開發指令

格式/靜態檢查：

```bash
poetry run black src scripts
poetry run flake8 src scripts
```

單元測試：

```bash
poetry run pytest -q
```

## 效能優化建議

1. **向量資料庫**：生產環境建議使用 PGVector
2. **快取**：使用 Streamlit caching 減少重複計算
3. **批次處理**：大量數據使用批次載入
4. **索引**：為常查詢欄位建立資料庫索引

## 安全性考量

1. **API Key 保護**：絕不將 API Key 提交到版本控制
2. **輸入驗證**：使用 SQL Agent 時防範 SQL 注入
3. **存取控制**：生產環境應加入認證機制
4. **數據脫敏**：患者個資應適當處理

## 未來擴展

- [ ] 支援即時數據同步
- [ ] 加入使用者認證與權限管理
- [ ] 整合更多數據來源（無人機、智慧冰箱）
- [ ] 建立預測模型（案件預測、資源分配）
- [ ] 多語言支援
- [ ] 行動裝置優化

## 授權

見 LICENSE 檔案

## 聯絡方式

專案維護者：[Your Name]
Email: [your.email@example.com]

## 致謝

- 新北市政府消防局提供救護作業程序手冊
- OpenAI 提供 AI 模型支援
- Streamlit 提供優秀的前端框架
