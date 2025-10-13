"""
Data Q&A Bot using SQL Agent for querying emergency case database
"""
from typing import Dict, Any
from types import MethodType
from functools import wraps
import re
import pandas as pd
import plotly.graph_objects as go

from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from openai import BadRequestError

from config import OPENAI_API_KEY, SQL_QA_MODEL, DATABASE_URL
from database.db_manager import DatabaseManager
from analytics.stats import infer_chart_type, create_chart_from_data


class DataQABot:
    """Q&A Bot for querying emergency case data using natural language"""

    def __init__(self):
        """Initialize the data Q&A bot"""
        self.db_manager = DatabaseManager()
        self.model_name = SQL_QA_MODEL
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=OPENAI_API_KEY
        )

        # Create SQLDatabase instance for LangChain
        engine = create_engine(DATABASE_URL)
        self.sql_database = SQLDatabase(engine)

        # Build schema-aware prompt to improve SQL generation
        prompt = self._build_sql_prompt()

        # Create SQL chain that returns SQL without executing
        # This allows us to clean the SQL before execution
        # Note: return_sql=True and return_intermediate_steps=True are incompatible
        self.sql_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.sql_database,
            verbose=True,
            return_intermediate_steps=False,  # Must be False when return_sql=True
            use_query_checker=True,
            return_sql=True,  # Return SQL only, don't execute
            prompt=prompt
        )
        # Neutralize built-in top_k limit to avoid incorrect counts with big data
        try:
            self.sql_chain.top_k = 100000000  # effectively no LIMIT in prompts
        except Exception:
            pass
        self._patch_llm_chain_stop()

    def _patch_llm_chain_stop(self) -> None:
        """Ensure SQLDatabaseChain can call `llm_chain.predict` and strip `stop`.

        - Some LangChain versions removed `predict` in favor of `invoke` on LLMChain.
        - SQLDatabaseChain still calls `llm_chain.predict(...)` in many releases.
        - We attach a lightweight compatibility shim that forwards to `invoke`
          (or the original `predict` when available) after removing unsupported
          kwargs like `stop`.
        """
        llm_chain = self.sql_chain.llm_chain

        original_predict = getattr(llm_chain, "predict", None)
        original_invoke = getattr(llm_chain, "invoke", None)

        def _call_underlying(input_kwargs: Dict[str, Any], callbacks=None):
            if callable(original_predict):
                try:
                    if callbacks is not None:
                        return original_predict(callbacks=callbacks, **input_kwargs)
                    return original_predict(**input_kwargs)
                except TypeError:
                    # Some implementations may not accept callbacks as kwarg
                    return original_predict(**input_kwargs)
            if callable(original_invoke):
                config = {"callbacks": callbacks} if callbacks is not None else None
                return original_invoke(input_kwargs, config=config)
            raise AttributeError("Neither predict nor invoke is available on llm_chain")

        def predict_compat(self_chain, **kwargs):
            callbacks = kwargs.pop("callbacks", None)
            # Reasoning models often reject `stop` — drop it if present
            kwargs.pop("stop", None)
            return _call_underlying(kwargs, callbacks)

        # Attach or override `predict` to the chain instance
        try:
            llm_chain.predict = MethodType(predict_compat, llm_chain)
        except Exception:
            try:
                setattr(llm_chain, "predict", MethodType(predict_compat, llm_chain))
            except Exception:
                # As a last resort, do nothing; SQLDatabaseChain may fail if it
                # expects `predict` on this version.
                pass

    def _build_sql_prompt(self) -> PromptTemplate:
        """Construct a schema-aware prompt for SQLDatabaseChain.

        Expected variables: {input}, {table_info}, {dialect}, {top_k}
        """
        column_guide = (
            "emergency_cases — important columns:\n"
            "- case_number (TEXT, unique): unique case ID.\n"
            "- date (DATETIME): incident date/time; for daily trends use DATE(date).\n"
            "- incident_district (TEXT): district name (e.g., 板橋區/新莊區).\n"
            "- destination_hospital (TEXT): receiving hospital.\n"
            "- dispatch_reason (TEXT): dispatch reason.\n"
            "- triage_level (TEXT): triage level.\n"
            "- critical_case (BOOLEAN): 1 critical, 0 non-critical.\n"
            "- response_time_seconds (INTEGER): response time in seconds.\n"
            "- on_scene_time_seconds (INTEGER): on-scene time in seconds.\n"
            "- transport_time_seconds (INTEGER): transport time in seconds.\n"
            "- hospital_stay_seconds (INTEGER): hospital stay time in seconds.\n"
        )

        template = (
            "You are an expert {dialect} SQL engineer.\n"
            "Translate the question into a single valid SQL query against the schema.\n\n"
            "Schema:\n{table_info}\n\n"
            "Domain notes:\n" + column_guide + "\n"
            "Rules:\n"
            "- Query only the emergency_cases table unless necessary.\n"
            "- Use double quotes for identifiers (\"col\"). Do not use backticks.\n"
            "- Booleans are 0/1 in SQLite (e.g., critical_case = 1).\n"
            "- Prefer aggregates without LIMIT unless the user explicitly requests a limit.\n"
            "- Return only raw SQL, no comments or code fences.\n\n"
            "{input}"
        )

        return PromptTemplate(
            input_variables=["input", "table_info", "dialect"],
            template=template,
        )

    # --- Helpers: SQL extraction & cleaning ---
    def _to_text(self, v) -> str:
        if isinstance(v, str):
            return v
        try:
            # Some LC objects have pretty printers
            if hasattr(v, "to_string"):
                return v.to_string()
        except Exception:
            pass
        return str(v)

    def _extract_sql_from_steps(self, steps):
        # 1) Prefer clean sql_cmd entries
        for s in steps or []:
            if isinstance(s, dict):
                sql = s.get("sql_cmd")
                if sql is not None:
                    sql_text = self._to_text(sql)
                    if "SELECT" in sql_text.upper():
                        return sql_text
        # 2) Fallback: direct strings
        for s in steps or []:
            s_text = self._to_text(s)
            if "SELECT" in s_text.upper():
                return s_text
        # 3) Fallback: from list/tuple parts
        for s in steps or []:
            if isinstance(s, (list, tuple)):
                for part in s:
                    p_text = self._to_text(part)
                    if "SELECT" in p_text.upper():
                        return p_text
        return None

    def _clean_sql(self, sql: str) -> str:
        if not sql:
            return sql
        
        original_sql = sql
        
        # Remove all backtick code fences (iterative approach)
        # This handles both opening ```sql and closing ```
        max_iterations = 10
        iteration = 0
        while "```" in sql and iteration < max_iterations:
            # Remove any line containing only backticks and optional language
            sql = re.sub(r"^```.*$", "", sql, flags=re.MULTILINE)
            # Remove inline backticks
            sql = sql.replace("```", "")
            iteration += 1
        
        # Remove tilde fences
        iteration = 0
        while "~~~" in sql and iteration < max_iterations:
            sql = re.sub(r"^~~~.*$", "", sql, flags=re.MULTILINE)
            sql = sql.replace("~~~", "")
            iteration += 1
        
        # Remove common LangChain markers
        sql = re.sub(r"(?i)SQLQuery:\s*", "", sql)
        sql = re.sub(r"(?i)SQLResult:.*$", "", sql, flags=re.DOTALL)
        sql = re.sub(r"(?i)Answer:.*$", "", sql, flags=re.DOTALL)
        
        # Remove leading/trailing whitespace and normalize
        sql = sql.strip()
        
        # Ensure single semicolon at end
        sql = sql.rstrip(";").strip()
        if sql:
            sql += ";"
        
        # Debug: print if cleaning made significant changes
        if "```" in original_sql and sql != original_sql:
            print(f"[DEBUG] Cleaned SQL from:\n{original_sql[:100]}...\nto:\n{sql[:100]}...")
        
        return sql

    def _normalize_sql_for_sqlite(self, sql: str) -> str:
        """Additional normalization for SQLite compatibility and accuracy.
        - Remove any remaining code fences
        - Convert backticks to double quotes
        - Remove LIMIT for aggregate queries (COUNT/SUM/AVG/MIN/MAX or GROUP BY)
        - Ensure a single trailing semicolon
        """
        if not sql:
            return sql
        
        s = sql.strip()
        
        # Final safety check: remove any remaining backticks (code fences or identifiers)
        # First remove any remaining ``` markers
        s = s.replace("```sql", "").replace("```", "")
        s = s.replace("~~~sql", "").replace("~~~", "")
        
        # Convert backtick identifiers to double quotes
        s = s.replace("`", '"')
        
        # Remove LIMIT when aggregation present (to get complete counts)
        agg_keywords = ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX(", "GROUP BY")
        if any(k in s.upper() for k in agg_keywords):
            s = re.sub(r"\s+LIMIT\s+\d+\s*;?\s*$", ";", s, flags=re.IGNORECASE)
        
        # Normalize whitespace and ensure single semicolon
        s = s.strip().rstrip(";").strip()
        if s:
            s += ";"
        
        return s

    def _is_safe_sql(self, sql: str) -> bool:
        """Basic guard to prevent destructive statements."""
        if not sql:
            return False
        blocked = ("DROP ", "DELETE ", "UPDATE ", "ALTER ", "TRUNCATE ", "INSERT ")
        upper = sql.upper()
        return not any(tok in upper for tok in blocked)

    def _fallback_sql(self, question: str) -> str | None:
        q = (question or "").lower()
        if any(k in q for k in ["行政區", "district"]):
            return (
                'SELECT "incident_district" AS district, COUNT(*) AS cases '\
                'FROM emergency_cases GROUP BY "incident_district" ORDER BY cases DESC;'
            )
        if any(k in q for k in ["醫院", "hospital"]):
            return (
                'SELECT "destination_hospital" AS hospital, COUNT(*) AS cases '
                'FROM emergency_cases GROUP BY "destination_hospital" ORDER BY cases DESC;'
            )
        if any(k in q for k in ["派遣", "dispatch"]):
            return (
                'SELECT "dispatch_reason" AS reason, COUNT(*) AS cases '
                'FROM emergency_cases GROUP BY "dispatch_reason" ORDER BY cases DESC;'
            )
        if any(k in q for k in ["檢傷", "triage"]):
            return (
                'SELECT "triage_level" AS triage, COUNT(*) AS cases '
                'FROM emergency_cases GROUP BY "triage_level" ORDER BY cases DESC;'
            )
        return None

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the emergency case data
        
        Returns:
            dict with 'answer', 'sql_query', 'data', and 'chart' (if applicable)
        """
        try:
            # Use SQL chain to generate SQL (return_sql=True means it won't execute)
            chain_input = {"query": question}
            result = self.sql_chain.invoke(chain_input)

            # When return_sql=True, the result is either:
            # - A dict with 'result' key containing SQL string
            # - Directly the SQL string
            raw_sql = None
            
            if isinstance(result, dict):
                # The 'result' field contains the generated SQL
                raw_sql = result.get('result', '')
            else:
                # Sometimes it returns the SQL directly as a string
                raw_sql = str(result)
            
            # Clean and normalize SQL through the full pipeline
            sql_query = None
            if raw_sql:
                sql_text = self._to_text(raw_sql)
                print(f"[DEBUG] Raw SQL from chain: {sql_text[:200]}")
                sql_query = self._clean_sql(sql_text)
                sql_query = self._normalize_sql_for_sqlite(sql_query)
                print(f"[DEBUG] Cleaned SQL: {sql_query[:200]}")

            # Execute the cleaned SQL ourselves
            data = None
            chart = None
            display_sql = None
            answer = ""

            if sql_query:
                try:
                    # Safety check before execution
                    if not self._is_safe_sql(sql_query):
                        raise ValueError("拒絕執行潛在破壞性 SQL")
                    
                    display_sql = sql_query
                    # Execute the cleaned SQL
                    data = self.db_manager.execute_raw_query(sql_query)
                    
                    # Generate a natural language answer from the data
                    if data is not None and not data.empty:
                        answer = f"查詢成功，找到 {len(data)} 筆結果。"
                    else:
                        answer = "查詢成功，但沒有找到符合條件的結果。"
                        
                except Exception as e:
                    print(f"[DEBUG] SQL execution error: {e}")
                    # Try fallback heuristics if execution failed
                    fallback = self._fallback_sql(question)
                    if fallback:
                        try:
                            data = self.db_manager.execute_raw_query(fallback)
                            display_sql = fallback
                            if data is not None and not data.empty:
                                answer = f"使用備用查詢成功，找到 {len(data)} 筆結果。"
                        except Exception as fallback_error:
                            print(f"[DEBUG] Fallback also failed: {fallback_error}")
                            answer = f"查詢執行失敗：{str(e)}"
                    else:
                        answer = f"查詢執行失敗：{str(e)}"

            # Try to create a chart if data is suitable
            if data is not None and not data.empty and len(data) > 0:
                try:
                    chart_type = infer_chart_type(question, data)
                    if chart_type:
                        chart = create_chart_from_data(data, chart_type, question)
                except Exception as e:
                    print(f"[DEBUG] Error creating chart: {e}")

            return {
                'answer': answer,
                'sql_query': display_sql or sql_query,
                'data': data,
                'chart': chart
            }

        except BadRequestError as e:
            message = (
                "OpenAI API 回應錯誤，請確認 SQL 模型是否支援 stop 參數或降低問題複雜度。"
                f" 詳細資訊：{getattr(e, 'message', None) or getattr(e, 'body', None) or str(e)}"
            )
            return {
                'answer': f"抱歉，處理您的問題時發生錯誤：{message}",
                'sql_query': None,
                'data': None,
                'chart': None
            }
        except Exception as e:
            return {
                'answer': f"抱歉，處理您的問題時發生錯誤：{str(e)}",
                'sql_query': None,
                'data': None,
                'chart': None
            }

    def execute_custom_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a custom SQL query"""
        return self.db_manager.execute_raw_query(sql_query)

    def get_table_info(self) -> str:
        """Get information about the database tables"""
        return self.sql_database.get_table_info()
