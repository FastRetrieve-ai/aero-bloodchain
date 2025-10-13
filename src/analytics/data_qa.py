"""
Data Q&A Bot using SQL Agent for querying emergency case database
"""
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go

from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

from ..config import OPENAI_API_KEY, OPENAI_MODEL, DATABASE_URL
from ..database.db_manager import DatabaseManager
from .stats import infer_chart_type, create_chart_from_data


class DataQABot:
    """Q&A Bot for querying emergency case data using natural language"""
    
    def __init__(self):
        """Initialize the data Q&A bot"""
        self.db_manager = DatabaseManager()
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create SQLDatabase instance for LangChain
        engine = create_engine(DATABASE_URL)
        self.sql_database = SQLDatabase(engine)
        
        # Create SQL chain
        self.sql_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.sql_database,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the emergency case data
        
        Returns:
            dict with 'answer', 'sql_query', 'data', and 'chart' (if applicable)
        """
        try:
            # Use SQL chain to generate and execute query
            result = self.sql_chain(question)
            
            answer = result['result']
            sql_query = None
            
            # Extract SQL query from intermediate steps
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    if isinstance(step, str) and 'SELECT' in step.upper():
                        sql_query = step
                        break
            
            # Try to extract data if SQL query is available
            data = None
            chart = None
            
            if sql_query:
                try:
                    data = self.db_manager.execute_raw_query(sql_query)
                    
                    # Try to create a chart if data is suitable
                    if not data.empty and len(data) > 1:
                        chart_type = infer_chart_type(question, data)
                        if chart_type:
                            chart = create_chart_from_data(data, chart_type, question)
                except Exception as e:
                    print(f"Error creating chart: {e}")
            
            return {
                'answer': answer,
                'sql_query': sql_query,
                'data': data,
                'chart': chart
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

