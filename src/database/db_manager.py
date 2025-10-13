"""
Database manager for handling connections and queries
"""
from sqlalchemy import create_engine, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from .models import Base, EmergencyCase
from ..config import DATABASE_URL


class DatabaseManager:
    """Manages database connections and provides query methods"""
    
    def __init__(self, database_url: str = DATABASE_URL):
        """Initialize database connection"""
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        
    def get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    def add_case(self, case_data: Dict[str, Any]) -> EmergencyCase:
        """Add a new emergency case"""
        session = self.get_session()
        try:
            case = EmergencyCase(**case_data)
            session.add(case)
            session.commit()
            session.refresh(case)
            return case
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_all_cases(self) -> List[EmergencyCase]:
        """Get all emergency cases"""
        session = self.get_session()
        try:
            return session.query(EmergencyCase).all()
        finally:
            session.close()
    
    def get_cases_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[EmergencyCase]:
        """Get cases within a date range"""
        session = self.get_session()
        try:
            return session.query(EmergencyCase).filter(
                and_(
                    EmergencyCase.date >= start_date,
                    EmergencyCase.date <= end_date
                )
            ).all()
        finally:
            session.close()
    
    def get_cases_by_district(self, district: str) -> List[EmergencyCase]:
        """Get cases by district"""
        session = self.get_session()
        try:
            return session.query(EmergencyCase).filter(
                EmergencyCase.incident_district == district
            ).all()
        finally:
            session.close()
    
    def get_critical_cases(self) -> List[EmergencyCase]:
        """Get all critical cases"""
        session = self.get_session()
        try:
            return session.query(EmergencyCase).filter(
                EmergencyCase.critical_case == True
            ).all()
        finally:
            session.close()
    
    def get_cases_dataframe(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Get cases as pandas DataFrame with optional filters"""
        session = self.get_session()
        try:
            query = session.query(EmergencyCase)
            
            if filters:
                if 'start_date' in filters and 'end_date' in filters:
                    query = query.filter(
                        and_(
                            EmergencyCase.date >= filters['start_date'],
                            EmergencyCase.date <= filters['end_date']
                        )
                    )
                if 'district' in filters:
                    query = query.filter(
                        EmergencyCase.incident_district == filters['district']
                    )
                if 'dispatch_reason' in filters:
                    query = query.filter(
                        EmergencyCase.dispatch_reason.like(f"%{filters['dispatch_reason']}%")
                    )
                if 'critical_only' in filters and filters['critical_only']:
                    query = query.filter(EmergencyCase.critical_case == True)
            
            return pd.read_sql(query.statement, session.bind)
        finally:
            session.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        session = self.get_session()
        try:
            total_cases = session.query(func.count(EmergencyCase.id)).scalar()
            critical_cases = session.query(func.count(EmergencyCase.id)).filter(
                EmergencyCase.critical_case == True
            ).scalar()
            
            # Cases by district
            cases_by_district = session.query(
                EmergencyCase.incident_district,
                func.count(EmergencyCase.id)
            ).group_by(EmergencyCase.incident_district).all()
            
            # Cases by dispatch reason
            cases_by_reason = session.query(
                EmergencyCase.dispatch_reason,
                func.count(EmergencyCase.id)
            ).group_by(EmergencyCase.dispatch_reason).all()
            
            # Average response time
            avg_response_time = session.query(
                func.avg(EmergencyCase.response_time_seconds)
            ).scalar()
            
            return {
                'total_cases': total_cases,
                'critical_cases': critical_cases,
                'cases_by_district': dict(cases_by_district),
                'cases_by_reason': dict(cases_by_reason),
                'avg_response_time_seconds': avg_response_time
            }
        finally:
            session.close()
    
    def execute_raw_query(self, query: str) -> pd.DataFrame:
        """Execute a raw SQL query and return results as DataFrame"""
        session = self.get_session()
        try:
            return pd.read_sql(query, session.bind)
        finally:
            session.close()
    
    def get_distinct_values(self, column_name: str) -> List[Any]:
        """Get distinct values for a column"""
        session = self.get_session()
        try:
            column = getattr(EmergencyCase, column_name)
            values = session.query(column).distinct().all()
            return [v[0] for v in values if v[0] is not None]
        finally:
            session.close()

