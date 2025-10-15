"""
Database manager for handling connections and queries
"""
from sqlalchemy import create_engine, func, case, and_
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional, Dict, Any, Sequence
from datetime import datetime
import pandas as pd

from .models import Base, EmergencyCase
from config import DATABASE_URL


class DatabaseManager:
    """Manages database connections and provides query methods"""
    
    def __init__(
        self,
        database_url: str = DATABASE_URL,
    ):
        """Initialize database connection"""
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @staticmethod
    def _normalize_limit(limit: Optional[int]) -> Optional[int]:
        """Normalize limit values so that non-positive or invalid inputs become None."""
        if limit is None:
            return None
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return None
        return limit_value if limit_value > 0 else None
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)

    def _apply_filters(self, query, filters: Optional[Dict[str, Any]] = None):
        """Apply common filter clauses to a SQLAlchemy query."""
        if not filters:
            return query

        start_date = filters.get("start_date")
        end_date = filters.get("end_date")
        if start_date and end_date:
            query = query.filter(
                EmergencyCase.date >= start_date,
                EmergencyCase.date <= end_date,
            )
        elif start_date:
            query = query.filter(EmergencyCase.date >= start_date)
        elif end_date:
            query = query.filter(EmergencyCase.date <= end_date)

        district_list = filters.get("districts")
        if district_list:
            query = query.filter(EmergencyCase.incident_district.in_(district_list))
        else:
            district = filters.get("district")
            if district:
                query = query.filter(EmergencyCase.incident_district == district)

        dispatch_reasons = filters.get("dispatch_reasons")
        if dispatch_reasons:
            query = query.filter(EmergencyCase.dispatch_reason.in_(dispatch_reasons))
        else:
            dispatch_reason = filters.get("dispatch_reason")
            if dispatch_reason:
                query = query.filter(
                    EmergencyCase.dispatch_reason.ilike(f"%{dispatch_reason}%")
                )

        triage_levels = filters.get("triage_levels")
        if triage_levels:
            query = query.filter(EmergencyCase.triage_level.in_(triage_levels))

        if filters.get("critical_only"):
            query = query.filter(EmergencyCase.critical_case.is_(True))

        return query
        
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
    
    def get_cases_summary(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate metrics for the current filter selection."""
        session = self.get_session()
        try:
            total_cases = (
                self._apply_filters(
                    session.query(func.count(EmergencyCase.id)),
                    filters,
                ).scalar()
                or 0
            )

            critical_cases = (
                self._apply_filters(
                    session.query(func.count(EmergencyCase.id)),
                    filters,
                )
                .filter(EmergencyCase.critical_case.is_(True))
                .scalar()
                or 0
            )

            avg_response_seconds = self._apply_filters(
                session.query(func.avg(EmergencyCase.response_time_seconds)),
                filters,
            ).scalar()

            date_bounds = self._apply_filters(
                session.query(
                    func.min(EmergencyCase.date),
                    func.max(EmergencyCase.date),
                ),
                filters,
            ).first()
            if date_bounds:
                period_start, period_end = date_bounds
            else:
                period_start, period_end = None, None

            covered_districts = (
                self._apply_filters(
                    session.query(
                        func.count(
                            func.distinct(EmergencyCase.incident_district)
                        )
                    ),
                    filters,
                ).scalar()
                or 0
            )

            return {
                "total_cases": int(total_cases),
                "critical_cases": int(critical_cases),
                "avg_response_seconds": float(avg_response_seconds)
                if avg_response_seconds is not None
                else None,
                "covered_districts": int(covered_districts),
                "period_start": period_start,
                "period_end": period_end,
            }
        finally:
            session.close()

    def get_district_aggregates(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Aggregate case metrics by administrative district."""
        session = self.get_session()
        try:
            query = (
                self._apply_filters(
                    session.query(
                        EmergencyCase.incident_district.label(
                            "incident_district"
                        ),
                        func.count(EmergencyCase.id).label("case_count"),
                        func.sum(
                            case(
                                (EmergencyCase.critical_case.is_(True), 1),
                                else_=0,
                            )
                        ).label("critical_count"),
                        func.avg(EmergencyCase.response_time_seconds).label(
                            "avg_response_seconds"
                        ),
                    ),
                    filters,
                )
                .filter(EmergencyCase.incident_district.isnot(None))
                .group_by(EmergencyCase.incident_district)
            )

            return pd.read_sql(query.statement, session.bind)
        finally:
            session.close()

    def get_daily_district_counts(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Aggregate case volume and critical counts by date and district."""
        session = self.get_session()
        try:
            date_expr = func.date(EmergencyCase.date)
            query = (
                self._apply_filters(
                    session.query(
                        date_expr.label("date"),
                        EmergencyCase.incident_district.label(
                            "incident_district"
                        ),
                        func.count(EmergencyCase.id).label("case_count"),
                        func.sum(
                            case(
                                (EmergencyCase.critical_case.is_(True), 1),
                                else_=0,
                            )
                        ).label("critical_count"),
                    ),
                    filters,
                )
                .filter(EmergencyCase.incident_district.isnot(None))
                .group_by(date_expr, EmergencyCase.incident_district)
                .order_by(date_expr, EmergencyCase.incident_district)
            )
            return pd.read_sql(query.statement, session.bind)
        finally:
            session.close()

    def get_daily_critical_counts(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Aggregate total critical cases per day."""
        session = self.get_session()
        try:
            date_expr = func.date(EmergencyCase.date)
            query = (
                self._apply_filters(
                    session.query(
                        date_expr.label("date"),
                        func.sum(
                            case(
                                (EmergencyCase.critical_case.is_(True), 1),
                                else_=0,
                            )
                        ).label("critical_count"),
                    ),
                    filters,
                )
                .group_by(date_expr)
                .order_by(date_expr)
            )
            return pd.read_sql(query.statement, session.bind)
        finally:
            session.close()

    def get_counts_by_field(
        self,
        column_name: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Aggregate case counts based on a categorical column."""
        session = self.get_session()
        try:
            try:
                column = getattr(EmergencyCase, column_name)
            except AttributeError as exc:
                raise ValueError(
                    f"Invalid column requested for aggregation: {column_name}"
                ) from exc

            query = session.query(
                column.label(column_name),
                func.count(EmergencyCase.id).label("case_count"),
            )
            query = self._apply_filters(query, filters)
            query = query.filter(column.isnot(None))
            query = query.group_by(column).order_by(
                func.count(EmergencyCase.id).desc()
            )
            if limit:
                query = query.limit(limit)
            return pd.read_sql(query.statement, session.bind)
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
        filters: Optional[Dict[str, Any]] = None,
        *,
        limit: Optional[int] = None,
        random_sample: Optional[bool] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Get cases as pandas DataFrame with optional filters and limits.

        Args:
            filters: Optional filter criteria keyed by column/condition.
            limit: Optional explicit row cap overriding the configured default.
            random_sample: When True, randomize order before limiting to reduce bias.
                Defaults to the manager-level setting.
        """
        session = self.get_session()
        try:
            if columns:
                try:
                    selected = [getattr(EmergencyCase, col) for col in columns]
                except AttributeError as exc:
                    raise ValueError(f"Invalid column requested: {exc}") from exc
                query = session.query(*selected)
            else:
                query = session.query(EmergencyCase)

            query = self._apply_filters(query, filters)

            normalized_limit = self._normalize_limit(limit)
            randomize = bool(random_sample and normalized_limit)
            if normalized_limit is not None:
                if randomize:
                    query = query.order_by(func.random())
                query = query.limit(normalized_limit)

            df = pd.read_sql(query.statement, session.bind)
            if normalized_limit is not None:
                df.attrs["query_limit"] = normalized_limit
                df.attrs["random_sample"] = randomize
            return df
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
