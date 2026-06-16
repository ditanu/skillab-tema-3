"""
Database utilities for PostgreSQL connection.

Usage:
    from skillab import get_engine, get_session

    engine = get_engine()
    with get_session() as session:
        session.execute(...)
"""
import os
from typing import Optional, Generator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

# Module-level engine instance
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_db_url() -> str:
    """
    Get database URL from environment.

    Supports:
        - DATABASE_URL (full connection string)
        - Individual vars: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
                          POSTGRES_USER, POSTGRES_PASSWORD
    """
    # Check for full URL first
    if url := os.getenv("DATABASE_URL"):
        return url

    # Build from individual components
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "skilllab")
    user = os.getenv("POSTGRES_USER", "user")
    password = os.getenv("POSTGRES_PASSWORD", "password")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_engine(echo: bool = False) -> Engine:
    """
    Get or create SQLAlchemy engine.

    Args:
        echo: If True, log all SQL statements

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine

    if _engine is None:
        _engine = create_engine(
            get_db_url(),
            echo=echo,
            pool_pre_ping=True,  # Check connection health
            pool_size=5,
            max_overflow=10,
        )

    return _engine


def init_db(echo: bool = False) -> Engine:
    """
    Initialize database connection on startup.

    Args:
        echo: If True, log all SQL statements

    Returns:
        SQLAlchemy Engine instance
    """
    global _SessionLocal

    engine = get_engine(echo=echo)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    print(f"Database initialized: {get_db_url().split('@')[1] if '@' in get_db_url() else get_db_url()}")
    return engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Get a database session with automatic cleanup.

    Usage:
        with get_session() as session:
            result = session.execute(select(User))

    Yields:
        SQLAlchemy Session instance
    """
    global _SessionLocal

    if _SessionLocal is None:
        init_db()

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_dependency():
    """
    FastAPI dependency for database sessions.

    Usage in FastAPI:
        @app.get("/users")
        def get_users(db: Session = Depends(get_session_dependency)):
            return db.query(User).all()
    """
    global _SessionLocal

    if _SessionLocal is None:
        init_db()

    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
