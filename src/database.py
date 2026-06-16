"""
Database - Connection + Transaction Context Manager
"""
import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://demo:demo123@localhost:5433/rag_demo"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


@contextmanager
def transaction() -> Generator[Session, None, None]:
    """
    Context manager pentru tranzacții.

    Usage:
        with transaction() as db:
            repo = DocumentChunkRepository(db)
            repo.add(chunk)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Returnează o sesiune nouă (trebuie închisă manual)."""
    return SessionLocal()
