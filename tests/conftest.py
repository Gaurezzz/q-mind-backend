import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from db import models

# --- Database Fixtures ---
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="function")
def db_session():
    """
    Creates a fresh database session for each test.
    """
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    models.Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        models.Base.metadata.drop_all(bind=engine)
