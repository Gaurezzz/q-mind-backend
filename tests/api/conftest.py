import os

os.environ.setdefault("DB_TYPE", "sqlite")
os.environ.setdefault("DB_NAME", ":memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-api-testing-only")
os.environ.setdefault("ALGORITHM", "HS256")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db import models
from db.config import get_db
from main import app

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def db_session():
    """
    Creates a fresh in-memory database session for each API test.
    StaticPool ensures all connections share the same in-memory DB,
    so tables created here are visible to the FastAPI TestClient.
    """
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
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


@pytest.fixture(scope="function")
def client(db_session):
    """
    Returns a FastAPI TestClient with get_db overridden to use the test database.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def registered_user(client):
    """
    Registers a test user and returns the response JSON.
    """
    response = client.post("/auth/register", json={
        "email": "testuser@test.com",
        "password": "StrongPassword1!"
    })
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="function")
def auth_headers(client, registered_user):
    """
    Logs in as the registered test user and returns Authorization headers.
    """
    response = client.post("/auth/login", data={
        "username": "testuser@test.com",
        "password": "StrongPassword1!"
    })
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
