from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class Config(BaseSettings):
    """
    Config: Database configuration settings.

    Manages environment variables and connection string generation for different database types.
    """
    DB_TYPE: str = ""
    DB_NAME: str = ""
    DB_USER: str = ""
    DB_PASSWORD: str = ""
    DB_HOST: str = ""
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(env_file=".env")

    @property
    def DB_URL(self):
        """
        Constructs the database connection URL based on configuration.

        Returns:
            str: The SQLAlchemy connection string.
        """
        if self.DB_TYPE == "sqlite":
            return f"sqlite:///{self.DB_NAME}"
        elif self.DB_TYPE == "postgresql":
            return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}/{self.DB_NAME}"
        else:
            raise ValueError(f"Unsupported DB_TYPE: {self.DB_TYPE}")

db_config = Config()

# Create SQLAlchemy engine instance
engine = create_engine(url=db_config.DB_URL, echo=True, connect_args={"check_same_thread": False} if db_config.DB_TYPE == "sqlite" else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency function to provide a database session.

    Yields:
        Session: A SQLAlchemy session for database operations.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()