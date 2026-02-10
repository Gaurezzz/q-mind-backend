from datetime import datetime, timedelta, timezone
from typing import Any, Union
from jose import jwt
from db.config import db_config 

def create_access_token(subject: Union[str, Any], expires_delta = None) -> str:
    """
    create_access_token: Generates a JWT access token.

    Args:
        subject (Union[str, Any]): The subject of the token (user identifier).
        expires_delta (timedelta, optional): The duration until the token expires.

    Returns:
        str: The encoded JWT access token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=30)

    to_encode = {"exp": expire, "sub": str(subject)}

    encoded_jwt = jwt.encode(
        to_encode,
        db_config.SECRET_KEY,
        algorithm=db_config.ALGORITHM
    )
    return encoded_jwt
