from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from db.config import db_config, get_db
from db import repository, models
from db import schemas

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
) -> models.User | None:
    """
    get_current_user: Retrieves the current authenticated user from the JWT token.

    Args:
        token (str): JWT token.
        db (Session): Database session.

    Returns:
        models.User | None: The authenticated user or None.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token=token,
            key=db_config.SECRET_KEY,
            algorithms=[db_config.ALGORITHM]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user=repository.user.get_user(db, user_id=int(user_id))
    if user is None:
        raise credentials_exception
    return user