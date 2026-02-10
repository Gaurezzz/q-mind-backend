from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from db import repository, schemas
from db.config import get_db, db_config
from api.security import create_access_token

router = APIRouter(
    prefix="/auth",
    tags=["auth"])

@router.post("/register", response_model=schemas.user.UserRead)
def register_user(user: schemas.user.UserCreate, db: Session = Depends(get_db)) -> schemas.user.UserRead:
    """
    register_user: Registers a new user.

    Args:
        user (UserCreate): User registration data.
        db (Session): Database session.

    Returns:
        UserRead: The newly created user.
    """
    db_user = repository.user.get_email(db=db, user_email=user.email)

    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email is already registered."
        )
    return repository.user.create_user(db=db, user=user)


@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    login: Authenticates a user and generates a JWT token.

    Args:
        form_data (OAuth2PasswordRequestForm): Login form data.
        db (Session): Database session.

    Returns:
        dict: Access token details.
    """
    user = repository.user.get_email(db=db, user_email=form_data.username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email is not registered."
        )
    if not repository.user.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The password entered is incorrect."
        )
    
    access_token_expires = timedelta(minutes=db_config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.id,
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token, 
        "token_type": "bearer"
        }

        