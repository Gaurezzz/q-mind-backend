from sqlalchemy.orm import Session
from passlib.context import CryptContext
from db import schemas, models

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_user(db: Session, user: schemas.user.UserCreate) -> models.User | None:
    """
    create_user: Creates a new user in the database with hashed password.

    Args:
        db (Session): Database session.
        user (UserCreate): User data to creation.

    Returns:
        UserRead: The created user.
    """
    hashed_password = get_password_hash(user.password)

    user_data = user.model_dump(exclude={"password"})
    db_user = models.User(**user_data, hashed_password=hashed_password)
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int) -> models.User | None:
    """
    get_user: Retrieves a user by ID.

    Args:
        db (Session): Database session.
        user_id (int): ID of the user to retrieve.
    
    Returns:
        UserRead | None: The user if found, else None.
    """
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    return db_user

def get_email(db: Session, user_email: str) -> models.User | None:
    """
    get_email: Retrieves a user by their email

    Args:
        db (Session): Database session.
        email (str): email of the user to retrieve.
    
    Returns:
        UserRead | None: The user if found, else None.
    """
    db_user = db.query(models.User).filter(models.User.email == user_email).first()
    return db_user

def get_users(db: Session, skip: int = 0, limit: int = 100) -> list[models.User] | None:
    """
    get_users: Retrieves a list of users with pagination.

    Args:
        db (Session): Database session.
        skip (int): Number of records to skip.
        limit (int): Maximum number of records to return.

    Returns:
        list[UserRead]: List of users.
    """
    users = db.query(models.User).offset(skip).limit(limit).all()
    return users

def delete_user(db: Session, user_id: int) -> None:
    """
    delete_user: Deletes a user by ID.

    Args:
        db (Session): Database session.
        user_id (int): ID of the user to delete.
    """
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
    return

def update_user(db: Session, user_id: int, user: schemas.user.UserUpdate) -> models.User | None:
    """
    update_user: Updates an existing user's information, hashing password if changed.

    Args:
        db (Session): Database session.
        user_id (int): ID of the user to update.
        user (UserUpdate): New user data.

    Returns:
        UserRead | None: The updated user if found, else None.
    """
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        update_data = user.model_dump(exclude_unset=True)
        
        if "password" in update_data:
            hashed = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed
            
        for key, value in update_data.items():
            setattr(db_user, key, value)
            
        db.commit()
        db.refresh(db_user)
        return db_user
    return None
