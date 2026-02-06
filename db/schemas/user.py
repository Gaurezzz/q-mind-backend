from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from typing import Optional

class UserBase(BaseModel):
    """
    UserBase: Base Pydantic model for User data.
    """
    email: EmailStr = Field(..., description="Unique email address")
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserCreate(UserBase):
    """
    UserCreate: Schema for creating a new user, including sensitive data.
    """
    password: str = Field(..., min_length=8, description="Plain text password")
    
    @field_validator('password')
    @classmethod
    def validate_password_complexity(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one number')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

class UserRead(UserBase):
    """
    UserRead: Schema for reading user public profiles. Excludes password.
    """
    id: int
    model_config = ConfigDict(from_attributes=True)

class UserUpdate(BaseModel):
    """
    UserUpdate: Schema for updating user information.
    """
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None

    password: Optional[str] = Field(None, min_length=8, description="Plain text password")

    @field_validator('password')
    @classmethod
    def validate_password_complexity(cls, v):
        if v is not None:
            if not any(char.isdigit() for char in v):
                raise ValueError('Password must contain at least one number')
            if not any(char.isupper() for char in v):
                raise ValueError('Password must contain at least one uppercase letter')
        return v
