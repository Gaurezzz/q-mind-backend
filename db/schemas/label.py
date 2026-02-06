from pydantic import BaseModel, ConfigDict
from typing import Optional

class LabelBase(BaseModel):
    """
    LabelBase: Base Pydantic model for labels.
    """
    name: str
    user_id: int
    description: Optional[str] = None

class LabelCreate(LabelBase):
    """
    LabelCreate: Schema for creating a new label.
    """
    pass

class LabelRead(LabelBase):
    """
    LabelRead: Schema for reading label details, includes ID.
    """
    id: int
    model_config = ConfigDict(from_attributes=True)

class LabelUpdate(BaseModel):
    """
    LabelUpdate: Schema for updating label information.
    """
    name: Optional[str] = None
    user_id: Optional[int] = None
    description: Optional[str] = None