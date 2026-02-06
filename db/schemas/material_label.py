from pydantic import BaseModel, ConfigDict
from typing import Optional

class MaterialLabelBase(BaseModel):
    """
    MaterialLabelBase: Base schema for linking materials and labels.
    """
    material_id: int
    label_id: int

class MaterialLabelCreate(MaterialLabelBase):
    """
    MaterialLabelCreate: Schema for creating a link.
    """
    pass

class MaterialLabelRead(MaterialLabelBase):
    """
    MaterialLabelRead: Schema for reading a link, includes ID.
    """
    id: int
    model_config = ConfigDict(from_attributes=True)
