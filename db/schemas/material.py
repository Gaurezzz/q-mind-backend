from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class MaterialBase(BaseModel):
    """
    MaterialBase: Base Pydantic model for semiconductor material properties.
    """
    name: str = Field(..., min_length=1, max_length=50, description="Chemical formula/name")
    user_id: int = Field(..., description="ID of the owner user")
    Eg_0K_eV: float = Field(..., gt=0, description="Bandgap energy at 0K in eV")
    Alpha_evK: float = Field(..., ge=0, description="Varshni parameter Alpha")
    Beta_K: float = Field(..., ge=0, description="Varshni parameter Beta")
    me_eff: float = Field(..., gt=0, description="Effective electron mass")
    mh_eff: float = Field(..., gt=0, description="Effective hole mass")
    epsilon_r: float = Field(..., gt=0, description="Relative permittivity")
    description: Optional[str] = None

class MaterialCreate(MaterialBase):
    """
    MaterialCreate: Schema for creating a new material.
    """
    pass

class MaterialRead(MaterialBase):
    """
    MaterialRead: Schema for reading material properties, includes ID.
    """
    id: int
    model_config = ConfigDict(from_attributes=True)

class MaterialUpdate(BaseModel):
    """
    MaterialUpdate: Schema for updating material properties.
    """
    name: Optional[str] = Field(None, min_length=1, max_length=50, description="Chemical formula/name")
    Eg_0K_eV: Optional[float] = Field(None, gt=0, description="Bandgap energy at 0K in eV")
    Alpha_evK: Optional[float] = Field(None, ge=0, description="Varshni parameter Alpha")
    Beta_K: Optional[float] = Field(None, ge=0, description="Varshni parameter Beta")
    me_eff: Optional[float] = Field(None, gt=0, description="Effective electron mass")
    mh_eff: Optional[float] = Field(None, gt=0, description="Effective hole mass")
    epsilon_r: Optional[float] = Field(None, gt=0, description="Relative permittivity")
    description: Optional[str] = None
