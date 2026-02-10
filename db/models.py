from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import Optional

Base = declarative_base()

class User(Base):
    """
    User: Represents a system user.
    
    Stores authentication credentials and personal information.
    Owners of Labels and Materials.
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    labels: Mapped[list["Label"]] = relationship("Label", back_populates="owner")

class Material(Base):
    """
    Material: Represents a semiconductor material with physical properties.

    Used in solar cell simulations (BrusEngine). Many-to-Many relationship with Labels.
    """
    __tablename__ = "materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    Eg_0K_eV: Mapped[float] = mapped_column(Float, nullable=False)
    Alpha_evK: Mapped[float] = mapped_column(Float, nullable=False)
    Beta_K: Mapped[float] = mapped_column(Float, nullable=False)
    me_eff: Mapped[float] = mapped_column(Float, nullable=False)
    mh_eff: Mapped[float] = mapped_column(Float, nullable=False)
    epsilon_r: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    labels: Mapped[list["Label"]] = relationship(
        "Label",
        secondary="material_labels",
        back_populates="materials"
    )

class Label(Base):
    """
    Label: Categorization tag for materials.

    Allows grouping materials (e.g., 'Toxic', 'High Efficiency').
    """
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    owner: Mapped["User"] = relationship("User", back_populates="labels")

    materials: Mapped[list["Material"]] = relationship(
        "Material",
        secondary="material_labels",
        back_populates="labels"
    )

class MaterialLabel(Base):
    """
    MaterialLabel: Association table for Many-to-Many relationship between Materials and Labels.
    """
    __tablename__ = "material_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    material_id: Mapped[int] = mapped_column(Integer, ForeignKey("materials.id"), nullable=False)
    label_id: Mapped[int] = mapped_column(Integer, ForeignKey("labels.id"), nullable=False)
