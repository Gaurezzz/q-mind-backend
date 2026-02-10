from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from db import schemas, models, repository
from db.config import get_db
from api.deps import get_current_user

router = APIRouter(
    prefix="/materials",
    tags=["materials"]
)

@router.post("/", response_model=schemas.material.MaterialRead, status_code=status.HTTP_201_CREATED)
def create_material(
    material: schemas.material.MaterialCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    create_material: Creates a new material.

    Args:
        material (MaterialCreate): Material creation data.
        db (Session): Database session.
        current_user (User): Authenticated user.

    Returns:
        MaterialRead: The created material.
    """
    return repository.material.create_material(db=db, material=material, user_id=current_user.id)

@router.get("/", response_model=List[schemas.material.MaterialRead])
def read_materials(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    read_materials: Retrieves a list of materials with pagination.

    Args:
        skip (int): Records to skip.
        limit (int): Max records to return.
        db (Session): Database session.
        current_user (User): Authenticated user.

    Returns:
        List[MaterialRead]: List of materials.
    """
    return repository.material.get_materials(db=db, user_id=current_user.id, skip=skip, limit=limit)

@router.get("/{material_id}", response_model=schemas.material.MaterialRead)
def read_material(
    material_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    read_material: Retrieves a specific material by ID.

    Args:
        material_id (int): Material ID.
        db (Session): Database session.
        current_user (User): Authenticated user.

    Returns:
        MaterialRead: The requested material.
    """
    db_material = repository.material.get_material(db=db, material_id=material_id, user_id=current_user.id)
    if not db_material:
        raise HTTPException(status_code=404, detail="Material was not found.")
    return db_material

@router.post("/{material_id}/labels/{label_id}", response_model=schemas.material.MaterialRead)
def add_label_to_material(
    material_id: int,
    label_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    add_label_to_material: Links a label to a material.

    Args:
        material_id (int): Material ID.
        label_id (int): Label ID.
        db (Session): Database session.
        current_user (User): Authenticated user.

    Returns:
        MaterialRead: The updated material with the new label.
    """
    db_material = repository.material.add_label_to_material(
        db=db, material_id=material_id, label_id=label_id, user_id=current_user.id
    )
    if not db_material:
        raise HTTPException(
            status_code=404, 
            detail="Make sure the material and label exist and belong to your account."
        )
    return db_material

@router.delete("/{material_id}/labels/{label_id}", response_model=schemas.material.MaterialRead)
def remove_label_from_material(
    material_id: int,
    label_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    remove_label_from_material: Removes a label from a material.

    Args:
        material_id (int): Material ID.
        label_id (int): Label ID.
        db (Session): Database session.
        current_user (User): Authenticated user.

    Returns:
        MaterialRead: The updated material without the label.
    """
    db_material = repository.material.remove_label_from_material(
        db=db, material_id=material_id, label_id=label_id, user_id=current_user.id
    )
    if not db_material:
        raise HTTPException(status_code=404, detail="Operation could not be completed. Make sure the material and label exist and belong to your account.")
    return db_material