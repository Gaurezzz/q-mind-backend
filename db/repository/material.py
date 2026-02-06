from sqlalchemy.orm import Session
from db import schemas, models 

def create_material(db: Session, material: schemas.material.MaterialCreate) -> schemas.material.MaterialRead:
    """
    create_material: Creates a new material in the database.

    Args:
        db (Session): Database session.
        material (MaterialCreate): Material data.

    Returns:
        MaterialRead: The created material.
    """
    db_material = models.Material(**material.model_dump())
    db.add(db_material)
    db.commit()
    db.refresh(db_material)
    return db_material

def get_material(db: Session, material_id: int) -> schemas.material.MaterialRead | None:
    """
    get_material: Retrieves a material by ID.

    Args:
        db (Session): Database session.
        material_id (int): ID of the material.

    Returns:
        MaterialRead | None: The material if found, else None.
    """
    return db.query(models.Material).filter(models.Material.id == material_id).first()

def get_materials(db: Session, skip: int = 0, limit: int = 100) -> list[schemas.material.MaterialRead]:
    """
    get_materials: Retrieves a list of materials with pagination.

    Args:
        db (Session): Database session.
        skip (int): Records to skip.
        limit (int): Max records to return.

    Returns:
        list[MaterialRead]: List of materials.
    """
    return db.query(models.Material).offset(skip).limit(limit).all()

def delete_material(db: Session, material_id: int) -> None:
    """
    delete_material: Deletes a material by ID.

    Args:
        db (Session): Database session.
        material_id (int): ID of the material to delete.
    """
    db_material = db.query(models.Material).filter(models.Material.id == material_id).first()
    if db_material:
        db.delete(db_material)
        db.commit()
    return

def update_material(db: Session, material_id: int, material: schemas.material.MaterialCreate) -> schemas.material.MaterialRead | None:
    """
    update_material: Updates an existing material.

    Args:
        db (Session): Database session.
        material_id (int): ID of the material.
        material (MaterialCreate): New material data.

    Returns:
        MaterialRead | None: The updated material if found.
    """
    db_material = db.query(models.Material).filter(models.Material.id == material_id).first()
    if db_material:
        for key, value in material.model_dump().items():
            setattr(db_material, key, value)
        db.commit()
        db.refresh(db_material)
        return db_material
    return None
