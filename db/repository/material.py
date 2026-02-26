from sqlalchemy.orm import Session
from db import schemas, models 

def create_material(db: Session, material: schemas.material.MaterialCreate, user_id: int) -> models.Material | None:
    """
    create_material: Creates a new material in the database.

    Args:
        db (Session): Database session.
        material (MaterialCreate): Material data.
        user_id (int): ID of the user creating the material.
    Returns:
        models.Material: The created material.
    """
    db_material = models.Material(**material.model_dump(exclude={'user_id'}), user_id=user_id)
    db.add(db_material)
    db.commit()
    db.refresh(db_material)
    return db_material

def get_material(db: Session, material_id: int, user_id: int) -> models.Material | None:
    """
    get_material: Retrieves a material by ID.

    Args:
        db (Session): Database session.
        material_id (int): ID of the material.

    Returns:
        models.Material | None: The material if found, else None.
    """
    db_material = db.query(models.Material).filter(models.Material.id == material_id, models.Material.user_id == user_id).first()
    return db_material

def get_materials(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> list[models.Material]:
    """
    get_materials: Retrieves a list of materials with pagination.

    Args:
        db (Session): Database session.
        user_id (int): ID of the user.
        skip (int): Records to skip.
        limit (int): Max records to return.

    Returns:
        list[models.Material]: List of materials.
    """
    materials = db.query(models.Material).filter(models.Material.user_id == user_id).offset(skip).limit(limit).all()
    return materials

def delete_material(db: Session, material_id: int, user_id: int) -> None:
    """
    delete_material: Deletes a material by ID.

    Args:
        db (Session): Database session.
        material_id (int): ID of the material to delete.
        user_id (int): ID of the user.
    """
    db_material = db.query(models.Material).filter(models.Material.id == material_id, models.Material.user_id == user_id).first()
    if db_material:
        db.delete(db_material)
        db.commit()
    return

def update_material(db: Session, material_id: int, material: schemas.material.MaterialUpdate, user_id: int) -> models.Material | None:
    """
    update_material: Updates an existing material.

    Args:
        db (Session): Database session.
        material_id (int): ID of the material.
        material (MaterialUpdate): New material data.
        user_id (int): ID of the user.

    Returns:
        models.Material | None: The updated material if found.
    """
    db_material = db.query(models.Material).filter(models.Material.id == material_id, models.Material.user_id == user_id).first()
    if db_material:
        for key, value in material.model_dump(exclude_unset=True).items():
            setattr(db_material, key, value)
        db.commit()
        db.refresh(db_material)
        return db_material
    return None

def add_label_to_material(db: Session, material_id: int, label_id: int, user_id: int) -> models.Material | None:
    """
    add_label_to_material: Adds a label to a material.

    Args:
        db (Session): Database session.
        material_id (int): ID of the material.
        label_id (int): ID of the label to add.
        user_id (int): ID of the user.

    Returns:
        models.Material | None: The updated material if found.
    """
    db_material = db.query(models.Material).filter(models.Material.id == material_id, models.Material.user_id == user_id).first()
    db_label = db.query(models.Label).filter(models.Label.id == label_id, models.Label.user_id == user_id).first()
    
    if db_material and db_label:
        if db_label not in db_material.labels:
            db_material.labels.append(db_label)
            db.commit()
            db.refresh(db_material)
        return db_material
    return None

def remove_label_from_material(db: Session, material_id: int, label_id: int, user_id: int) -> models.Material | None:
    """
    remove_label_from_material: Removes a label from a material.

    Args:
        db (Session): Database session.
        material_id (int): ID of the material.
        label_id (int): ID of the label to remove.
        user_id (int): ID of the user.

    Returns:
        models.Material | None: The updated material if found.
    """
    db_material = db.query(models.Material).filter(models.Material.id == material_id, models.Material.user_id == user_id).first()
    db_label = db.query(models.Label).filter(models.Label.id == label_id, models.Label.user_id == user_id).first()
    
    if db_material and db_label:
        if db_label in db_material.labels:
            db_material.labels.remove(db_label)
            db.commit()
            db.refresh(db_material)
        return db_material
    return None