from sqlalchemy.orm import Session
from db import schemas, models 

def create_material_label(db: Session, material_label: schemas.material_label.MaterialLabelCreate) -> schemas.material_label.MaterialLabelRead:
    """
    create_material_label: Creates a new material-label association (Many-to-Many).

    Args:
        db (Session): Database session.
        material_label (MaterialLabelCreate): Association data.

    Returns:
        MaterialLabelRead: The created association.
    """
    db_material_label = models.MaterialLabel(**material_label.model_dump())
    db.add(db_material_label)
    db.commit()
    db.refresh(db_material_label)
    return db_material_label

def get_material_label(db: Session, material_label_id: int) -> schemas.material_label.MaterialLabelRead | None:
    """
    get_material_label: Retrieves a material-label association by its ID.

    Args:
        db (Session): Database session.
        material_label_id (int): ID of the association.

    Returns:
        MaterialLabelRead | None: The association if found.
    """
    return db.query(models.MaterialLabel).filter(models.MaterialLabel.id == material_label_id).first()

def get_material_labels(db: Session, skip: int = 0, limit: int = 100) -> list[schemas.material_label.MaterialLabelRead]:
    """
    get_material_labels: Retrieves a list of associations with pagination.

    Args:
        db (Session): Database session.
        skip (int): Skip count.
        limit (int): Max count.

    Returns:
        list[MaterialLabelRead]: List of associations.
    """
    return db.query(models.MaterialLabel).offset(skip).limit(limit).all()

def delete_material_label(db: Session, material_label_id: int) -> None:
    """
    delete_material_label: Deletes a material-label association.

    Args:
        db (Session): Database session.
        material_label_id (int): ID of the association.
    """
    db_material_label = db.query(models.MaterialLabel).filter(models.MaterialLabel.id == material_label_id).first()
    if db_material_label:
        db.delete(db_material_label)
        db.commit()
    return
