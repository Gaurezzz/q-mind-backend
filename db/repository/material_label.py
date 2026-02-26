from sqlalchemy.orm import Session
from db import schemas, models


def create_material_label(db: Session, material_label: schemas.material_label.MaterialLabelCreate) -> models.MaterialLabel:
    """
    create_material_label: Creates a new Material-Label association.

    Args:
        db (Session): Database session.
        material_label (MaterialLabelCreate): Association data.

    Returns:
        models.MaterialLabel: The created association.
    """
    db_ml = models.MaterialLabel(**material_label.model_dump())
    db.add(db_ml)
    db.commit()
    db.refresh(db_ml)
    return db_ml


def get_material_label(db: Session, material_label_id: int) -> models.MaterialLabel | None:
    """
    get_material_label: Retrieves an association by ID.

    Args:
        db (Session): Database session.
        material_label_id (int): ID of the association.

    Returns:
        models.MaterialLabel | None: The association if found, else None.
    """
    return db.query(models.MaterialLabel).filter(models.MaterialLabel.id == material_label_id).first()


def delete_material_label(db: Session, material_label_id: int) -> None:
    """
    delete_material_label: Deletes a Material-Label association by ID.

    Args:
        db (Session): Database session.
        material_label_id (int): ID of the association to delete.
    """
    db_ml = db.query(models.MaterialLabel).filter(models.MaterialLabel.id == material_label_id).first()
    if db_ml:
        db.delete(db_ml)
        db.commit()
