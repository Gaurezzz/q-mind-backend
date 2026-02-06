from sqlalchemy.orm import Session
from db import schemas, models 

def create_label(db: Session, label: schemas.label.LabelCreate) -> schemas.label.LabelRead:
    """
    create_label: Creates a new label.

    Args:
        db (Session): Database session.
        label (LabelCreate): Label data.

    Returns:
        LabelRead: The created label.
    """
    db_label = models.Label(**label.model_dump())
    db.add(db_label)
    db.commit()
    db.refresh(db_label)
    return db_label

def get_label(db: Session, label_id: int) -> schemas.label.LabelRead | None:
    """
    get_label: Retrieves a label by ID.

    Args:
        db (Session): Database session.
        label_id (int): ID of the label.

    Returns:
        LabelRead | None: The label if found.
    """
    return db.query(models.Label).filter(models.Label.id == label_id).first()

def get_labels(db: Session, skip: int = 0, limit: int = 100) -> list[schemas.label.LabelRead]:
    """
    get_labels: Retrieves a list of labels with pagination.

    Args:
        db (Session): Database session.
        skip (int): Skip count.
        limit (int): Max count.

    Returns:
        list[LabelRead]: List of labels.
    """
    return db.query(models.Label).offset(skip).limit(limit).all()

def delete_label(db: Session, label_id: int) -> None:
    """
    delete_label: Deletes a label by ID.

    Args:
        db (Session): Database session.
        label_id (int): ID of the label.
    """
    db_label = db.query(models.Label).filter(models.Label.id == label_id).first()
    if db_label:
        db.delete(db_label)
        db.commit()
    return

def update_label(db: Session, label_id: int, label: schemas.label.LabelCreate) -> schemas.label.LabelRead | None:
    """
    update_label: Updates an existing label.

    Args:
        db (Session): Database session.
        label_id (int): ID of the label.
        label (LabelCreate): New label data.

    Returns:
        LabelRead | None: The updated label if found.
    """
    db_label = db.query(models.Label).filter(models.Label.id == label_id).first()
    if db_label:
        for key, value in label.model_dump().items():
            setattr(db_label, key, value)
        db.commit()
        db.refresh(db_label)
        return db_label
    return None
