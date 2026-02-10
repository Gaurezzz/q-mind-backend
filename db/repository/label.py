from sqlalchemy.orm import Session
from db import schemas, models 

def create_label(db: Session, label: schemas.label.LabelCreate, user_id: int) -> models.Label | None:
    """
    create_label: Creates a new label.

    Args:
        db (Session): Database session.
        label (LabelCreate): Label data.
        user_id (int): ID of the user creating the label.
    Returns:
        models.Label: The created label.
    """
    db_label = models.Label(**label.model_dump(), user_id=user_id)
    db.add(db_label)
    db.commit()
    db.refresh(db_label)
    return db_label

def get_label(db: Session, label_id: int, user_id: int) -> models.Label | None:
    """
    get_label: Retrieves a label by ID.

    Args:
        db (Session): Database session.
        label_id (int): ID of the label.

    Returns:
        models.Label | None: The label if found.
    """
    db_label = db.query(models.Label).filter(models.Label.id == label_id, models.Label.user_id == user_id).first()
    return db_label

def get_labels(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> list[models.Label]:
    """
    get_labels: Retrieves a list of labels with pagination.

    Args:
        db (Session): Database session.
        skip (int): Skip count.
        limit (int): Max count.
        user_id (int): ID of the user.

    Returns:
        list[models.Label]: List of labels.
    """
    labels = db.query(models.Label).filter(models.Label.user_id == user_id).offset(skip).limit(limit).all()
    return labels

def delete_label(db: Session, label_id: int, user_id: int) -> None:
    """
    delete_label: Deletes a label by ID.

    Args:
        db (Session): Database session.
        label_id (int): ID of the label.
        user_id (int): ID of the user.
    """
    db_label = db.query(models.Label).filter(models.Label.id == label_id, models.Label.user_id == user_id).first()
    if db_label:
        db.delete(db_label)
        db.commit()
    return

def update_label(db: Session, label_id: int, label: schemas.label.LabelUpdate, user_id: int) -> models.Label | None:
    """
    update_label: Updates an existing label.

    Args:
        db (Session): Database session.
        label_id (int): ID of the label.
        label (LabelUpdate): New label data.
        user_id (int): ID of the user.

    Returns:
        models.Label | None: The updated label if found.
    """
    db_label = db.query(models.Label).filter(models.Label.id == label_id, models.Label.user_id == user_id).first()
    if db_label:
        for key, value in label.model_dump(exclude_unset=True).items():
            setattr(db_label, key, value)
        db.commit()
        db.refresh(db_label)
        return db_label
    return None
