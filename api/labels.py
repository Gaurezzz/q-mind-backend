from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from db import schemas, models, repository
from db.config import get_db
from api.deps import get_current_user

router = APIRouter(
    prefix="/labels",
    tags=["labels"]
)

@router.post("/", response_model=schemas.label.LabelRead, status_code=status.HTTP_201_CREATED)
def create_label(
    label: schemas.label.LabelCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    create_label: Create a new label for the user.
    
    Args:
        label (LabelCreate): Label data.
        db (Session): Database session.
        current_user (User): Authenticated user.
        
    Returns:
        LabelRead: The created label.
    """
    return repository.label.create_label(db=db, label=label, user_id=current_user.id)

@router.get("/", response_model=List[schemas.label.LabelRead])
def read_labels(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    read_labels: Retrieve all labels for the user with pagination.
    
    Args:
        skip (int): Records to skip.
        limit (int): Max records to return.
        db (Session): Database session.
        current_user (User): Authenticated user.
        
    Returns:
        List[LabelRead]: List of labels.
    """
    return repository.label.get_labels(db=db, user_id=current_user.id, skip=skip, limit=limit)

@router.get("/{label_id}", response_model=schemas.label.LabelRead)
def read_label(
    label_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    read_label: Retrieve a specific label by ID.
    
    Args:
        label_id (int): ID of the label.
        db (Session): Database session.
        current_user (User): Authenticated user.
        
    Returns:
        LabelRead: The requested label.
    """
    db_label = repository.label.get_label(db=db, label_id=label_id, user_id=current_user.id)
    if not db_label:
        raise HTTPException(status_code=404, detail="Label not found.")
    return db_label

@router.put("/{label_id}", response_model=schemas.label.LabelRead)
def update_label(
    label_id: int,
    label_update: schemas.label.LabelUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    update_label: Update a label.
    
    Args:
        label_id (int): ID of the label.
        label_update (LabelUpdate): Data to update.
        db (Session): Database session.
        current_user (User): Authenticated user.
        
    Returns:
        LabelRead: The updated label.
    """
    db_label = repository.label.update_label(
        db=db, label_id=label_id, label=label_update, user_id=current_user.id
    )
    if not db_label:
        raise HTTPException(status_code=404, detail="Could not update the label. Make sure it exists and belongs to your account.")
    return db_label

@router.delete("/{label_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_label(
    label_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    delete_label: Delete a label.
    
    Args:
        label_id (int): ID of the label.
        db (Session): Database session.
        current_user (User): Authenticated user.
    """
    db_label = repository.label.get_label(db=db, label_id=label_id, user_id=current_user.id)
    if not db_label:
        raise HTTPException(status_code=404, detail="Label not found. Make sure it exists and belongs to your account.")
    
    repository.label.delete_label(db=db, label_id=label_id, user_id=current_user.id)
    return None