import pytest
from db.repository import label as label_repo
from db.repository import user as user_repo
from db.schemas.label import LabelCreate
from db.schemas.user import UserCreate

class TestLabelRepository:
    """
    Tests for Label Repository Operations.
    """

    @pytest.fixture
    def owner(self, db_session):
        user_in = UserCreate(email="lbl_owner@test.com", password="StrongPassword1!")
        return user_repo.create_user(db_session, user_in)

    def test_create_label(self, db_session, owner):
        """
        Scenario: Create a new label.
        Expected: Label is created and linked to owner.
        """
        label_in = LabelCreate(name="Toxic", user_id=owner.id, description="Warning")
        label = label_repo.create_label(db_session, label_in, owner.id)
        
        assert label.name == "Toxic"
        assert label.user_id == owner.id

    def test_get_labels(self, db_session, owner):
        """
        Scenario: List all labels.
        Expected: All created labels are returned.
        """
        label_repo.create_label(db_session, LabelCreate(name="L1", user_id=owner.id), owner.id)
        label_repo.create_label(db_session, LabelCreate(name="L2", user_id=owner.id), owner.id)
        
        labels = label_repo.get_labels(db_session, owner.id)
        
        assert len(labels) == 2

    def test_update_label(self, db_session, owner):
        """
        Scenario: Update label name.
        Expected: Name is updated.
        """
        label = label_repo.create_label(db_session, LabelCreate(name="Old", user_id=owner.id), owner.id)
        
        update_in = LabelCreate(name="New", user_id=owner.id)
        updated = label_repo.update_label(db_session, label.id, update_in, owner.id)
        
        assert updated.name == "New"
        
    def test_delete_label(self, db_session, owner):
        """
        Scenario: Delete a label.
        Expected: Label is removed.
        """
        label = label_repo.create_label(db_session, LabelCreate(name="Del", user_id=owner.id), owner.id)
        
        label_repo.delete_label(db_session, label.id, owner.id)
        
        assert label_repo.get_label(db_session, label.id, owner.id) is None

    def test_create_label_invalid_user(self, db_session):
        """
        Scenario: Create label with non-existent user_id.
        Expected: Database raises foreign key violation.
        """
        with pytest.raises(Exception):
            label_repo.create_label(db_session, LabelCreate(name="Invalid", user_id=99999), 99999)

    def test_get_nonexistent_label(self, db_session, owner):
        """
        Scenario: Retrieve non-existent label.
        Expected: Returns None.
        """
        label = label_repo.get_label(db_session, 99999, owner.id)
        
        assert label is None

    def test_update_nonexistent_label(self, db_session, owner):
        """
        Scenario: Update non-existent label.
        Expected: Returns None.
        """
        update_data = LabelCreate(name="Nonexistent", user_id=owner.id)
        result = label_repo.update_label(db_session, 99999, update_data, owner.id)
        
        assert result is None

    def test_delete_nonexistent_label(self, db_session, owner):
        """
        Scenario: Delete non-existent label.
        Expected: Operation completes without error.
        """
        label_repo.delete_label(db_session, 99999, owner.id)
