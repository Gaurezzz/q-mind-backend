import pytest
from db.repository import user as user_repo
from db.schemas.user import UserCreate, UserUpdate

class TestUserRepository:
    """
    Tests for User Repository Operations.
    """

    def test_create_user(self, db_session):
        """
        Scenario: Create a valid user.
        Expected: User is persisted, password is hashed.
        """
        user_in = UserCreate(email="mk@example.com", password="StrongPassword1!")
        
        user = user_repo.create_user(db_session, user_in)
        
        assert user.email == "mk@example.com"
        assert hasattr(user, "hashed_password")
        assert user.hashed_password != "StrongPassword1!"

    def test_get_user(self, db_session):
        """
        Scenario: Retrieve an existing user by ID.
        Expected: Correct user object is returned.
        """
        user_in = UserCreate(email="get@example.com", password="StrongPassword1!")
        created_user = user_repo.create_user(db_session, user_in)
        
        fetched_user = user_repo.get_user(db_session, created_user.id)
        
        assert fetched_user is not None
        assert fetched_user.id == created_user.id
        assert fetched_user.email == "get@example.com"

    def test_get_users_pagination(self, db_session):
        """
        Scenario: List users with pagination.
        Expected: Returns correct number of users.
        """
        for i in range(3):
            user_repo.create_user(db_session, UserCreate(email=f"u{i}@test.com", password="Pass1234!"))
        
        users = user_repo.get_users(db_session, limit=2)
        
        assert len(users) == 2

    def test_update_user_password(self, db_session):
        """
        Scenario: Update user password.
        Expected: Password hash changes.
        """
        user = user_repo.create_user(db_session, UserCreate(email="upd@test.com", password="OldPassword1!"))
        old_hash = user.hashed_password
        
        update_data = UserUpdate(password="NewPassword1!")
        updated_user = user_repo.update_user(db_session, user.id, update_data)
        
        assert updated_user.hashed_password != old_hash
        assert updated_user.hashed_password != "NewPassword1!"

    def test_delete_user(self, db_session):
        """
        Scenario: Delete an existing user.
        Expected: User is no longer retrievable.
        """
        user = user_repo.create_user(db_session, UserCreate(email="del@test.com", password="Pass1234!"))
        user_repo.delete_user(db_session, user.id)
        
        assert user_repo.get_user(db_session, user.id) is None

    def test_create_user_duplicate_email(self, db_session):
        """
        Scenario: Create user with duplicate email.
        Expected: Database raises an exception.
        """
        user_repo.create_user(db_session, UserCreate(email="dup@test.com", password="Pass1234!"))
        
        with pytest.raises(Exception):
            user_repo.create_user(db_session, UserCreate(email="dup@test.com", password="Pass5678!"))

    def test_get_nonexistent_user(self, db_session):
        """
        Scenario: Attempt to retrieve non-existent user.
        Expected: Returns None.
        """
        user = user_repo.get_user(db_session, 99999)
        
        assert user is None

    def test_update_nonexistent_user(self, db_session):
        """
        Scenario: Update non-existent user.
        Expected: Returns None.
        """
        update_data = UserUpdate(password="NewPass123!")
        result = user_repo.update_user(db_session, 99999, update_data)
        
        assert result is None

    def test_delete_nonexistent_user(self, db_session):
        """
        Scenario: Delete non-existent user.
        Expected: Operation completes without error.
        """
        user_repo.delete_user(db_session, 99999)
