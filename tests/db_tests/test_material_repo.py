import pytest
from db.repository import material as material_repo
from db.repository import user as user_repo
from db.schemas.material import MaterialCreate
from db.schemas.user import UserCreate

class TestMaterialRepository:
    """
    Tests for Material Repository Operations.
    """

    @pytest.fixture
    def owner(self, db_session):
        user_in = UserCreate(email="mat_owner@test.com", password="StrongPassword1!")
        return user_repo.create_user(db_session, user_in)

    def test_create_material(self, db_session, owner):
        """
        Scenario: Create a valid material linked to a user.
        Expected: Material is persisted with correct user_id.
        """
        material_in = MaterialCreate(
            name="PbS",
            user_id=owner.id,
            Eg_0K_eV=0.41,
            Alpha_evK=0.0004,
            Beta_K=265.0,
            me_eff=0.09,
            mh_eff=0.088,
            epsilon_r=17.0
        )
        
        material = material_repo.create_material(db_session, material_in, owner.id)
        
        assert material.id is not None
        assert material.name == "PbS"
        assert material.user_id == owner.id

    def test_get_material(self, db_session, owner):
        """
        Scenario: Retrieve material by ID.
        Expected: Correct material is returned.
        """
        mat = material_repo.create_material(db_session, MaterialCreate(
            name="CdSe", user_id=owner.id, Eg_0K_eV=1.74, Alpha_evK=0.0, Beta_K=0.0, me_eff=1.0, mh_eff=1.0, epsilon_r=10.0
        ), owner.id)
        
        fetched = material_repo.get_material(db_session, mat.id, owner.id)
        
        assert fetched.name == "CdSe"

    def test_update_material(self, db_session, owner):
        """
        Scenario: Update material properties.
        Expected: Fields are updated correctly.
        """
        mat = material_repo.create_material(db_session, MaterialCreate(
            name="Si", user_id=owner.id, Eg_0K_eV=1.1, Alpha_evK=0, Beta_K=0, me_eff=1, mh_eff=1, epsilon_r=11.9
        ), owner.id)
        
        update_data = MaterialCreate(
            name="Silicon",
            user_id=owner.id, 
            Eg_0K_eV=1.1, Alpha_evK=0, Beta_K=0, me_eff=1, mh_eff=1, epsilon_r=12.0
        )
        updated = material_repo.update_material(db_session, mat.id, update_data, owner.id)
        
        assert updated.name == "Silicon"
        assert updated.epsilon_r == 12.0

    def test_delete_material(self, db_session, owner):
        """
        Scenario: Delete a material.
        Expected: Material is removed from DB.
        """
        mat = material_repo.create_material(db_session, MaterialCreate(
            name="ToDel", user_id=owner.id, Eg_0K_eV=1, Alpha_evK=0, Beta_K=0, me_eff=1, mh_eff=1, epsilon_r=1
        ), owner.id)
        
        material_repo.delete_material(db_session, mat.id, owner.id)
        
        assert material_repo.get_material(db_session, mat.id, owner.id) is None

    def test_create_material_invalid_user(self, db_session):
        """
        Scenario: Create material with non-existent user_id.
        Expected: Database raises foreign key violation.
        """
        with pytest.raises(Exception):
            material_repo.create_material(db_session, MaterialCreate(
                name="Invalid", user_id=99999, Eg_0K_eV=1, Alpha_evK=0, Beta_K=0, me_eff=1, mh_eff=1, epsilon_r=1
            ), 99999)

    def test_get_nonexistent_material(self, db_session, owner):
        """
        Scenario: Retrieve non-existent material.
        Expected: Returns None.
        """
        material = material_repo.get_material(db_session, 99999, owner.id)
        
        assert material is None

    def test_update_nonexistent_material(self, db_session, owner):
        """
        Scenario: Update non-existent material.
        Expected: Returns None.
        """
        update_data = MaterialCreate(
            name="Nonexistent", user_id=owner.id, Eg_0K_eV=1, Alpha_evK=0, Beta_K=0, me_eff=1, mh_eff=1, epsilon_r=1
        )
        result = material_repo.update_material(db_session, 99999, update_data, owner.id)
        
        assert result is None

    def test_delete_nonexistent_material(self, db_session, owner):
        """
        Scenario: Delete non-existent material.
        Expected: Operation completes without error.
        """
        material_repo.delete_material(db_session, 99999, owner.id)
