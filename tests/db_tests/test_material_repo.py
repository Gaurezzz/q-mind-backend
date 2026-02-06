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
        
        material = material_repo.create_material(db_session, material_in)
        
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
        ))
        
        fetched = material_repo.get_material(db_session, mat.id)
        
        assert fetched.name == "CdSe"

    def test_update_material(self, db_session, owner):
        """
        Scenario: Update material properties.
        Expected: Fields are updated correctly.
        """
        mat = material_repo.create_material(db_session, MaterialCreate(
            name="Si", user_id=owner.id, Eg_0K_eV=1.1, Alpha_evK=0, Beta_K=0, me_eff=1, mh_eff=1, epsilon_r=11.9
        ))
        
        update_data = MaterialCreate(
            name="Silicon",
            user_id=owner.id, 
            Eg_0K_eV=1.1, Alpha_evK=0, Beta_K=0, me_eff=1, mh_eff=1, epsilon_r=12.0
        )
        updated = material_repo.update_material(db_session, mat.id, update_data)
        
        assert updated.name == "Silicon"
        assert updated.epsilon_r == 12.0

    def test_delete_material(self, db_session, owner):
        """
        Scenario: Delete a material.
        Expected: Material is removed from DB.
        """
        mat = material_repo.create_material(db_session, MaterialCreate(
            name="ToDel", user_id=owner.id, Eg_0K_eV=1, Alpha_evK=0, Beta_K=0, me_eff=1, mh_eff=1, epsilon_r=1
        ))
        
        material_repo.delete_material(db_session, mat.id)
        
        assert material_repo.get_material(db_session, mat.id) is None
