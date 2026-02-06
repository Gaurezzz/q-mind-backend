import pytest
from db.repository import material_label as ml_repo
from db.repository import material as material_repo
from db.repository import label as label_repo
from db.repository import user as user_repo
from db.schemas.material_label import MaterialLabelCreate
from db.schemas.material import MaterialCreate
from db.schemas.label import LabelCreate
from db.schemas.user import UserCreate

class TestMaterialLabelRepository:
    """
    Tests for Many-to-Many relationship (Material <-> Label).
    """

    @pytest.fixture
    def setup_entities(self, db_session):
        user = user_repo.create_user(db_session, UserCreate(email="ml@test.com", password="Pass1234!"))
        
        mat = material_repo.create_material(db_session, MaterialCreate(
            name="Mat1", user_id=user.id, Eg_0K_eV=1, Alpha_evK=0.1, Beta_K=1, me_eff=1, mh_eff=1, epsilon_r=1
        ))
        
        label = label_repo.create_label(db_session, LabelCreate(name="Lbl1", user_id=user.id))
        
        return mat, label

    def test_link_material_to_label(self, db_session, setup_entities):
        """
        Scenario: Create an association between material and label.
        Expected: Association is created correctly.
        """
        material, label = setup_entities
        ml_in = MaterialLabelCreate(material_id=material.id, label_id=label.id)
        
        link = ml_repo.create_material_label(db_session, ml_in)
        
        assert link.material_id == material.id
        assert link.label_id == label.id

    def test_delete_association(self, db_session, setup_entities):
        """
        Scenario: Remove an association.
        Expected: Link is removed, but Material and Label remain.
        """
        material, label = setup_entities
        ml_in = MaterialLabelCreate(material_id=material.id, label_id=label.id)
        link = ml_repo.create_material_label(db_session, ml_in)
        
        ml_repo.delete_material_label(db_session, link.id)
        
        assert ml_repo.get_material_label(db_session, link.id) is None
        assert material_repo.get_material(db_session, material.id) is not None
        assert label_repo.get_label(db_session, label.id) is not None
