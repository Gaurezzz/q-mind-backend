import pytest


MATERIAL_DATA = {
    "name": "PbS",
    "Eg_0K_eV": 0.41,
    "Alpha_evK": 0.0004,
    "Beta_K": 265.0,
    "me_eff": 0.09,
    "mh_eff": 0.088,
    "epsilon_r": 17.0,
    "description": "Lead Sulfide quantum dot"
}


class TestMaterialEndpoints:
    """
    Tests for /materials/ endpoints.
    """

    @pytest.fixture
    def material_payload(self, registered_user):
        return {**MATERIAL_DATA, "user_id": registered_user["id"]}

    @pytest.fixture
    def created_material(self, client, auth_headers, material_payload):
        response = client.post("/materials/", json=material_payload, headers=auth_headers)
        assert response.status_code == 201
        return response.json()

    @pytest.fixture
    def created_label(self, client, auth_headers, registered_user):
        response = client.post("/labels/", json={
            "name": "Quantum",
            "user_id": registered_user["id"]
        }, headers=auth_headers)
        assert response.status_code == 201
        return response.json()

    def test_create_material(self, client, auth_headers, material_payload):
        """
        Scenario: Create a new material while authenticated.
        Expected: 201 Created with material data.
        """
        response = client.post("/materials/", json=material_payload, headers=auth_headers)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "PbS"
        assert data["Eg_0K_eV"] == 0.41
        assert "id" in data

    def test_create_material_without_auth(self, client, material_payload):
        """
        Scenario: Create a material without authentication.
        Expected: 401 Unauthorized.
        """
        response = client.post("/materials/", json=material_payload)

        assert response.status_code == 401

    def test_get_materials(self, client, auth_headers, created_material):
        """
        Scenario: Retrieve list of materials for the authenticated user.
        Expected: 200 OK with a list containing the created material.
        """
        response = client.get("/materials/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert any(m["id"] == created_material["id"] for m in data)

    def test_get_materials_empty(self, client, auth_headers):
        """
        Scenario: Retrieve materials when none have been created.
        Expected: 200 OK with an empty list.
        """
        response = client.get("/materials/", headers=auth_headers)

        assert response.status_code == 200
        assert response.json() == []

    def test_get_materials_without_auth(self, client):
        """
        Scenario: Retrieve materials without authentication.
        Expected: 401 Unauthorized.
        """
        response = client.get("/materials/")

        assert response.status_code == 401

    def test_get_material_by_id(self, client, auth_headers, created_material):
        """
        Scenario: Retrieve a specific material by its ID.
        Expected: 200 OK with the correct material data.
        """
        material_id = created_material["id"]
        response = client.get(f"/materials/{material_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == material_id
        assert data["name"] == created_material["name"]

    def test_get_nonexistent_material(self, client, auth_headers):
        """
        Scenario: Retrieve a material that does not exist.
        Expected: 404 Not Found.
        """
        response = client.get("/materials/99999", headers=auth_headers)

        assert response.status_code == 404

    def test_add_label_to_material(self, client, auth_headers, created_material, created_label):
        """
        Scenario: Associate an existing label with an existing material.
        Expected: 200 OK with the material reflecting the new label.
        """
        material_id = created_material["id"]
        label_id = created_label["id"]
        response = client.post(
            f"/materials/{material_id}/labels/{label_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == material_id

    def test_add_nonexistent_label_to_material(self, client, auth_headers, created_material):
        """
        Scenario: Associate a non-existent label with a material.
        Expected: 404 Not Found.
        """
        material_id = created_material["id"]
        response = client.post(
            f"/materials/{material_id}/labels/99999",
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_remove_label_from_material(self, client, auth_headers, created_material, created_label):
        """
        Scenario: Remove an associated label from a material.
        Expected: 200 OK with the material no longer containing that label.
        """
        material_id = created_material["id"]
        label_id = created_label["id"]

        client.post(f"/materials/{material_id}/labels/{label_id}", headers=auth_headers)

        response = client.delete(
            f"/materials/{material_id}/labels/{label_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == material_id

    def test_remove_label_from_nonexistent_material(self, client, auth_headers, created_label):
        """
        Scenario: Remove a label from a material that does not exist.
        Expected: 404 Not Found.
        """
        label_id = created_label["id"]
        response = client.delete(
            f"/materials/99999/labels/{label_id}",
            headers=auth_headers
        )

        assert response.status_code == 404
