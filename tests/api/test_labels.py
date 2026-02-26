import pytest


class TestLabelsEndpoints:
    """
    Tests for /labels/ endpoints.
    """

    @pytest.fixture
    def label_payload(self, registered_user):
        return {
            "name": "Toxic",
            "user_id": registered_user["id"],
            "description": "Hazardous material"
        }

    @pytest.fixture
    def created_label(self, client, auth_headers, label_payload):
        response = client.post("/labels/", json=label_payload, headers=auth_headers)
        assert response.status_code == 201
        return response.json()

    def test_create_label(self, client, auth_headers, label_payload):
        """
        Scenario: Create a new label while authenticated.
        Expected: 201 Created with label data.
        """
        response = client.post("/labels/", json=label_payload, headers=auth_headers)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Toxic"
        assert "id" in data

    def test_create_label_without_auth(self, client, label_payload):
        """
        Scenario: Create a label without authentication.
        Expected: 401 Unauthorized.
        """
        response = client.post("/labels/", json=label_payload)

        assert response.status_code == 401

    def test_get_labels(self, client, auth_headers, created_label):
        """
        Scenario: Retrieve the list of labels for the authenticated user.
        Expected: 200 OK with a list containing at least the created label.
        """
        response = client.get("/labels/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert any(label["id"] == created_label["id"] for label in data)

    def test_get_labels_empty(self, client, auth_headers):
        """
        Scenario: Retrieve labels when none have been created.
        Expected: 200 OK with an empty list.
        """
        response = client.get("/labels/", headers=auth_headers)

        assert response.status_code == 200
        assert response.json() == []

    def test_get_label_by_id(self, client, auth_headers, created_label):
        """
        Scenario: Retrieve a specific label by its ID.
        Expected: 200 OK with the correct label data.
        """
        label_id = created_label["id"]
        response = client.get(f"/labels/{label_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == label_id
        assert data["name"] == created_label["name"]

    def test_get_nonexistent_label(self, client, auth_headers):
        """
        Scenario: Retrieve a label that does not exist.
        Expected: 404 Not Found.
        """
        response = client.get("/labels/99999", headers=auth_headers)

        assert response.status_code == 404

    def test_update_label(self, client, auth_headers, created_label):
        """
        Scenario: Update the name and description of an existing label.
        Expected: 200 OK with updated label data.
        """
        label_id = created_label["id"]
        response = client.put(f"/labels/{label_id}", json={
            "name": "NonToxic",
            "description": "Safe material"
        }, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "NonToxic"
        assert data["description"] == "Safe material"

    def test_update_nonexistent_label(self, client, auth_headers):
        """
        Scenario: Update a label that does not exist.
        Expected: 404 Not Found.
        """
        response = client.put("/labels/99999", json={"name": "Ghost"}, headers=auth_headers)

        assert response.status_code == 404

    def test_delete_label(self, client, auth_headers, created_label):
        """
        Scenario: Delete an existing label.
        Expected: 204 No Content, and subsequent GET returns 404.
        """
        label_id = created_label["id"]
        response = client.delete(f"/labels/{label_id}", headers=auth_headers)

        assert response.status_code == 204

        get_response = client.get(f"/labels/{label_id}", headers=auth_headers)
        assert get_response.status_code == 404

    def test_delete_nonexistent_label(self, client, auth_headers):
        """
        Scenario: Delete a label that does not exist.
        Expected: 404 Not Found.
        """
        response = client.delete("/labels/99999", headers=auth_headers)

        assert response.status_code == 404

    def test_delete_label_without_auth(self, client, created_label):
        """
        Scenario: Delete a label without authentication.
        Expected: 401 Unauthorized.
        """
        label_id = created_label["id"]
        response = client.delete(f"/labels/{label_id}")

        assert response.status_code == 401
