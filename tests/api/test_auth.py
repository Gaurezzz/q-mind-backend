import pytest


class TestAuthEndpoints:
    """
    Tests for /auth/ endpoints.
    """

    def test_register_new_user(self, client):
        """
        Scenario: Register a brand new user.
        Expected: 200 OK with user data (id and email).
        """
        response = client.post("/auth/register", json={
            "email": "newuser@test.com",
            "password": "StrongPassword1!"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newuser@test.com"
        assert "id" in data

    def test_register_duplicate_email(self, client):
        """
        Scenario: Register with an email already in use.
        Expected: 400 Bad Request.
        """
        payload = {"email": "dup@test.com", "password": "StrongPassword1!"}
        client.post("/auth/register", json=payload)

        response = client.post("/auth/register", json=payload)

        assert response.status_code == 400

    def test_login_valid_credentials(self, client, registered_user):
        """
        Scenario: Login with correct email and password.
        Expected: 200 OK with a bearer access_token.
        """
        response = client.post("/auth/login", data={
            "username": "testuser@test.com",
            "password": "StrongPassword1!"
        })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, registered_user):
        """
        Scenario: Login with incorrect password.
        Expected: 400 Bad Request.
        """
        response = client.post("/auth/login", data={
            "username": "testuser@test.com",
            "password": "WrongPassword1!"
        })

        assert response.status_code == 400

    def test_login_unregistered_email(self, client):
        """
        Scenario: Login with an email that does not exist.
        Expected: 400 Bad Request.
        """
        response = client.post("/auth/login", data={
            "username": "ghost@test.com",
            "password": "StrongPassword1!"
        })

        assert response.status_code == 400

    def test_protected_route_without_token(self, client):
        """
        Scenario: Access a protected endpoint without providing a token.
        Expected: 401 Unauthorized.
        """
        response = client.get("/labels/")

        assert response.status_code == 401

    def test_protected_route_with_invalid_token(self, client):
        """
        Scenario: Access a protected endpoint with a malformed token.
        Expected: 401 Unauthorized.
        """
        headers = {"Authorization": "Bearer this.is.not.a.valid.token"}
        response = client.get("/labels/", headers=headers)

        assert response.status_code == 401
