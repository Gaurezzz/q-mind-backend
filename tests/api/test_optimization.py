import pytest


MINIMAL_PARAMS = {
    "population_size": 5,
    "max_iterations": 2,
    "crossover_alpha": 0.5,
    "mutation_strength": 0.1,
    "operating_temperature": 300.0,
}


class TestOptimizationEndpoint:
    """
    Tests for POST /optimization/run endpoint.
    """

    def test_run_optimization_single_material(self, client, auth_headers):
        """
        Scenario: Run optimization with a single valid material.
        Expected: 200 OK with a complete OptimizationResponse.
        """
        payload = {**MINIMAL_PARAMS, "materials": ["CdSe"]}
        response = client.post("/optimization/run", json=payload, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "COMPLETED"
        assert len(data["optimal_radii_nm"]) == 1
        assert len(data["fitness_history"]) == MINIMAL_PARAMS["max_iterations"]
        assert 2.0 <= data["optimal_radii_nm"][0] <= 10.0
        assert "simulation_id" in data
        assert data["computation_time_ms"] >= 0

    def test_run_optimization_tandem(self, client, auth_headers):
        """
        Scenario: Run optimization with two materials (tandem architecture).
        Expected: 200 OK with two optimal radii.
        """
        payload = {**MINIMAL_PARAMS, "materials": ["CdSe", "PbS"]}
        response = client.post("/optimization/run", json=payload, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["optimal_radii_nm"]) == 2
        assert all(2.0 <= r <= 10.0 for r in data["optimal_radii_nm"])

    def test_run_optimization_three_layers(self, client, auth_headers):
        """
        Scenario: Run optimization with three materials (multi-junction).
        Expected: 200 OK with three optimal radii.
        """
        payload = {**MINIMAL_PARAMS, "materials": ["CdS", "CdSe", "GaAs"]}
        response = client.post("/optimization/run", json=payload, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["optimal_radii_nm"]) == 3

    def test_run_optimization_invalid_material(self, client, auth_headers):
        """
        Scenario: Request optimization with a material not in the catalog.
        Expected: 422 Unprocessable Entity.
        """
        payload = {**MINIMAL_PARAMS, "materials": ["UnknownMaterial"]}
        response = client.post("/optimization/run", json=payload, headers=auth_headers)

        assert response.status_code == 422

    def test_run_optimization_mixed_valid_invalid_materials(self, client, auth_headers):
        """
        Scenario: Request with one valid and one invalid material.
        Expected: 422 Unprocessable Entity.
        """
        payload = {**MINIMAL_PARAMS, "materials": ["CdSe", "NotAMaterial"]}
        response = client.post("/optimization/run", json=payload, headers=auth_headers)

        assert response.status_code == 422

    def test_run_optimization_without_auth(self, client):
        """
        Scenario: Call the endpoint without a JWT token.
        Expected: 401 Unauthorized.
        """
        payload = {**MINIMAL_PARAMS, "materials": ["CdSe"]}
        response = client.post("/optimization/run", json=payload)

        assert response.status_code == 401

    def test_run_optimization_with_invalid_token(self, client):
        """
        Scenario: Call the endpoint with a malformed token.
        Expected: 401 Unauthorized.
        """
        headers = {"Authorization": "Bearer not.a.valid.token"}
        payload = {**MINIMAL_PARAMS, "materials": ["CdSe"]}
        response = client.post("/optimization/run", json=payload, headers=headers)

        assert response.status_code == 401

    def test_response_simulation_id_is_unique(self, client, auth_headers):
        """
        Scenario: Run two optimizations back to back.
        Expected: Each response has a different simulation_id.
        """
        payload = {**MINIMAL_PARAMS, "materials": ["GaAs"]}
        r1 = client.post("/optimization/run", json=payload, headers=auth_headers)
        r2 = client.post("/optimization/run", json=payload, headers=auth_headers)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["simulation_id"] != r2.json()["simulation_id"]

    def test_fitness_history_length_matches_iterations(self, client, auth_headers):
        """
        Scenario: Request 3 iterations.
        Expected: fitness_history contains exactly 3 values.
        """
        payload = {**MINIMAL_PARAMS, "materials": ["CdSe"], "max_iterations": 3}
        response = client.post("/optimization/run", json=payload, headers=auth_headers)

        assert response.status_code == 200
        assert len(response.json()["fitness_history"]) == 3

    def test_default_values_are_accepted(self, client, auth_headers):
        """
        Scenario: Send only the required field (materials), relying on schema defaults.
        Expected: 200 OK — all optional fields have valid defaults.
        """
        response = client.post(
            "/optimization/run",
            json={"materials": ["CdSe"], "population_size": 5, "max_iterations": 2},
            headers=auth_headers
        )

        assert response.status_code == 200
