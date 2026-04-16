import pytest
import numpy as np
import mindspore as ms

from modelarts_worker.logic.DataAnalyzer import DataAnalyzer
from db.schemas.optimization import OptimizationRequest


CSV_PATH = "db/materials.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**overrides) -> OptimizationRequest:
    """Builds a minimal but valid OptimizationRequest."""
    defaults = dict(
        materials=["CdSe", "PbS"],
        population_size=10,
        max_iterations=3,
        crossover_alpha=0.5,
        mutation_strength=0.1,
        operating_temperature=300.0,
        wavelength_input_csv=False,
        wavelength_left_bound=280.0,
        wavelength_right_bound=860.0,
        wavelength_step=20.0,
    )
    defaults.update(overrides)
    return OptimizationRequest(**defaults)


class TestDataAnalyzer:

    @pytest.fixture
    def analyzer(self):
        return DataAnalyzer(csv_path=CSV_PATH, kappa=0.5)

    # --- catalog helpers ---

    def test_available_materials_non_empty(self, analyzer):
        """available_materials lists all CSV entries."""
        mats = analyzer.available_materials
        assert len(mats) > 0, "Catalog should not be empty"
        for expected in ["CdSe", "PbS", "CdS", "GaAs"]:
            assert expected in mats, f"{expected} must be in catalog"

    def test_validate_materials_all_valid(self, analyzer):
        """validate_materials returns [] when all materials exist."""
        invalid = analyzer.validate_materials(["CdSe", "PbS"])
        assert invalid == [], f"Expected no invalid materials, got {invalid}"

    def test_validate_materials_detects_unknown(self, analyzer):
        """validate_materials returns only the unknown entries."""
        invalid = analyzer.validate_materials(["CdSe", "UnknownXYZ", "PbS", "AlsoUnknown"])
        assert set(invalid) == {"UnknownXYZ", "AlsoUnknown"}

    # --- _build_wavelengths ---

    def test_build_wavelengths_shape_and_dtype(self, analyzer):
        """Wavelength array matches the requested range and is float32."""
        req = _make_request(wavelength_left_bound=280.0, wavelength_right_bound=860.0, wavelength_step=20.0)
        wl = analyzer._build_wavelengths(req)
        assert wl.dtype == np.float32
        assert wl[0] == pytest.approx(280.0)
        assert wl[-1] < 860.0
        expected_len = len(np.arange(280, 860, 20, dtype=np.float32))
        assert len(wl) == expected_len

    def test_build_wavelengths_csv_raises(self, analyzer):
        """NotImplementedError raised when wavelength_input_csv=True."""
        req = _make_request(
            wavelength_input_csv=True,
            wavelength_left_bound=None,
            wavelength_right_bound=None,
            wavelength_step=None,
        )
        with pytest.raises(NotImplementedError):
            analyzer._build_wavelengths(req)

    # --- _detect_convergence ---

    def test_detect_convergence_flat_history(self, analyzer):
        """Detects plateau at generation 2 (1-indexed) when fitness is constant."""
        history = [0.5, 0.5, 0.5, 0.5, 0.5]
        gen = analyzer._detect_convergence(history)
        assert gen == 2, f"Expected convergence at gen 2, got {gen}"

    def test_detect_convergence_strictly_improving(self, analyzer):
        """Returns total length when fitness strictly improves every generation."""
        history = [0.1, 0.2, 0.3, 0.4, 0.5]
        gen = analyzer._detect_convergence(history, tol=1e-4)
        assert gen == len(history)

    def test_detect_convergence_single_element(self, analyzer):
        """Single-element history returns 1 without error."""
        assert analyzer._detect_convergence([0.3]) == 1

    # --- analyze: full pipeline ---

    def test_analyze_single_layer_returns_complete_dict(self, analyzer):
        """analyze() returns a dict with all required keys for a single-layer run."""
        req = _make_request(materials=["CdSe"], population_size=8, max_iterations=2)
        result = analyzer.analyze(req)

        required_keys = [
            "optimal_radii_nm", "projected_pce", "fitness_history", "pce_history",
            "avg_fitness_history", "materials", "bandgaps_eV", "wavelengths_nm",
            "absorption_spectrum", "current_mismatch_index", "photon_harvesting_efficiency",
            "generations_to_convergence",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_analyze_radii_within_physical_bounds(self, analyzer):
        """Optimal radii must stay in [2, 10] nm."""
        req = _make_request(materials=["CdSe", "PbS"], population_size=8, max_iterations=2)
        result = analyzer.analyze(req)
        for r in result["optimal_radii_nm"]:
            assert 2.0 <= r <= 10.0, f"Radius {r} out of [2, 10] nm bounds"

    def test_analyze_history_lengths_match_iterations(self, analyzer):
        """fitness_history and pce_history lengths equal max_iterations."""
        req = _make_request(materials=["CdSe"], population_size=8, max_iterations=4)
        result = analyzer.analyze(req)
        assert len(result["fitness_history"]) == 4
        assert len(result["pce_history"]) == 4
        assert len(result["avg_fitness_history"]) == 4

    def test_analyze_pce_always_gte_fitness(self, analyzer):
        """Raw PCE must be >= penalized fitness every generation."""
        req = _make_request(materials=["CdSe", "PbS"], population_size=8, max_iterations=3)
        result = analyzer.analyze(req)
        for pce, fit in zip(result["pce_history"], result["fitness_history"]):
            assert pce >= fit - 1e-5, f"PCE {pce:.6f} < fitness {fit:.6f}"

    def test_analyze_single_layer_cmi_is_zero(self, analyzer):
        """Single-layer cells have no current mismatch → CMI == 0."""
        req = _make_request(materials=["CdSe"], population_size=8, max_iterations=2)
        result = analyzer.analyze(req)
        assert result["current_mismatch_index"] == pytest.approx(0.0, abs=1e-6)

    def test_analyze_tandem_cmi_non_negative(self, analyzer):
        """CMI is always non-negative for tandem cells."""
        req = _make_request(materials=["CdSe", "PbS"], population_size=8, max_iterations=2)
        result = analyzer.analyze(req)
        assert result["current_mismatch_index"] >= 0.0

    def test_analyze_phe_in_unit_interval(self, analyzer):
        """Photon harvesting efficiency per layer must be in [0, 1]."""
        req = _make_request(materials=["CdSe", "PbS"], population_size=8, max_iterations=2)
        result = analyzer.analyze(req)
        for phe in result["photon_harvesting_efficiency"]:
            assert 0.0 <= phe <= 1.0, f"PHE {phe} out of [0, 1]"

    def test_analyze_bandgaps_positive(self, analyzer):
        """All reported bandgaps must be positive eV values."""
        req = _make_request(materials=["CdSe", "PbS"], population_size=8, max_iterations=2)
        result = analyzer.analyze(req)
        for bg in result["bandgaps_eV"]:
            assert bg > 0, f"Bandgap {bg} should be positive"

    def test_analyze_output_shapes_match_materials(self, analyzer):
        """Per-layer fields must have length equal to number of materials."""
        materials = ["CdS", "CdSe", "PbS"]
        req = _make_request(materials=materials, population_size=8, max_iterations=2)
        result = analyzer.analyze(req)
        n = len(materials)
        assert len(result["optimal_radii_nm"]) == n
        assert len(result["bandgaps_eV"]) == n
        assert len(result["photon_harvesting_efficiency"]) == n
        assert len(result["absorption_spectrum"]) == n

    def test_analyze_convergence_within_range(self, analyzer):
        """generations_to_convergence must be between 1 and max_iterations."""
        req = _make_request(materials=["CdSe"], population_size=8, max_iterations=5)
        result = analyzer.analyze(req)
        gc = result["generations_to_convergence"]
        assert 1 <= gc <= 5, f"generations_to_convergence={gc} out of [1, 5]"

    def test_analyze_wavelengths_match_request(self, analyzer):
        """wavelengths_nm in output matches the requested grid."""
        req = _make_request(
            wavelength_left_bound=300.0,
            wavelength_right_bound=700.0,
            wavelength_step=10.0,
        )
        result = analyzer.analyze(req)
        expected = np.arange(300, 700, 10, dtype=np.float32).tolist()
        assert len(result["wavelengths_nm"]) == len(expected)
        assert result["wavelengths_nm"][0] == pytest.approx(300.0)
        assert result["wavelengths_nm"][-1] < 700.0
