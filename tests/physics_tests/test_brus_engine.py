import pytest
import pandas as pd
import numpy as np
import mindspore as ms
from modelarts_worker.physics.BrusEngine import BrusEngine

def load_material_data():
    """Load material properties from CSV file."""
    df = pd.read_csv('db/materials.csv')
    return df.to_dict(orient='records')

class TestBrusEngineMaterials:
    
    @pytest.mark.parametrize("material_props", load_material_data())
    def test_material_physics(self, material_props):
        """
        Validates that the engine correctly processes each material from the CSV.
        """
        name = material_props['Material']
        
        engine = BrusEngine(
            bandgap=material_props['Eg_0K_eV'],
            alpha=material_props['Alpha_evK'],
            beta=material_props['Beta_K'],
            me_eff=material_props['me_eff'],
            mh_eff=material_props['mh_eff'],
            eps_r=material_props['epsilon_r']
        )

        temp = ms.Tensor([300.0], ms.float32)
        radius = ms.Tensor([3.0], ms.float32) 
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)

        absorption, e_qd = engine(temp, radius, wavelengths)

        # Validate absorption tensor
        assert isinstance(absorption, ms.Tensor), f"Error in {name}: Absorption is not a Tensor"
        abs_np = absorption.asnumpy()
        assert abs_np.shape == (2800,), f"Error in {name}: Unexpected shape {abs_np.shape}"
        assert not np.isnan(abs_np).any(), f"Error in {name}: NaN values in absorption"
        assert not np.isinf(abs_np).any(), f"Error in {name}: Inf values in absorption"
        assert np.all(abs_np >= 0), f"Error in {name}: Negative absorption values"
        assert np.max(abs_np) > 0, f"Error in {name}: Absorption is zero everywhere"
        
        # Validate quantum dot bandgap
        assert isinstance(e_qd, ms.Tensor), f"Error in {name}: e_qd is not a Tensor"
        e_qd_val = e_qd.asnumpy().item()
        assert not np.isnan(e_qd_val), f"Error in {name}: e_qd is NaN"
        
        # Quantum confinement effect: e_qd can be larger or smaller than bulk depending on radius and material
        # Validate the change is within physically reasonable bounds
        bandgap_change = e_qd_val - material_props['Eg_0K_eV']
        assert abs(bandgap_change) < 5.0, f"Error in {name}: Bandgap change ({bandgap_change:.3f} eV) is too large"
        assert e_qd_val > 0.1, f"Error in {name}: QD bandgap ({e_qd_val:.3f} eV) is too small"
        assert e_qd_val < 10.0, f"Error in {name}: QD bandgap ({e_qd_val:.3f} eV) is unrealistically high"
        
        # Validate peak wavelength is in reasonable range
        peak_idx = np.argmax(abs_np)
        peak_wavelength = 200 + peak_idx
        expected_wavelength = 1239.84 / e_qd_val
        assert abs(peak_wavelength - expected_wavelength) < 20, f"Error in {name}: Peak at {peak_wavelength} nm, expected ~{expected_wavelength:.1f} nm"
        
        # Validate Gaussian-like absorption profile
        max_abs = np.max(abs_np)
        assert max_abs == abs_np[peak_idx], f"Error in {name}: Peak not at maximum"
        # Check absorption decays on both sides of peak
        if peak_idx > 100 and peak_idx < len(abs_np) - 100:
            left_avg = np.mean(abs_np[peak_idx-100:peak_idx-50])
            right_avg = np.mean(abs_np[peak_idx+50:peak_idx+100])
            assert left_avg < max_abs * 0.8, f"Error in {name}: Absorption doesn't decay properly on left side"
            assert right_avg < max_abs * 0.8, f"Error in {name}: Absorption doesn't decay properly on right side"

        print(f"{name}: e_qd={e_qd_val:.3f} eV, peak={peak_wavelength} nm, max_abs={np.max(abs_np):.2e}")

    def test_csv_structure(self):
        """Verifies that the CSV file has all required columns."""
        df = pd.read_csv('db/materials.csv')
        required_columns = ['Material', 'Eg_0K_eV', 'Alpha_evK', 'Beta_K', 'me_eff', 'mh_eff', 'epsilon_r']
        for col in required_columns:
            assert col in df.columns, f"Missing critical column: {col}"
    
    def test_size_dependence(self):
        """Validates that smaller QDs have larger bandgaps (quantum confinement)."""
        engine = BrusEngine(bandgap=1.5, alpha=0.0005, beta=200, me_eff=0.07, mh_eff=0.45, eps_r=10.0)
        
        temp = ms.Tensor([300.0], ms.float32)
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)
        
        radius_small = ms.Tensor([2.0], ms.float32)
        radius_large = ms.Tensor([5.0], ms.float32)
        
        _, e_qd_small = engine(temp, radius_small, wavelengths)
        _, e_qd_large = engine(temp, radius_large, wavelengths)
        
        e_small = e_qd_small.asnumpy().item()
        e_large = e_qd_large.asnumpy().item()
        
        assert e_small > e_large, f"Smaller QD should have larger bandgap: {e_small:.3f} eV vs {e_large:.3f} eV"
        print(f"Quantum confinement: E(2nm)={e_small:.3f} eV > E(5nm)={e_large:.3f} eV")
    
    def test_temperature_dependence(self):
        """Validates that bandgap decreases with temperature (Varshni's law)."""
        engine = BrusEngine(bandgap=1.5, alpha=0.0005, beta=200, me_eff=0.07, mh_eff=0.45, eps_r=10.0)
        
        radius = ms.Tensor([3.0], ms.float32)
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)
        
        temp_low = ms.Tensor([100.0], ms.float32)
        temp_high = ms.Tensor([500.0], ms.float32)
        
        _, e_qd_low = engine(temp_low, radius, wavelengths)
        _, e_qd_high = engine(temp_high, radius, wavelengths)
        
        e_low = e_qd_low.asnumpy().item()
        e_high = e_qd_high.asnumpy().item()
        
        assert e_low > e_high, f"Bandgap should decrease with temperature: E(100K)={e_low:.3f} eV vs E(500K)={e_high:.3f} eV"
        print(f"Varshni effect: E(100K)={e_low:.3f} eV > E(500K)={e_high:.3f} eV")
    
    def test_edge_cases(self):
        """Validates behavior at extreme radii."""
        engine = BrusEngine(bandgap=1.5, alpha=0.0005, beta=200, me_eff=0.07, mh_eff=0.45, eps_r=10.0)
        
        temp = ms.Tensor([300.0], ms.float32)
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)
        
        # Very small radius (strong confinement)
        radius_tiny = ms.Tensor([1.0], ms.float32)
        abs_tiny, e_qd_tiny = engine(temp, radius_tiny, wavelengths)
        e_tiny = e_qd_tiny.asnumpy().item()
        
        assert not np.isnan(abs_tiny.asnumpy()).any(), "NaN values at tiny radius"
        assert e_tiny > 1.5, f"Strong confinement should increase bandgap: {e_tiny:.3f} eV"
        
        # Very large radius (weak confinement, approaches bulk)
        radius_large = ms.Tensor([10.0], ms.float32)
        abs_large, e_qd_large = engine(temp, radius_large, wavelengths)
        e_large = e_qd_large.asnumpy().item()
        
        assert not np.isnan(abs_large.asnumpy()).any(), "NaN values at large radius"
        assert abs(e_large - 1.5) < 0.5, f"Large QD should approach bulk bandgap: {e_large:.3f} eV"
        
        print(f"Edge cases: E(1nm)={e_tiny:.3f} eV, E(10nm)={e_large:.3f} eV")