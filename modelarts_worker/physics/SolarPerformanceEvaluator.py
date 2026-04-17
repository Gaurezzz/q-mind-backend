import os
import pandas as pd
from mindspore import Tensor, dtype as mstype
from mindspore.nn import Cell
from pvlib import spectrum
from mindspore import ops
from modelarts_worker.mindspore_config import get_logger

logger = get_logger(__name__)

class SolarPerformanceEvaluator(Cell):
    """
    Evaluates the photovoltaic performance of Quantum Dot Solar Cells (QDSCs).
    
    This engine calculates the Short-Circuit Current (J_sc), Open-Circuit Voltage (V_oc),
    and Power Conversion Efficiency (PCE) for a batch of solar cell designs.
    It implements a multi-objective fitness function that maximizes efficiency while
    penalizing current mismatch in tandem structures.
    """

    def __init__(self, kappa: float = 0.5):
        """
        Initializes the physical constants and loads the solar spectrum.

        Args:
            kappa (float): Penalty coefficient for current mismatch in the fitness function. Controls how strictly the algorithm enforces current matching.
        """
        super().__init__()

        # --- Physical Constants & Control Parameters ---
        self.ENERGY_CONST = 1.9864e-16      # hc in J*m
        self.ELECTRON_CHARGE = 1.60218e-19  # q in Coulombs
        self.FF = 0.75                      # Fill Factor (estimated standard)
        self.thickness = 300e-9             # Active layer thickness in meters (300 nm)
        self.kappa = kappa                  # Optimization penalty weight

        # --- Load AM1.5G Standard Spectrum ---
        # We load the global irradiance data to simulate real-world conditions
        am15 = spectrum.get_reference_spectra()
        am15['global'] = am15['global'].astype('float32')

        self.global_irradiance = Tensor(am15['global'].values, mstype.float32)
        self.wavelengths = Tensor(am15.index.values, mstype.float32)
        self.delta = self.wavelengths[1] - self.wavelengths[0]

        # --- Pre-calculate Photon Flux ---
        # Convert Irradiance (W/m^2/nm) to Photon Flux (photons/s/m^2/nm)
        # Formula: Phi = Irradiance / (hc / lambda)
        energy = self.ENERGY_CONST / self.wavelengths
        self.photon_flux = self.global_irradiance / energy

        # Calculate total incident solar power (P_sun) ~ 100 mW/cm^2
        self.p_sun = (self.global_irradiance * self.delta).sum()

        logger.debug(
            "SolarPerformanceEvaluator ready | kappa=%.3f  thickness=%.0f nm  "
            "AM1.5G points=%d  P_sun=%.2f W/m²",
            kappa, self.thickness * 1e9, len(self.wavelengths), float(self.p_sun.asnumpy()),
        )

    def _interpolate_spectrum(self, target_wavelengths: Tensor) -> Tensor:
        """
        Interpolates the AM1.5G solar photon flux to target wavelengths.

        Implemented with pure MindSpore ops so it is safe to call from inside
        a Cell.construct() graph without triggering numpy fallback / compilation hang.

        Args:
            target_wavelengths (Tensor): Desired wavelength grid [nm]. Shape: (M,)

        Returns:
            Tensor: Interpolated photon flux [photons/s/m²/nm]. Shape: (M,)
        """
        target = target_wavelengths.unsqueeze(1)          # (M, 1)
        ref    = self.wavelengths.unsqueeze(0)             # (1, N)

        # Number of reference points strictly less than each target value
        mask     = (target > ref).astype(mstype.float32)  # (M, N)
        idx_left = ops.cast(mask.sum(axis=1), mstype.int32) - 1  # (M,)

        # Clamp so we never index out of bounds
        n_ref    = self.wavelengths.shape[0]
        idx_left = ops.clip_by_value(
            idx_left,
            Tensor(0,         mstype.int32),
            Tensor(n_ref - 2, mstype.int32),
        )
        idx_right = idx_left + 1

        wl_left  = self.wavelengths[idx_left]          # (M,)
        wl_right = self.wavelengths[idx_right]         # (M,)
        ir_left  = self.global_irradiance[idx_left]    # (M,)
        ir_right = self.global_irradiance[idx_right]   # (M,)

        # Linear interpolation weight
        t = (target_wavelengths - wl_left) / (wl_right - wl_left + 1e-10)
        interpolated_irradiance = ir_left + t * (ir_right - ir_left)

        # Convert irradiance [W/m²/nm] → photon flux [photons/s/m²/nm]
        energy = self.ENERGY_CONST / target_wavelengths
        return interpolated_irradiance / energy

    def construct(self, absorption_coefficient: Tensor, e_qd: Tensor, wavelengths: Tensor) -> Tensor:
        """
        Computes the fitness score for a batch of solar cell individuals simultaneously.

        Args:
            absorption_coefficient (Tensor): Shape (Batch, Layers, Wavelengths). The spectral absorption profile for each layer.
            e_qd (Tensor): Shape (Batch, Layers). Quantum Dot Bandgaps in eV for each layer.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - fitness     Shape (Batch,): Penalized score driving the GA (PCE - κ·CMI).
                - efficiency  Shape (Batch,): Raw PCE without penalty, for reporting.
                - cmi_batch   Shape (Batch,): Normalized current mismatch index per individual.
        """
        
        # Calculate Short-Circuit Current (J_sc) for every layer using Beer-Lambert law.
        # In a tandem stack, each layer only receives photons NOT absorbed by the layers above it:
        #   Φ_i(λ) = Φ_0(λ) × exp(−Σ_{k<i} α_k(λ) × d)
        # Result shape: (Batch, Layers)
        photon_flux_interp = self._interpolate_spectrum(wavelengths)   # (W,)
        delta = wavelengths[1] - wavelengths[0]

        # Optical depth per layer: τ_k(λ) = α_k(λ) × d  — shape (Batch, Layers, W)
        optical_depth = absorption_coefficient * self.thickness

        # Exclusive cumulative optical depth reaching each layer (sum of layers above it).
        # inclusive_tau[i] = Σ_{k<=i} τ_k  →  exclusive_tau[i] = inclusive_tau[i] - τ_i = Σ_{k<i} τ_k
        # This avoids concat/slicing and is safe in graph mode.
        inclusive_tau = ops.cumsum(optical_depth, 1)           # (Batch, Layers, W) inclusive
        exclusive_tau = inclusive_tau - optical_depth          # (Batch, Layers, W) exclusive

        # Attenuated photon flux arriving at each layer — reshape flux for explicit broadcasting
        photon_flux_3d = photon_flux_interp.reshape(1, 1, -1)              # (1, 1, W)
        photon_flux_per_layer = photon_flux_3d * ops.exp(-exclusive_tau)   # (Batch, Layers, W)

        # Exact Beer-Lambert absorbed fraction in each layer: 1 − exp(−τ)
        absorbed_fraction = 1.0 - ops.exp(-optical_depth)                  # (Batch, Layers, W)

        # J_sc per layer [A/m²]: integrate absorbed photons over wavelengths
        j_layers = self.ELECTRON_CHARGE * (photon_flux_per_layer * absorbed_fraction * delta).sum(axis=-1)

        # Calculate Open-Circuit Voltage (V_oc)
        # Estimation: V_oc approx (E_g / q) - 0.4V loss.
        # For tandem cells in series, voltages add up. We sum across layers (axis=1).
        # FIX: Clamp to 0 — if E_qd < 0.4 eV V_oc would be negative and corrupt efficiency.
        v_layers = ops.maximum(e_qd - 0.4, ops.zeros_like(e_qd))
        if len(e_qd.shape) > 1 and e_qd.shape[1] > 1:
            v_oc_total = v_layers.sum(axis=1)
        else:
            v_oc_total = v_layers[:, 0]

        # Apply Current Matching Condition
        # In a series connection, the total current is limited by the layer generating the least current.
        # We take the minimum across layers (axis=1).
        # NOTE: Tensor.min(axis=N) returns a (values, indices) tuple in MindSpore;
        #       use ops.reduce_min to get values directly.
        if len(j_layers.shape) > 1 and j_layers.shape[1] > 1:
            j_sc_limit = ops.reduce_min(j_layers, 1)
        else:
            j_sc_limit = j_layers[:, 0]

        # Calculate Power Conversion Efficiency (PCE)
        # Eta = (J_sc * V_oc * FF) / P_in
        efficiency = (j_sc_limit * v_oc_total * self.FF) / self.p_sun

        # Calculate Current Mismatch Penalty
        # We penalize designs where layers generate vastly different currents.
        # The penalty is proportional to the sum of absolute differences from the minimum current, normalized by the minimum current.
        

        # keepdims=True is not supported by Tensor.min() in MindSpore;
        # use ops.reduce_min and unsqueeze manually.
        j_min_flat = ops.reduce_min(j_layers, 1)          # (Batch,)
        j_min = j_min_flat.unsqueeze(1)                    # (Batch, 1)  broadcast-ready

        if len(j_layers.shape) > 1 and j_layers.shape[1] > 1:
            diff_j = ops.abs(j_layers[:, 1:] - j_min).sum(axis=1)
        else:
            diff_j = ops.zeros(j_layers.shape[0], mstype.float32)

        # CMI: pure normalized mismatch — reported to the client as-is (no kappa).
        
        j_max_flat = ops.reduce_max(j_layers, 1)                  # (Batch,)
        cmi_batch = diff_j / (j_max_flat + 1e-10)  # Avoid division by zero

        # Final Fitness Calculation
        # kappa arrives as a decimal fraction from the frontend (0.0–1.0).
        # Fitness = Efficiency - kappa * CMI
        fitness = efficiency - self.kappa * cmi_batch

        # Return penalized fitness (drives GA), raw PCE, and raw CMI (for reporting)
        return fitness, efficiency, cmi_batch