import numpy as np
import mindspore as ms
from typing import Any, Dict, List

from modelarts_worker.logic.SolarOptimizationManager import SolarOptimizationManager


class DataAnalyzer:
    """
    DataAnalyzer: Bridge between the optimization API and the SolarOptimizationManager.

    Responsibilities
    ----------------
    * Expose catalog helpers (available materials, validation).
    * Translate a raw parameters dictionary into the internal ``user_params`` dict consumed by :class:`SolarOptimizationManager`.
    * Orchestrate the optimization run.
    * Post-process raw tensors into the derived metrics.
    """

    def __init__(self, csv_path: str, kappa: float):
        """
        Args:
            csv_path (str): Path to the semiconductor material CSV catalog.
            kappa (float): Penalty coefficient for current mismatch in the fitness function.
        """
        self.manager = SolarOptimizationManager(csv_path=csv_path, kappa=kappa)

    @property
    def available_materials(self) -> List[str]:
        """All material keys present in the CSV catalog."""
        return list(self.manager.catalog.keys())

    def validate_materials(self, materials: List[str]) -> List[str]:
        """Returns the subset of materials that are not in the catalog."""
        return [m for m in materials if m not in self.manager.catalog]

    def _build_wavelengths(self, request) -> np.ndarray:
        """
        Converts the wavelength specification in the request to a float32 numpy array.
        Accepts an OptimizationRequest (Pydantic model) or a plain dict.
        """
        if isinstance(request, dict):
            csv_mode = request.get("wavelength_input_csv", False)
            left     = request.get("wavelength_left_bound", 280.0)
            right    = request.get("wavelength_right_bound", 2500.0)
            step     = request.get("wavelength_step", 10.0)
        else:
            csv_mode = request.wavelength_input_csv
            left     = request.wavelength_left_bound
            right    = request.wavelength_right_bound
            step     = request.wavelength_step
        if not csv_mode:
            return np.arange(left, right, step, dtype=np.float32)
        raise NotImplementedError(
            "CSV wavelength input is not yet supported. "
            "Please use the automatic range mode."
        )

    def _detect_convergence(
        self, fitness_history: List[float], tol: float = 1e-4
    ) -> int:
        """
        Returns the first generation (1-indexed) after which the best fitness
        no longer improves by more than tol. Returns the total number of
        generations when no plateau is detected.
        """
        for i in range(1, len(fitness_history)):
            if all(abs(fitness_history[j] - fitness_history[-1]) <= tol for j in range(i, len(fitness_history))):
                return i + 1
        return len(fitness_history)

    def analyze(self, request) -> Dict[str, Any]:
        """
        Args:
            request: Validated incoming request — either an OptimizationRequest Pydantic
                model or a plain dict with equivalent keys.

        Returns:
            dict: All fields required for the final response except
            status and computation_time_ms.
        """
        if isinstance(request, dict):
            get = lambda key, default=None: request.get(key, default)
        else:
            get = lambda key, default=None: getattr(request, key, default)

        wavelengths_np = self._build_wavelengths(request)
        temp_tensor = ms.Tensor([get("operating_temperature", 298.15)], ms.float32)
        wl_tensor = ms.Tensor(wavelengths_np, ms.float32)

        materials = get("materials", [])

        user_params = {
            "materials":  materials,
            "pop_size":   get("population_size", 100),
            "alpha":      get("crossover_alpha", 0.5),
            "mutation":   get("mutation_strength", 0.1),
            "iterations": get("max_iterations", 50),
            "kappa":      get("kappa", 0.5),
            "temp":       temp_tensor,
            "wavelength": wl_tensor,
        }

        fitness_history, pce_history, avg_fitness_history, best_radii, absorption_spectrum, bandgaps, cmi, phe = (
            self.manager.run_study(user_params)  
        )

        optimal_radii: List[float] = best_radii.asnumpy().tolist()  # type: ignore[union-attr]
        projected_pce: float = pce_history[-1] if pce_history else 0.0
        gen_conv = self._detect_convergence(fitness_history)

        return {
            "optimal_radii_nm":             optimal_radii,
            "projected_pce":                projected_pce,
            "fitness_history":              fitness_history,
            "pce_history":                  pce_history,
            "avg_fitness_history":          avg_fitness_history,
            "materials":                    materials,
            "bandgaps_eV":                  bandgaps,
            "wavelengths_nm":               wavelengths_np.tolist(),
            "absorption_spectrum":          absorption_spectrum,
            "current_mismatch_index":       cmi,
            "photon_harvesting_efficiency": phe,
            "generations_to_convergence":   gen_conv,
        }