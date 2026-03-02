import mindspore_config 
import pandas as pd
import mindspore as ms
from physics.BrusEngine import BrusEngine
from physics.SolarPerformanceEvaluator import SolarPerformanceEvaluator
from logic.GeneticSolarOptimizer import GeneticSolarOptimizer

class SolarOptimizationManager:
    """
    SolarOptimizationManager: Orchestrator class that bridges raw material data 
    with the MindSpore-based genetic optimization engine.
    
    Architecture: Controller / Factory Pattern
    Responsibility: Material data management, engine instantiation, and study execution.
    """

    def __init__(self, csv_path, kappa):
        """
        Initializes the material database and prepares a local cache for physics engines.
        
        Args:
            csv_path (str): Path to the CSV containing semiconductor physical constants.
            kappa (float): Penalty coefficient for current mismatch in the fitness function.
        """
        # Load physical constants and index by material name for O(1) lookup
        df = pd.read_csv(csv_path)
        self.catalog = df.set_index('Material').to_dict('index')
        
        # Cache to store instantiated BrusEngine objects (prevents redundant allocations)
        self.engines_cache = {}
        # Evaluator instance used for post-study metrics (PHE). kappa is irrelevant here.
        self._evaluator = SolarPerformanceEvaluator(kappa=kappa)

    def _compute_photon_harvesting_efficiency(self, absorption_tensor, wavelengths_tensor) -> list:
        """
        Fraction of incident photons absorbed by each layer at the champion configuration.

        Computed here while absorption_tensor is still on-device, avoiding an extra
        Python-level numpy roundtrip from the caller.

        Args:
            absorption_tensor: MindSpore Tensor, shape (Layers, Wavelengths).
            wavelengths_tensor: MindSpore Tensor, the wavelength grid [nm].

        Returns:
            List[float]: PHE values in [0, 1], one per layer, rounded to 6 d.p.
        """
        photon_flux = self._evaluator._interpolate_spectrum(wavelengths_tensor)
        delta = wavelengths_tensor[1] - wavelengths_tensor[0]
        total_flux = (photon_flux * delta).sum()

        return [
            round(float(((absorption_tensor[i] * self._evaluator.thickness * photon_flux * delta).sum() / total_flux).asnumpy()), 6)
            for i in range(absorption_tensor.shape[0])
        ]

    def _compute_bandgaps(
        self,
        materials: list,
        winner_radii,          # ms.Tensor shape (num_layers,) – champion from GA
        temp_tensor,           # ms.Tensor float32 – operating temperature
        wavelengths_tensor,    # ms.Tensor float32 – wavelength grid
    ) -> list:
        """
        Extracts the temperature- and confinement-corrected bandgap for each layer
        using the already-cached BrusEngine instances.

        Returns:
            List[float]: Bandgap values [eV], one per layer, rounded to 6 d.p.
        """
        bandgaps = []
        for i, name in enumerate(materials):
            engine = self.get_engine(name)
            radius = winner_radii[i:i+1].unsqueeze(0)
            _, e_qd = engine(
                temperature=temp_tensor,
                radius=radius,
                wavelengths=wavelengths_tensor,
            )
            bandgaps.append(round(float(e_qd.asnumpy().flat[0]), 6))
        return bandgaps

    def get_engine(self, name):
        """
        Retrieves or instantiates a BrusEngine for a specific material.
        
        Args:
            name (str): The material identifier (e.g., 'PbS', 'CdSe').
            
        Returns:
            BrusEngine: The physics-informed MindSpore cell for the requested material.
        """
        if name not in self.engines_cache:
            props = self.catalog[name]
            self.engines_cache[name] = BrusEngine(
                bandgap=props['Eg_0K_eV'], 
                alpha=props['Alpha_evK'],
                beta=props['Beta_K'], 
                me_eff=props['me_eff'],
                mh_eff=props['mh_eff'], 
                eps_r=props['epsilon_r']
            )
        return self.engines_cache[name]

    def run_study(self, user_params):
        """
        Executes a complete evolutionary optimization study based on user-defined parameters.
        
        Args:
            user_params (dict): Configuration dictionary with the following keys:
                - 'materials'(list[str]): Material names from the CSV catalog. e.g. ['CdSe', 'PbS', 'CdS', 'GaAs']
                - 'pop_size'(int): Population size for the genetic algorithm.
                - 'alpha'(float): Arithmetic crossover weight [0.0 - 1.0].
                - 'mutation'(float): Gaussian mutation strength [0.0 - 1.0].
                - 'iterations'(int): Number of generations to evolve.
                - 'temp'(ms.Tensor float32) : Temperature in Kelvin. e.g. ms.Tensor([300.0], ms.float32)
                - 'wavelength' (ms.Tensor float32) : Wavelength range in nm (AM1.5G: 280-2000).e.g. ms.Tensor(np.arange(280, 2000, 20), ms.float32)
                                
        Returns:
            tuple: (fitness_history, pce_history, avg_fitness_history, optimal_radii_vector, absorption_spectrum, bandgaps_eV, champion_cmi, phe)
        """
        selected_engines = [self.get_engine(m) for m in user_params['materials']]

        optimizer = GeneticSolarOptimizer(
            engines=selected_engines,
            population_size=user_params['pop_size'],
            num_layers=len(user_params['materials']),
            alpha=user_params['alpha'],
            mutation_strength=user_params['mutation']
        )

        fitness_results = []
        pce_results = []
        avg_fitness_results = []
        for i in range(user_params['iterations']):
            fitness, pce, cmi, winner, absorption_tensor, avg_fitness = optimizer(user_params['temp'], user_params['wavelength']) # type: ignore
            
            fitness_results.append(float(fitness.asnumpy())) # type: ignore
            pce_results.append(float(pce.asnumpy())) # type: ignore
            avg_fitness_results.append(round(float(avg_fitness.asnumpy()), 6)) # type: ignore
        
        absorption_spectrum = absorption_tensor.asnumpy().tolist() # type: ignore
        champion_cmi = round(float(cmi.asnumpy()), 6) # type: ignore
        phe = self._compute_photon_harvesting_efficiency(absorption_tensor, user_params['wavelength'])

        # Bandgap extraction: engines are already compiled and cached from the loop above.
        # Computing here avoids a redundant second pass from the caller.
        bandgaps_eV = self._compute_bandgaps(
            user_params['materials'], winner, user_params['temp'], user_params['wavelength']
        )

        return fitness_results, pce_results, avg_fitness_results, winner, absorption_spectrum, bandgaps_eV, champion_cmi, phe