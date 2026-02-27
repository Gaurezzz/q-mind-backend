import mindspore_config 
import pandas as pd
import mindspore as ms
from physics.BrusEngine import BrusEngine
from logic.GeneticSolarOptimizer import GeneticSolarOptimizer

class SolarOptimizationManager:
    """
    SolarOptimizationManager: Orchestrator class that bridges raw material data 
    with the MindSpore-based genetic optimization engine.
    
    Architecture: Controller / Factory Pattern
    Responsibility: Material data management, engine instantiation, and study execution.
    """

    def __init__(self, csv_path):
        """
        Initializes the material database and prepares a local cache for physics engines.
        
        Args:
            csv_path (str): Path to the CSV containing semiconductor physical constants.
        """
        # Load physical constants and index by material name for O(1) lookup
        df = pd.read_csv(csv_path)
        self.catalog = df.set_index('Material').to_dict('index')
        
        # Cache to store instantiated BrusEngine objects (prevents redundant allocations)
        self.engines_cache = {} 

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
            tuple: (fitness_history, optimal_radii_vector)
        """
        selected_engines = [self.get_engine(m) for m in user_params['materials']]

        optimizer = GeneticSolarOptimizer(
            engines=selected_engines,
            population_size=user_params['pop_size'],
            num_layers=len(user_params['materials']),
            alpha=user_params['alpha'],
            mutation_strength=user_params['mutation']
        )

        results = []
        for i in range(user_params['iterations']):
            fitness, winner, absorption_tensor = optimizer(user_params['temp'], user_params['wavelength']) # type: ignore
            
            results.append(float(fitness.asnumpy())) # type: ignore
        
        absorption_spectrum = absorption_tensor.asnumpy().tolist() # type: ignore
        return results, winner, absorption_spectrum