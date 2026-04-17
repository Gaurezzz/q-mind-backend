from modelarts_worker.mindspore_config import get_logger
from mindspore.nn import Cell
from mindspore import ops, Tensor, dtype
from modelarts_worker.physics.SolarPerformanceEvaluator import SolarPerformanceEvaluator
from modelarts_worker.physics.BrusEngine import BrusEngine
import mindspore as ms
from typing import List, Tuple

logger = get_logger(__name__)

class GeneticSolarOptimizer(Cell):
    """
    GeneticSolarOptimizer: A MindSpore-based evolutionary engine for QD solar cells.
    
    This class performs batch optimization of multi-layer quantum dot configurations
    using tournament selection, arithmetic crossover, and Gaussian mutation.
    """

    def __init__(self, population_size: int = 100, num_layers: int = 2, engines: list = [], alpha: float = 0.5, mutation_strength: float = 0.1, kappa: float = 0.5):
        super().__init__()
        # Physical bounds for QD radii [nm]
        self.r_min = Tensor(2.0, ms.float32)
        self.r_max = Tensor(10.0, ms.float32)
        
        # Performance evaluator (Efficiency + Current Mismatch Penalty)
        self.evaluator = SolarPerformanceEvaluator(kappa=kappa)
        self.engines = engines
        self.alpha = alpha
        self.mutation_strength = mutation_strength
        
        # Population Initialization: Random uniform distribution of radii
        # Shape: (population_size, num_layers)
        self.population = ms.Parameter(ops.uniform(
            (population_size, num_layers), 
            self.r_min, 
            self.r_max, 
            dtype=ms.float32
        ), name="population")

        logger.info(
            "GeneticSolarOptimizer ready | pop=%d  layers=%d  α=%.2f  "
            "mutation=%.3f  κ=%.3f  r=[%.1f, %.1f] nm",
            population_size, num_layers, alpha, mutation_strength, kappa,
            float(self.r_min.asnumpy()), float(self.r_max.asnumpy()),
        )

    def construct(self, temperature, wavelength) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        absorption_list = []
        e_qd_list = []

        # Spectral Response Calculation: Map each engine to its corresponding layer
        for i, engine in enumerate(self.engines):
            # Extract radii for the specific layer [Batch, 1]
            radius_layer = self.population[:, i:i+1]
            
            # Physical modeling of absorption and energy gaps
            abs_layer, e_qd_layer = engine(
                temperature = temperature,
                radius = radius_layer,
                wavelengths = wavelength
            )
            absorption_list.append(abs_layer)
            e_qd_list.append(e_qd_layer)

        # Vectorized assembly: Shape (Batch, Layers, Wavelengths)
        absortion_batch = ops.stack(absorption_list, axis=1)
        e_qd_batch = ops.stack(e_qd_list, axis=1).squeeze(axis=-1) 

        # Fitness Evaluation: PCE - kappa * Sum(|Ji - Ji+1|)
        fitness_batch, efficiency_batch, cmi_batch = self.evaluator(
            absorption_coefficient = absortion_batch,
            e_qd = e_qd_batch,
            wavelengths = wavelength
        )
            
        # Elitism: Identify the best performing individual (Champion)
        winner = fitness_batch.argmax()
        winner_radii = self.population[winner,:]

        # Tournament Selection: Pairwise competition
        # Formula: Select Parent if Fitness(A) > Fitness(B)
        candidates = ops.randint(low=0, high=self.population.shape[0], size=(self.population.shape[0], 2), dtype=ms.int32)
        selected_parents_indices = ops.where(
            fitness_batch[candidates[:,0]] > fitness_batch[candidates[:,1]], 
            candidates[:,0], 
            candidates[:,1]
        )
        parents_radii = self.population[selected_parents_indices]

        # Arithmetic Crossover: Weighted average of parent genes
        # Formula: Offspring = alpha * P1 + (1 - alpha) * P2
        pop_size = self.population.shape[0]
        half_pop = pop_size // 2
        
        p1 = parents_radii[:half_pop, :]
        p2 = parents_radii[half_pop:half_pop*2, :]
        offspring = self.alpha * p1 + (1 - self.alpha) * p2 
        
        # Gaussian Mutation: Stochastic search with boundary clipping
        # Formula: R_new = Clip(R_old + N(0, sigma), r_min, r_max)
        offspring = offspring + ops.standard_normal(p1.shape) * self.mutation_strength
        offspring = ops.clip_by_value(offspring, self.r_min, self.r_max)

        # Generational Update: Combine Elite + Offspring + Survived Parents
        survivors_count = pop_size - half_pop - 1  # Remaining slots after offspring and elite
        new_population = ops.concat((offspring, winner_radii.expand_dims(axis=0), parents_radii[:survivors_count]))
        ops.assign(self.population, new_population)

        avg_fitness = fitness_batch.mean()
        return fitness_batch[winner], efficiency_batch[winner], cmi_batch[winner], winner_radii, absortion_batch[winner], avg_fitness  #type: ignore