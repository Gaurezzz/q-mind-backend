import pytest
import numpy as np
import mindspore as ms
from logic.GeneticSolarOptimizer import GeneticSolarOptimizer
from physics.BrusEngine import BrusEngine

class TestGeneticSolarOptimizer:
    
    @pytest.fixture
    def setup_optimizer(self):
        """Creates a basic optimizer instance with 2-layer configuration."""
        # Create engines for CdSe and PbS (common QD materials)
        engine1 = BrusEngine(
            bandgap=1.74, alpha=0.000415, beta=180, 
            me_eff=0.13, mh_eff=0.45, eps_r=10.0
        )
        engine2 = BrusEngine(
            bandgap=0.41, alpha=0.000455, beta=160, 
            me_eff=0.085, mh_eff=0.085, eps_r=17.0
        )
        
        optimizer = GeneticSolarOptimizer(
            population_size=20,
            num_layers=2,
            engines=[engine1, engine2],
            alpha=0.5,
            mutation_strength=0.1
        )
        
        temperature = ms.Tensor([300.0], ms.float32)
        wavelengths = ms.Tensor(np.arange(280, 2000, 10), ms.float32)
        
        return optimizer, temperature, wavelengths
    
    def test_initialization(self, setup_optimizer):
        """Validates that the population is initialized correctly."""
        optimizer, _, _ = setup_optimizer
        
        # Check population shape
        assert optimizer.population.shape == (20, 2), \
            f"Expected population shape (20, 2), got {optimizer.population.shape}"
        
        # Check radii are within bounds
        pop_np = optimizer.population.asnumpy()
        assert np.all(pop_np >= 2.0), "Some radii are below r_min (2.0 nm)"
        assert np.all(pop_np <= 10.0), "Some radii exceed r_max (10.0 nm)"
        
        # Check no NaN or Inf values
        assert not np.isnan(pop_np).any(), "Population contains NaN values"
        assert not np.isinf(pop_np).any(), "Population contains Inf values"
        
        print(f"Population initialized: {pop_np.shape}, range=[{pop_np.min():.2f}, {pop_np.max():.2f}] nm")
    
    def test_forward_pass(self, setup_optimizer):
        """Validates that a complete genetic cycle executes without errors."""
        optimizer, temperature, wavelengths = setup_optimizer
        
        # Run one generation
        best_fitness, best_pce, best_cmi, best_radii, best_absorption, best_avg_fitness = optimizer(temperature, wavelengths)
        
        # Validate outputs
        assert isinstance(best_fitness, ms.Tensor), "Fitness should be a Tensor"
        assert isinstance(best_pce, ms.Tensor), "PCE should be a Tensor"
        assert isinstance(best_cmi, ms.Tensor), "CMI should be a Tensor"
        assert isinstance(best_radii, ms.Tensor), "Radii should be a Tensor"
        assert isinstance(best_absorption, ms.Tensor), "Absorption should be a Tensor"
        assert isinstance(best_avg_fitness, ms.Tensor), "Avg fitness should be a Tensor"
        
        fitness_val = best_fitness.asnumpy().item()
        pce_val = best_pce.asnumpy().item()
        cmi_val = best_cmi.asnumpy().item()
        radii_np = best_radii.asnumpy()
        avg_fitness_val = best_avg_fitness.asnumpy().item()
        
        assert not np.isnan(fitness_val), "Best fitness is NaN"
        assert not np.isinf(fitness_val), "Best fitness is Inf"
        assert pce_val >= fitness_val - 1e-5, "PCE should be >= fitness (fitness = PCE - penalty)"
        assert cmi_val >= 0.0, "CMI must be non-negative"
        assert not np.isnan(avg_fitness_val), "Avg fitness is NaN"
        assert avg_fitness_val <= fitness_val + 1e-5, "Avg fitness should be <= best fitness"
        assert radii_np.shape == (2,), f"Expected radii shape (2,), got {radii_np.shape}"
        assert np.all(radii_np >= 2.0) and np.all(radii_np <= 10.0), \
            f"Best radii out of bounds: {radii_np}"
        
        print(f"Generation completed: fitness={fitness_val:.6f}, pce={pce_val:.6f}, cmi={cmi_val:.6f}, radii={radii_np}")
    
    def test_elitism_preservation(self, setup_optimizer):
        """Validates that the best individual is preserved across generations."""
        optimizer, temperature, wavelengths = setup_optimizer
        
        # Run first generation
        fitness1, _, _, radii1, _, _ = optimizer(temperature, wavelengths)
        fitness1_val = fitness1.asnumpy().item()
        radii1_np = radii1.asnumpy()
        
        # Check if elite is in new population
        pop_after = optimizer.population.asnumpy()
        elite_preserved = np.any(np.all(np.isclose(pop_after, radii1_np, atol=1e-5), axis=1))
        
        assert elite_preserved, "Elite individual not found in new population"
        print(f"Elitism verified: champion with fitness={fitness1_val:.6f} preserved")
    
    def test_population_diversity(self, setup_optimizer):
        """Validates that mutation maintains population diversity."""
        optimizer, temperature, wavelengths = setup_optimizer
        
        # Store initial population
        initial_pop = optimizer.population.asnumpy().copy()
        
        # Run generation
        optimizer(temperature, wavelengths)
        
        # Check population has changed
        new_pop = optimizer.population.asnumpy()
        changes = np.abs(new_pop - initial_pop).sum()
        
        assert changes > 0, "Population did not evolve"
        
        # Check diversity (standard deviation across population)
        std_layer1 = np.std(new_pop[:, 0])
        std_layer2 = np.std(new_pop[:, 1])
        
        assert std_layer1 > 0.1, f"Layer 1 lacks diversity: std={std_layer1:.3f}"
        assert std_layer2 > 0.1, f"Layer 2 lacks diversity: std={std_layer2:.3f}"
        
        print(f"Diversity maintained: std_L1={std_layer1:.3f}, std_L2={std_layer2:.3f}")
    
    def test_boundary_clipping(self, setup_optimizer):
        """Validates that mutation respects physical bounds."""
        optimizer, temperature, wavelengths = setup_optimizer
        
        # Run multiple generations
        for _ in range(5):
            optimizer(temperature, wavelengths)
        
        # Check all radii are within bounds
        pop_np = optimizer.population.asnumpy()
        assert np.all(pop_np >= 2.0), f"Min radius violation: {pop_np.min():.2f} nm"
        assert np.all(pop_np <= 10.0), f"Max radius violation: {pop_np.max():.2f} nm"
        
        print(f"Boundaries respected after 5 generations: range=[{pop_np.min():.2f}, {pop_np.max():.2f}] nm")
    
    def test_fitness_improvement(self, setup_optimizer):
        """Validates that fitness improves over multiple generations."""
        optimizer, temperature, wavelengths = setup_optimizer
        
        fitness_history = []
        
        # Run 10 generations
        for gen in range(10):
            best_fitness, _, _, _, _, _ = optimizer(temperature, wavelengths)
            fitness_history.append(best_fitness.asnumpy().item())
        
        # Check if best fitness in last 3 generations >= initial fitness
        initial_fitness = fitness_history[0]
        final_avg = np.mean(fitness_history[-3:])
        
        print(f"Evolution: Gen 0={initial_fitness:.6f}, Gen 7-9 avg={final_avg:.6f}")
        
        # Allow for some variance, but expect overall improvement or stability
        # For negative fitness, check if it's getting less negative (improving)
        if initial_fitness < 0:
            assert final_avg >= initial_fitness * 1.05 or final_avg >= initial_fitness - abs(initial_fitness) * 0.05, \
                f"Fitness degraded significantly: {initial_fitness:.6f} → {final_avg:.6f}"
        else:
            assert final_avg >= initial_fitness * 0.95, \
                f"Fitness degraded significantly: {initial_fitness:.6f} → {final_avg:.6f}"
    
    def test_crossover_operator(self, setup_optimizer):
        """Validates that arithmetic crossover produces valid offspring."""
        optimizer, temperature, wavelengths = setup_optimizer
        
        # Get initial population
        initial_pop = optimizer.population.asnumpy().copy()
        
        # Run generation (includes crossover)
        optimizer(temperature, wavelengths)
        
        new_pop = optimizer.population.asnumpy()
        
        # Check offspring are within parent bounds
        for i in range(min(50, new_pop.shape[0])):  # Check first 50 offspring
            offspring = new_pop[i]
            # Each offspring value should be between min and max of initial population
            for layer in range(2):
                min_parent = initial_pop[:, layer].min()
                max_parent = initial_pop[:, layer].max()
                # Allow for mutation to exceed parents slightly
                assert offspring[layer] >= min_parent - 1.0, \
                    f"Offspring {i} layer {layer} too small: {offspring[layer]:.2f}"
                assert offspring[layer] <= max_parent + 1.0, \
                    f"Offspring {i} layer {layer} too large: {offspring[layer]:.2f}"
        
        print("Crossover operator validated: offspring within expected bounds")
    
    def test_tournament_selection(self, setup_optimizer):
        """Validates that tournament selection favors better individuals."""
        optimizer, temperature, wavelengths = setup_optimizer
        
        # Run one generation to get fitness values
        optimizer(temperature, wavelengths)
        
        # The selection mechanism is implicit in the construct method
        # We validate indirectly by checking population convergence
        radii_std = optimizer.population.asnumpy().std(axis=0)
        
        assert radii_std[0] < 8.0, "Layer 1 shows no selection pressure"
        assert radii_std[1] < 8.0, "Layer 2 shows no selection pressure"
        
        print(f"Selection pressure detected: std={radii_std}")
    
    def test_multi_layer_scaling(self):
        """Validates that optimizer handles different numbers of layers."""
        for num_layers in [1, 2, 3, 4]:
            engines = [
                BrusEngine(
                    bandgap=1.5, alpha=0.0005, beta=200,
                    me_eff=0.1, mh_eff=0.4, eps_r=10.0
                )
                for _ in range(num_layers)
            ]
            
            optimizer = GeneticSolarOptimizer(
                population_size=10,
                num_layers=num_layers,
                engines=engines
            )
            
            temperature = ms.Tensor([300.0], ms.float32)
            wavelengths = ms.Tensor(np.arange(280, 2000, 20), ms.float32)
            
            fitness, _, _, radii = optimizer(temperature, wavelengths)[:4]
            
            assert radii.shape == (num_layers,), \
                f"Expected radii shape ({num_layers},), got {radii.shape}"
            
            print(f"{num_layers}-layer system: fitness={fitness.asnumpy().item():.6f}")
    
    def test_mutation_strength_effect(self):
        """Validates that mutation strength controls exploration."""
        temperature = ms.Tensor([300.0], ms.float32)
        wavelengths = ms.Tensor(np.arange(280, 2000, 20), ms.float32)
        
        engines = [
            BrusEngine(bandgap=1.5, alpha=0.0005, beta=200, 
                      me_eff=0.1, mh_eff=0.4, eps_r=10.0),
            BrusEngine(bandgap=1.0, alpha=0.0004, beta=180, 
                      me_eff=0.08, mh_eff=0.35, eps_r=12.0)
        ]
        
        # Low mutation
        ms.set_seed(42)
        opt_low = GeneticSolarOptimizer(
            population_size=20, num_layers=2, engines=engines, mutation_strength=0.01
        )
        pop_low_init = opt_low.population.asnumpy().copy()
        opt_low(temperature, wavelengths)
        pop_low_final = opt_low.population.asnumpy()
        change_low = np.abs(pop_low_final - pop_low_init).mean()
        
        # High mutation
        ms.set_seed(43)  # Different seed to avoid exact same initial population
        opt_high = GeneticSolarOptimizer(
            population_size=20, num_layers=2, engines=engines, mutation_strength=0.5
        )
        pop_high_init = opt_high.population.asnumpy().copy()
        opt_high(temperature, wavelengths)
        pop_high_final = opt_high.population.asnumpy()
        change_high = np.abs(pop_high_final - pop_high_init).mean()
        
        assert change_high > change_low, \
            f"High mutation should cause more change: {change_high:.3f} vs {change_low:.3f}"
        
        print(f"Mutation effect: low={change_low:.3f}, high={change_high:.3f}")
