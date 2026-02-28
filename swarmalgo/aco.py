from util.define import *
from util.util import *
import numpy as np
from util.result import DiscreteResult
from function.discrete_function import DiscreteProblem

def aco_discrete(
    problem: DiscreteProblem,
    num_ants: int = 20,
    generation: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.5,
    pheromone_deposit_weight: float = 1.0,
    initial_pheromone: float = 1.0,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Ant Colony Optimization (ACO) for discrete optimization problems.
    
    This implementation uses a pheromone-based approach where ants probabilistically
    construct solutions based on pheromone trails and heuristic information.
    
    Args:
        problem: The discrete optimization problem to solve.
        num_ants: Number of ants in the colony.
        generation: Number of iterations to run.
        alpha: Pheromone importance factor (higher = more influenced by pheromone).
        beta: Heuristic importance factor (higher = more influenced by heuristic).
        evaporation_rate: Rate at which pheromone evaporates (0-1).
        pheromone_deposit_weight: Weight for pheromone deposit (Q in classical ACO).
        initial_pheromone: Initial pheromone level on all paths.
        rng_seed: Seed for random number generator.
        
    Returns:
        DiscreteResult containing the optimization results.
    """
    if num_ants <= 0:
        raise ValueError("num_ants must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if not (0 <= evaporation_rate <= 1):
        raise ValueError("evaporation_rate must be in [0, 1]")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize solutions for all ants
    solutions: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(num_ants)]
    fitness = np.array([problem.evaluate(sol) for sol in solutions])
    
    # Track best solution
    best_idx = np.argmin(fitness)
    global_best_solution = solutions[best_idx].copy()
    global_best_fitness = fitness[best_idx]
    
    # Initialize pheromone matrix (simplified: each solution has a pheromone level)
    # In practice, for combinatorial problems, this would be edge-based
    pheromone_trails: dict[str, float] = {}
    
    def get_solution_key(solution: FloatVector) -> str:
        """Convert solution to string key for pheromone lookup"""
        return ','.join(map(str, solution.astype(int)))
    
    # History tracking
    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([sol.copy() for sol in solutions], list(fitness))
    
    for gen in range(generation):
        # Construct new solutions for each ant
        new_solutions: list[FloatVector] = []
        new_fitness: list[Float] = []
        
        for ant_idx in range(num_ants):
            # Probabilistic solution construction
            # Start from a random solution or best solution
            if rng_wrapper.random() < 0.1:  # 10% chance to start from global best
                base_solution = global_best_solution.copy()
            else:
                base_solution = problem.random_solution(rng_wrapper)
            
            # Apply local search guided by pheromone
            # Generate multiple neighbors and select based on pheromone + heuristic
            num_neighbors = 5
            candidates = []
            candidate_fitness = []
            
            for _ in range(num_neighbors):
                neighbor = problem.neighbor(base_solution, 1, rng_wrapper)
                candidates.append(neighbor)
                candidate_fitness.append(problem.evaluate(neighbor))
            
            # Calculate selection probability based on pheromone and heuristic
            probabilities = []
            for candidate, fit in zip(candidates, candidate_fitness):
                key = get_solution_key(candidate)
                pheromone = pheromone_trails.get(key, initial_pheromone)
                
                # Heuristic: inverse of absolute fitness (better fitness = higher heuristic)
                # Handle special cases
                if fit == float('inf') or fit == float('-inf'):
                    heuristic = 1e-10  # Very small value for infeasible solutions
                else:
                    # Use absolute value to ensure non-negative heuristic
                    heuristic = 1.0 / (abs(fit) + 1e-9)
                
                # ACO probability formula: τ^α * η^β
                prob = (pheromone ** alpha) * (heuristic ** beta)
                probabilities.append(prob)
            
            # Normalize probabilities
            prob_sum = sum(probabilities)
            if prob_sum > 0:
                probabilities = [p / prob_sum for p in probabilities]
            else:
                probabilities = [1.0 / num_neighbors] * num_neighbors
            
            # Select solution based on probability
            selected_idx = rng_wrapper.rng.choice(num_neighbors, p=probabilities)
            selected_solution = candidates[selected_idx]
            selected_fitness = candidate_fitness[selected_idx]
            
            new_solutions.append(selected_solution)
            new_fitness.append(selected_fitness)
            
            # Update global best
            if selected_fitness < global_best_fitness:
                global_best_fitness = selected_fitness
                global_best_solution = selected_solution.copy()
        
        # Pheromone evaporation
        for key in list(pheromone_trails.keys()):
            pheromone_trails[key] *= (1 - evaporation_rate)
            # Remove very low pheromone trails to save memory
            if pheromone_trails[key] < 1e-10:
                del pheromone_trails[key]
        
        # Pheromone deposit
        for solution, fit in zip(new_solutions, new_fitness):
            key = get_solution_key(solution)
            # Better solutions deposit more pheromone
            # Deposit amount inversely proportional to absolute fitness
            if fit == float('inf') or fit == float('-inf'):
                deposit = 1e-10  # Minimal deposit for infeasible solutions
            else:
                deposit = pheromone_deposit_weight / (abs(fit) + 1e-9)
            
            if key in pheromone_trails:
                pheromone_trails[key] += deposit
            else:
                pheromone_trails[key] = initial_pheromone + deposit
        
        # Extra pheromone for global best solution (elitist strategy)
        best_key = get_solution_key(global_best_solution)
        if global_best_fitness == float('inf') or global_best_fitness == float('-inf'):
            elite_deposit = 1e-10
        else:
            elite_deposit = pheromone_deposit_weight / (abs(global_best_fitness) + 1e-9)
        if best_key in pheromone_trails:
            pheromone_trails[best_key] += elite_deposit
        else:
            pheromone_trails[best_key] = initial_pheromone + elite_deposit
        
        # Update solutions
        solutions = new_solutions
        fitness = np.array(new_fitness)
        
        # Track history
        avg_pheromone = np.mean(list(pheromone_trails.values())) if pheromone_trails else initial_pheromone
        history.add(
            [sol.copy() for sol in solutions], 
            list(fitness),
            f"gen={gen+1}, best={global_best_fitness:.4f}, avg_pheromone={avg_pheromone:.4f}, trails={len(pheromone_trails)}"
        )
    
    total_time = timer.stop()
    
    # Get the best solution from history
    best_x, best_value = history.get_best_value()
    
    return DiscreteResult(
        algorithm="Ant Colony Optimization",
        problem=problem,
        time=total_time,
        last_x=solutions,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )
