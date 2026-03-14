from util.define import *
from util.util import *
import numpy as np
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem, TSPFunction, KnapsackFunction, GraphColoringFunction

def abc_continuous(
    problem: ContinuousProblem,
    population_size: int = 50,
    generation: int = 100,
    limit: int = 10,
    rng_seed: int | None = None,
    single_dimension_update: bool = False
) -> ContinuousResult:
    """
    Artificial Bee Colony (ABC) algorithm for continuous optimization problems, with whole dimension update.

    Args:
        problem: The continuous optimization problem to solve.
        population_size: The total number of bees (half employed, half onlookers).
        generation: The number of iterations to run the algorithm.
        limit: The number of trials after which a food source is abandoned.
        rng_seed: Seed for the random number generator for reproducibility.
        single_dimension_update: If True, update one dimension at a time (original ABC). If False, update all dimensions at once.
    """
    if population_size <= 0 or population_size % 2 != 0:
        raise ValueError("population_size must be a positive even number")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    history = HistoryEntry()

    timer = TimerWrapper()
    timer.start()

    num_employed = population_size // 2
    
    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(num_employed)]
    fitness = np.array([problem.evaluate(p) for p in population])
    trials = np.zeros(num_employed)

    history.add([p.copy() for p in population], list(fitness))

    for _ in range(generation):
        # Employed Bee Phase 
        for i in range(num_employed):
            # Select a random partner solution
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            if single_dimension_update:
                # Update one dimension at a time (original ABC)
                j = rng_wrapper.rng.integers(0, problem.dimension)  # Randomly select a dimension
                phi = rng_wrapper.rng.uniform(-1, 1)
                new_solution = population[i].copy()
                new_solution[j] += phi * (population[i][j] - partner[j])
            else:
                phi = rng_wrapper.rng.uniform(-1, 1, size=problem.dimension)
                new_solution = population[i] + phi * (population[i] - partner)

            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = problem.evaluate(new_solution)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bee Phase
        # Taking inverse for minimization problems
        # transform fitness to avoid non-negative
        transformed_fitness = np.where(
            fitness >= 0,
            1.0 / (1.0 + fitness),
            1.0 + np.abs(fitness)
        )
        probabilities = transformed_fitness / np.sum(transformed_fitness)

        for _ in range(num_employed): # Onlookers = employed bees
            # Select a food source based on probability (roulette wheel selection)
            i = rng_wrapper.rng.choice(num_employed, p=probabilities)
            
            # Select a random partner solution
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            # Generate a new candidate solution
            if single_dimension_update:
                j = rng_wrapper.rng.integers(0, problem.dimension)
                phi = rng_wrapper.rng.uniform(-1, 1)

                new_solution = population[i].copy()
                new_solution[j] += phi * (population[i][j] - partner[j])
            else:
                phi = rng_wrapper.rng.uniform(-1, 1, size=problem.dimension)
                new_solution = population[i] + phi * (population[i] - partner)

            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = problem.evaluate(new_solution)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout Bee Phase
        # reset if trial exceeds limit
        for i in range(num_employed):
            if trials[i] >= limit:
                population[i] = rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension)
                fitness[i] = problem.evaluate(population[i])
                trials[i] = 0

        history.add([p.copy() for p in population], list(fitness), info=f"prob: {probabilities}, trials: {trials}")

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return ContinuousResult(
        algorithm=f"Artificial Bee Colony(NP={population_size}, Gen={generation}, Lim={limit})",
        short_name="ABC",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )

def abc_discrete_tsp(
    problem: TSPFunction,
    population_size: int = 50,
    generation: int = 100,
    limit: int = 10,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Artificial Bee Colony (ABC) algorithm for TSP problems.
    Employed bees find neighbors using swap/insert operations based on differences with a random partner bee.

    Args:
        problem: The TSP problem to solve.
        population_size: The total number of bees (half employed, half onlookers).
        generation: The number of iterations to run the algorithm.
        limit: The number of trials after which a food source is abandoned.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0 or population_size % 2 != 0:
        raise ValueError("population_size must be a positive even number")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    num_employed = population_size // 2
    
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(num_employed)]
    fitness = np.array([problem.evaluate(p) for p in population])
    trials = np.zeros(num_employed)

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        # Employed Bee Phase - explore based on difference with partner
        for i in range(num_employed):
            # Select a random partner bee
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            # Find positions where current solution differs from partner
            diff_positions = np.where(population[i] != partner)[0]
            
            new_solution = population[i].copy()
            if len(diff_positions) > 0:
                # Number of swaps based on difference magnitude
                num_swaps = max(1, min(3, len(diff_positions) // 4))
                
                # Perform swaps/inserts at positions where they differ
                for _ in range(num_swaps):
                    # Select a position where they differ
                    pos = rng_wrapper.rng.choice(diff_positions)
                    
                    # Swap with a random position to explore neighborhood
                    swap_pos = rng_wrapper.rng.integers(0, problem.dimension)
                    new_solution[pos], new_solution[swap_pos] = new_solution[swap_pos], new_solution[pos]
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bee Phase - exploit best solutions
        transformed_fitness = np.where(
            fitness >= 0,
            1.0 / (1.0 + fitness),
            1.0 + np.abs(fitness)
        )
        probabilities = transformed_fitness / np.sum(transformed_fitness)

        for _ in range(num_employed):
            i = rng_wrapper.rng.choice(num_employed, p=probabilities)
            
            # Similar exploration as employed bees
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            diff_positions = np.where(population[i] != partner)[0]
            new_solution = population[i].copy()
            
            if len(diff_positions) > 0:
                num_swaps = max(1, min(2, len(diff_positions) // 5))
                for _ in range(num_swaps):
                    # 2-opt style swap for TSP
                    indices = sorted(rng_wrapper.rng.choice(problem.dimension, size=2, replace=False))
                    i_swap, j_swap = indices[0], indices[1]
                    new_solution[i_swap:j_swap+1] = new_solution[i_swap:j_swap+1][::-1]
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout Bee Phase - abandon exhausted sources
        for i in range(num_employed):
            if trials[i] > limit:
                population[i] = problem.random_solution(rng_wrapper)
                fitness[i] = problem.evaluate(population[i])
                trials[i] = 0

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Artificial Bee Colony(NP={population_size}, Gen={generation}, Lim={limit})",
        short_name="ABC-TSP",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def abc_discrete_knapsack(
    problem: KnapsackFunction,
    population_size: int = 50,
    generation: int = 100,
    limit: int = 10,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Artificial Bee Colony (ABC) algorithm for Knapsack problems.
    Employed bees find neighbors using XOR operation with a random partner bee.

    Args:
        problem: The Knapsack problem to solve.
        population_size: The total number of bees (half employed, half onlookers).
        generation: The number of iterations to run the algorithm.
        limit: The number of trials after which a food source is abandoned.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0 or population_size % 2 != 0:
        raise ValueError("population_size must be a positive even number")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    num_employed = population_size // 2
    
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(num_employed)]
    fitness = np.array([problem.evaluate(p) for p in population])
    trials = np.zeros(num_employed)

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        # Employed Bee Phase - explore using XOR with partner
        for i in range(num_employed):
            # Select a random partner bee
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            # XOR operation: flip bits where current and partner differ
            # phi controls how much we move towards XOR result
            phi = rng_wrapper.rng.uniform(0, 1)
            
            # Find positions where XOR would flip bits
            xor_result = (population[i].astype(int) ^ partner.astype(int)).astype(float)
            
            # Apply XOR selectively based on phi
            # For each dimension where XOR would flip, decide probabilistically
            new_solution = population[i].copy()
            for d in range(problem.dimension):
                if xor_result[d] == 1.0:  # Position where XOR would flip
                    if rng_wrapper.random() < phi:
                        new_solution[d] = 1 - new_solution[d]  # Flip the bit
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bee Phase
        transformed_fitness = np.where(
            fitness >= 0,
            1.0 / (1.0 + fitness),
            1.0 + np.abs(fitness)
        )
        probabilities = transformed_fitness / np.sum(transformed_fitness)

        for _ in range(num_employed):
            i = rng_wrapper.rng.choice(num_employed, p=probabilities)
            
            # Similar XOR-based exploration
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            phi = rng_wrapper.rng.uniform(0, 1)
            xor_result = (population[i].astype(int) ^ partner.astype(int)).astype(float)
            
            new_solution = population[i].copy()
            for d in range(problem.dimension):
                if xor_result[d] == 1.0:
                    if rng_wrapper.random() < phi:
                        new_solution[d] = 1 - new_solution[d]
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout Bee Phase
        for i in range(num_employed):
            if trials[i] > limit:
                population[i] = problem.random_solution(rng_wrapper)
                fitness[i] = problem.evaluate(population[i])
                trials[i] = 0

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Artificial Bee Colony(NP={population_size}, Gen={generation}, Lim={limit})",
        short_name="ABC-KP",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def abc_discrete_graphcoloring(
    problem: GraphColoringFunction,
    population_size: int = 50,
    generation: int = 100,
    limit: int = 10,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Artificial Bee Colony (ABC) algorithm for Graph Coloring problems.
    Employed bees find neighbors by modifying colors based on difference with a random partner bee.

    Args:
        problem: The Graph Coloring problem to solve.
        population_size: The total number of bees (half employed, half onlookers).
        generation: The number of iterations to run the algorithm.
        limit: The number of trials after which a food source is abandoned.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if population_size <= 0 or population_size % 2 != 0:
        raise ValueError("population_size must be a positive even number")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    num_employed = population_size // 2
    
    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(num_employed)]
    fitness = np.array([problem.evaluate(p) for p in population])
    trials = np.zeros(num_employed)

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([p.copy() for p in population], list(fitness))

    for gen in range(generation):
        # Employed Bee Phase - modify colors based on difference with partner
        for i in range(num_employed):
            # Select a random partner bee
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            # phi controls intensity of change
            phi = rng_wrapper.rng.uniform(-1, 1)
            
            # Find dimensions where current and partner differ
            diff_mask = population[i] != partner
            
            new_solution = population[i].copy()
            
            # Modify colors at differing positions
            for d in range(problem.dimension):
                if diff_mask[d] and rng_wrapper.random() < abs(phi):
                    # Move towards or away from partner's color
                    if phi > 0:
                        # Move towards partner
                        new_solution[d] = partner[d]
                    else:
                        # Move away - pick a different random color
                        max_color = int(max(np.max(population[i]), np.max(partner)))
                        available_colors = [c for c in range(max_color + 1) if c != int(population[i][d])]
                        if available_colors:
                            new_solution[d] = rng_wrapper.rng.choice(available_colors)
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bee Phase
        transformed_fitness = np.where(
            fitness >= 0,
            1.0 / (1.0 + fitness),
            1.0 + np.abs(fitness)
        )
        probabilities = transformed_fitness / np.sum(transformed_fitness)

        for _ in range(num_employed):
            i = rng_wrapper.rng.choice(num_employed, p=probabilities)
            
            # Similar color modification based on partner
            partner_idx = rng_wrapper.rng.choice([idx for idx in range(num_employed) if idx != i])
            partner = population[partner_idx]
            
            phi = rng_wrapper.rng.uniform(-1, 1)
            diff_mask = population[i] != partner
            
            new_solution = population[i].copy()
            for d in range(problem.dimension):
                if diff_mask[d] and rng_wrapper.random() < abs(phi):
                    if phi > 0:
                        new_solution[d] = partner[d]
                    else:
                        max_color = int(max(np.max(population[i]), np.max(partner)))
                        available_colors = [c for c in range(max_color + 1) if c != int(population[i][d])]
                        if available_colors:
                            new_solution[d] = rng_wrapper.rng.choice(available_colors)
            
            new_fitness = problem.evaluate(new_solution)
            
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout Bee Phase
        for i in range(num_employed):
            if trials[i] > limit:
                population[i] = problem.random_solution(rng_wrapper)
                fitness[i] = problem.evaluate(population[i])
                trials[i] = 0

        history.add([p.copy() for p in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Artificial Bee Colony(NP={population_size}, Gen={generation}, Lim={limit})",
        short_name="ABC-GC",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def abc_discrete(
    problem: DiscreteProblem,
    population_size: int = 50,
    generation: int = 100,
    limit: int = 10,
    step_size: int = 1,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Artificial Bee Colony (ABC) dispatcher for discrete optimization problems.
    Automatically selects the appropriate specialized ABC implementation based on problem type.
    
    Specialized implementations:
    - TSPFunction: Uses swap/insert operations based on difference with partner
    - KnapsackFunction: Uses XOR operations for bit flipping
    - GraphColoringFunction: Modifies colors based on difference with partner
    - Generic: Uses neighbor function from problem definition

    Args:
        problem: The discrete optimization problem to solve.
        population_size: The total number of bees (half employed, half onlookers).
        generation: The number of iterations to run the algorithm.
        limit: The number of trials after which a food source is abandoned.
        step_size: The step size for neighborhood generation (only for generic problems).
        rng_seed: Seed for the random number generator for reproducibility.
    
    Returns:
        DiscreteResult containing the optimization results.
    """
    if isinstance(problem, TSPFunction):
        return abc_discrete_tsp(problem, population_size, generation, limit, rng_seed)
    elif isinstance(problem, KnapsackFunction):
        return abc_discrete_knapsack(problem, population_size, generation, limit, rng_seed)
    elif isinstance(problem, GraphColoringFunction):
        return abc_discrete_graphcoloring(problem, population_size, generation, limit, rng_seed)
    else:
        # Generic ABC for other discrete problems
        if population_size <= 0 or population_size % 2 != 0:
            raise ValueError("population_size must be a positive even number")
        if generation <= 0:
            raise ValueError("generation must be positive")
        if limit <= 0:
            raise ValueError("limit must be positive")

        rng_wrapper = RNGWrapper(rng_seed)
        timer = TimerWrapper()
        timer.start()

        num_employed = population_size // 2
        
        population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(num_employed)]
        fitness = np.array([problem.evaluate(p) for p in population])
        trials = np.zeros(num_employed)

        history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
        history.add([p.copy() for p in population], list(fitness))

        for gen in range(generation):
            # Employed Bee Phase 
            for i in range(num_employed):
                new_solution = problem.neighbor(population[i], step_size, rng_wrapper)
                new_fitness = problem.evaluate(new_solution)
                
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1

            transformed_fitness = np.where(
                fitness >= 0,
                1.0 / (1.0 + fitness),
                1.0 + np.abs(fitness)
            )
            probabilities = transformed_fitness / np.sum(transformed_fitness)

            for _ in range(num_employed):
                i = rng_wrapper.rng.choice(num_employed, p=probabilities)
                
                new_solution = problem.neighbor(population[i], step_size, rng_wrapper)
                new_fitness = problem.evaluate(new_solution)
                
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1

            for i in range(num_employed):
                if trials[i] > limit:
                    population[i] = problem.random_solution(rng_wrapper)
                    fitness[i] = problem.evaluate(population[i])
                    trials[i] = 0

            history.add([p.copy() for p in population], list(fitness))

        total_time = timer.stop()
        best_x, best_value = history.get_best_value()

        return DiscreteResult(
            algorithm=f"Artificial Bee Colony(NP={population_size}, Gen={generation}, Lim={limit})",
            short_name="ABC",
            problem=problem,
            time=total_time,
            last_x=population,
            last_value=list(fitness),
            best_x=best_x,
            best_value=best_value,
            iterations=generation,
            rng_seed=rng_wrapper.get_seed(),
            history=history
        )