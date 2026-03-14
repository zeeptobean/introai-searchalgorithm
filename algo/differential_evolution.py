from function.discrete_function import DiscreteProblem, TSPFunction, KnapsackFunction, GraphColoringFunction
from util.define import *
from util.util import *
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem

def differential_evolution_continuous(
    problem: ContinuousProblem, 
    population_size: int = 20, 
    mutation_factor: Float = 0.8, 
    crossover_rate: Float = 0.7, 
    generation: int = 100, 
    rng_seed: int | None = None
) -> ContinuousResult:
    if mutation_factor < 0 or mutation_factor > 2:
        raise ValueError("mutation_factor must be in [0, 2]")
    if crossover_rate < 0 or crossover_rate > 1:
        raise ValueError("crossover_rate must be in [0, 1]")
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    history = HistoryEntry()

    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in population])

    history_x: list[list[FloatVector]] = [[x.copy() for x in population]]
    history_value: list[list[Float]] = [list(fitness)]
    history.add([x.copy() for x in population], list(fitness))

    for _ in range(generation):
        for i in range(population_size):
            # Mutation
            idx_list = [idx for idx in range(population_size) if idx != i]
            a_idx, b_idx, c_idx = rng_wrapper.rng.choice(idx_list, 3, replace=False)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]
            mutant: FloatVector = np.clip(a + mutation_factor * (b - c), lower_bound, upper_bound)

            # Crossover
            crossover_mask = rng_wrapper.rng.random(problem.dimension) < crossover_rate
            trial = np.where(crossover_mask, mutant, population[i])
            trial_fitness = problem.evaluate(trial)

            # Selection
            if trial_fitness < fitness[i]:  # Minimizing fitness
                population[i] = trial
                fitness[i] = trial_fitness
        history.add([x.copy() for x in population], list(fitness))
    total_time = timer.stop()

    best_x, best_value = history.get_best_value()

    return ContinuousResult(
        algorithm=f"Differential Evolution(F={mutation_factor:.4f}, CR={crossover_rate:.4f}), NP={population_size}, Gen={generation})",
        short_name="DE",
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

def differential_evolution_discrete_tsp(
    problem: TSPFunction, 
    population_size: int = 20, 
    mutation_factor: Float = 0.8, 
    crossover_rate: Float = 0.7, 
    generation: int = 100, 
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Differential Evolution for TSP problems.
    Uses position-based mutation from 3 individuals and order crossover.
    
    Args:
        problem: The TSP problem to solve.
        population_size: The number of individuals in the population.
        mutation_factor: Controls the difference vector amplification.
        crossover_rate: Probability of crossover.
        generation: Number of generations.
        rng_seed: Seed for the random number generator.
    """
    if mutation_factor < 0 or mutation_factor > 2:
        raise ValueError("mutation_factor must be in [0, 2]")
    if crossover_rate < 0 or crossover_rate > 1:
        raise ValueError("crossover_rate must be in [0, 1]")
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in population])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([x.copy() for x in population], list(fitness))

    for _ in range(generation):
        for i in range(population_size):
            # Mutation: Position-based from 3 individuals
            idx_list = [idx for idx in range(population_size) if idx != i]
            a_idx, b_idx, c_idx = rng_wrapper.rng.choice(idx_list, 3, replace=False)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]
            
            # Find positions where b and c differ
            diff_positions = np.where(b != c)[0]
            
            # Create mutant starting from a
            mutant = a.copy()
            
            # Apply position-based mutation with mutation_factor probability
            if len(diff_positions) > 0:
                num_changes = int(len(diff_positions) * mutation_factor)
                num_changes = max(1, min(num_changes, len(diff_positions)))
                
                selected_positions = rng_wrapper.rng.choice(diff_positions, size=num_changes, replace=False)
                
                for pos in selected_positions:
                    # Find the value from b at this position
                    value_b = b[pos]
                    # Find where this value is in mutant
                    current_pos = np.where(mutant == value_b)[0][0]
                    # Swap to put it at the right position
                    mutant[pos], mutant[current_pos] = mutant[current_pos], mutant[pos]

            # Crossover: Order Crossover (OX)
            if rng_wrapper.rng.random() < crossover_rate:
                parent = population[i]
                trial = mutant.copy()
                
                # Select two crossover points
                point1, point2 = sorted(rng_wrapper.rng.choice(problem.dimension, 2, replace=False))
                
                # Copy segment from parent
                segment = parent[point1:point2+1]
                trial[point1:point2+1] = segment
                
                # Fill remaining positions from mutant
                mutant_vals = [val for val in mutant if val not in segment]
                idx = 0
                for j in range(problem.dimension):
                    if j < point1 or j > point2:
                        trial[j] = mutant_vals[idx]
                        idx += 1
            else:
                trial = mutant
            
            trial_fitness = problem.evaluate(trial)

            # Selection
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
        history.add([x.copy() for x in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Differential Evolution(F={mutation_factor:.4f}, CR={crossover_rate:.4f}), NP={population_size}, Gen={generation})",
        short_name="DE-TSP",
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


def differential_evolution_discrete_knapsack(
    problem: KnapsackFunction, 
    population_size: int = 20, 
    mutation_factor: Float = 0.8, 
    crossover_rate: Float = 0.7, 
    generation: int = 100, 
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Differential Evolution for Knapsack problems.
    Uses binary mutation selecting bits from 3 individuals.
    
    Args:
        problem: The Knapsack problem to solve.
        population_size: The number of individuals in the population.
        mutation_factor: Controls the mutation intensity.
        crossover_rate: Probability of crossover for each dimension.
        generation: Number of generations.
        rng_seed: Seed for the random number generator.
    """
    if mutation_factor < 0 or mutation_factor > 2:
        raise ValueError("mutation_factor must be in [0, 2]")
    if crossover_rate < 0 or crossover_rate > 1:
        raise ValueError("crossover_rate must be in [0, 1]")
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in population])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([x.copy() for x in population], list(fitness))

    for _ in range(generation):
        for i in range(population_size):
            # Binary Mutation: Select bits from 3 individuals
            idx_list = [idx for idx in range(population_size) if idx != i]
            a_idx, b_idx, c_idx = rng_wrapper.rng.choice(idx_list, 3, replace=False)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]
            
            # Create mutant using binary operations
            # For each dimension: mutant[d] = a[d] XOR (b[d] XOR c[d]) if mutation applies
            mutant = np.zeros(problem.dimension)
            
            for d in range(problem.dimension):
                # Apply XOR-based mutation with probability based on mutation_factor
                if rng_wrapper.random() < mutation_factor:
                    # XOR of b and c, then use it to decide mutation direction
                    diff = int(b[d]) ^ int(c[d])
                    if diff == 1:  # b and c differ
                        # Choose randomly between b[d] and c[d]
                        mutant[d] = b[d] if rng_wrapper.random() < 0.5 else c[d]
                    else:  # b and c are same
                        mutant[d] = a[d]
                else:
                    mutant[d] = a[d]

            # Binary Crossover
            trial = np.zeros(problem.dimension)
            for d in range(problem.dimension):
                if rng_wrapper.random() < crossover_rate:
                    trial[d] = mutant[d]
                else:
                    trial[d] = population[i][d]
            
            trial_fitness = problem.evaluate(trial)

            # Selection
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
        history.add([x.copy() for x in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Differential Evolution(F={mutation_factor:.4f}, CR={crossover_rate:.4f}), NP={population_size}, Gen={generation})",
        short_name="DE-KP",
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


def differential_evolution_discrete_graphcoloring(
    problem: GraphColoringFunction, 
    population_size: int = 20, 
    mutation_factor: Float = 0.8, 
    crossover_rate: Float = 0.7, 
    generation: int = 100, 
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Differential Evolution for Graph Coloring problems.
    Uses continuous-to-discrete mapping for mutation from 3 individuals.
    
    Args:
        problem: The Graph Coloring problem to solve.
        population_size: The number of individuals in the population.
        mutation_factor: Differential weight for mutation.
        crossover_rate: Probability of crossover for each dimension.
        generation: Number of generations.
        rng_seed: Seed for the random number generator.
    """
    if mutation_factor < 0 or mutation_factor > 2:
        raise ValueError("mutation_factor must be in [0, 2]")
    if crossover_rate < 0 or crossover_rate > 1:
        raise ValueError("crossover_rate must be in [0, 1]")
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in population])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([x.copy() for x in population], list(fitness))

    for _ in range(generation):
        for i in range(population_size):
            # Mutation: Map from continuous space
            idx_list = [idx for idx in range(population_size) if idx != i]
            a_idx, b_idx, c_idx = rng_wrapper.rng.choice(idx_list, 3, replace=False)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]
            
            # Apply DE mutation formula in continuous space
            # mutant_continuous = a + F * (b - c)
            mutant_continuous = a + mutation_factor * (b - c)
            
            # Map to discrete space using rounding and clipping
            # Determine max color from population to avoid unnecessary colors
            max_color_in_pop = int(max(np.max(a), np.max(b), np.max(c)))
            
            # Round to nearest integer and clip to valid range
            mutant = np.clip(np.round(mutant_continuous), 0, max_color_in_pop).astype(float)

            # Discrete Crossover: Choose colors from mutant or parent
            trial = np.zeros(problem.dimension)
            for d in range(problem.dimension):
                if rng_wrapper.random() < crossover_rate:
                    trial[d] = mutant[d]
                else:
                    trial[d] = population[i][d]
            
            trial_fitness = problem.evaluate(trial)

            # Selection
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
        history.add([x.copy() for x in population], list(fitness))

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Differential Evolution(F={mutation_factor:.4f}, CR={crossover_rate:.4f}), NP={population_size}, Gen={generation})",
        short_name="DE-GC",
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


def differential_evolution_discrete(
    problem: DiscreteProblem, 
    population_size: int = 20, 
    mutation_factor: Float = 0.8, 
    crossover_rate: Float = 0.7, 
    generation: int = 100, 
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Differential Evolution dispatcher for discrete optimization problems.
    Automatically selects the appropriate specialized implementation based on problem type.
    
    Specialized implementations:
    - TSPFunction: Uses position-based mutation from 3 individuals
    - KnapsackFunction: Uses binary mutation with XOR operations
    - GraphColoringFunction: Uses continuous-to-discrete mapping
    - Generic: Uses neighbor-based approach
    
    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of individuals in the population.
        mutation_factor: Controls mutation intensity.
        crossover_rate: Probability of crossover.
        generation: Number of generations.
        rng_seed: Seed for the random number generator.
    
    Returns:
        DiscreteResult containing the optimization results.
    """
    if isinstance(problem, TSPFunction):
        return differential_evolution_discrete_tsp(problem, population_size, mutation_factor, crossover_rate, generation, rng_seed)
    elif isinstance(problem, KnapsackFunction):
        return differential_evolution_discrete_knapsack(problem, population_size, mutation_factor, crossover_rate, generation, rng_seed)
    elif isinstance(problem, GraphColoringFunction):
        return differential_evolution_discrete_graphcoloring(problem, population_size, mutation_factor, crossover_rate, generation, rng_seed)
    else:
        # Generic DE for other discrete problems
        if mutation_factor < 0 or mutation_factor > 2:
            raise ValueError("mutation_factor must be in [0, 2]")
        if crossover_rate < 0 or crossover_rate > 1:
            raise ValueError("crossover_rate must be in [0, 1]")
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        if generation <= 0:
            raise ValueError("generation must be positive")

        rng_wrapper = RNGWrapper(rng_seed)
        timer = TimerWrapper()
        timer.start()

        population: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
        fitness = np.array([problem.evaluate(x) for x in population])

        history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
        history.add([x.copy() for x in population], list(fitness))

        for _ in range(generation):
            for i in range(population_size):
                # Mutation
                idx_list = [idx for idx in range(population_size) if idx != i]
                a_idx = rng_wrapper.rng.choice(idx_list)
                
                swap_cnt = int(rng_wrapper.rng.integers(1, max(2, int(problem.dimension * mutation_factor))))
                mutant = problem.neighbor(population[a_idx], swap_cnt, rng_wrapper)

                # Crossover
                if rng_wrapper.rng.random() < crossover_rate:
                    parent = population[i]
                    is_permutation = len(np.unique(parent.astype(int))) == len(parent)
                    
                    if is_permutation:
                        trial = mutant.copy()
                        point1, point2 = sorted(rng_wrapper.rng.choice(problem.dimension, 2, replace=False))
                        segment = parent[point1:point2+1]
                        trial[point1:point2+1] = segment
                        
                        mutant_vals = [val for val in mutant if val not in segment]
                        idx = 0
                        for j in range(problem.dimension):
                            if j < point1 or j > point2:
                                trial[j] = mutant_vals[idx]
                                idx += 1
                    else:
                        crossover_mask = rng_wrapper.rng.random(problem.dimension) < 0.5
                        trial = np.where(crossover_mask, mutant, parent)
                else:
                    trial = population[i].copy()
                
                trial_fitness = problem.evaluate(trial)

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
            history.add([x.copy() for x in population], list(fitness))

        total_time = timer.stop()
        best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm=f"Differential Evolution(F={mutation_factor:.4f}, CR={crossover_rate:.4f}), NP={population_size}, Gen={generation})",
        short_name="DE",
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