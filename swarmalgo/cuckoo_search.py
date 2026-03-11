from util.define import *
from util.util import *
import math 
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem, TSPFunction, KnapsackFunction, GraphColoringFunction

# Mantegna's algorithm for continuous Lévy flights
def _levy_flight_step(rng_wrapper: RNGWrapper, dimension: int, beta: Float) -> FloatVector:
    sigma_u_num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    sigma_u_den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (sigma_u_num / sigma_u_den) ** (1 / beta)
    sigma_v = 1.0

    u = rng_wrapper.rng.normal(0, sigma_u, size=dimension)
    v = rng_wrapper.rng.normal(0, sigma_v, size=dimension)
    
    step = u / (np.abs(v) ** (1 / beta))
    return step


def _discrete_levy_flight(rng_wrapper: RNGWrapper, beta: Float, scale: Float = 1.0) -> int:
    """
    Discrete Lévy flight: returns an integer step size based on Lévy distribution.
    
    Uses Mantegna's method to generate a Lévy-distributed random variable,
    then converts it to a discrete integer step size.
    
    Args:
        rng_wrapper: Random number generator wrapper
        beta: Exponent for the Lévy distribution (typically 1.5)
        scale: Scaling factor for the step size
        
    Returns:
        Integer step size (at least 1)
    """
    sigma_u_num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    sigma_u_den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (sigma_u_num / sigma_u_den) ** (1 / beta)
    sigma_v = 1.0

    u = rng_wrapper.rng.normal(0, sigma_u)
    v = rng_wrapper.rng.normal(0, sigma_v)
    
    step = u / (abs(v) ** (1 / beta))
    
    # Convert to discrete integer step (at least 1)
    discrete_step = max(1, int(abs(step) * scale))
    
    return discrete_step

"""
Cuckoo search, continuous
alpha: Step size scaling factor for Lévy flights.
discovery_rate: Fraction of worst nests to be abandoned.
beta: Exponent for the Lévy distribution (typically in [1, 2]).
generation: The number of generations to run the algorithm.
rng_seed: Seed for the random number generator.
"""
def cuckoo_search_continuous(
    problem: ContinuousProblem,
    population_size: int = 25,
    alpha: Float = 0.01,
    discovery_rate: Float = 0.25,
    beta: Float = 1.5,
    generation: int = 100,
    rng_seed: int | None = None
) -> ContinuousResult:
    if not (0 < discovery_rate < 1):
        raise ValueError("discovery_rate must be in (0, 1)")
    if not (1 <= beta <= 2):
        raise ValueError("beta for Lévy flight must be in [1, 2]")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")

    rng_wrapper = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    history = HistoryEntry()

    timer = TimerWrapper()
    timer.start()

    nests: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in nests])

    history.add([x.copy() for x in nests], list(fitness))

    for _ in range(generation):
        # Generate new solutions (cuckoos) via Lévy flights
        for i in range(population_size):
            step = _levy_flight_step(rng_wrapper, problem.dimension, beta)
            new_nest = nests[i] + alpha * step
            new_nest = np.clip(new_nest, lower_bound, upper_bound)
            new_fitness = problem.evaluate(new_nest)

            # Choose a random nest to compare
            j = rng_wrapper.rng.integers(0, population_size)
            if new_fitness < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fitness

        # Abandon a fraction of the worst nests and build new ones
        sorted_indices = np.argsort(fitness)
        num_abandon = max(1, int(discovery_rate * population_size))
        worst_indices = sorted_indices[-num_abandon:]

        # re-eval worst nests after abandoning 
        for k in worst_indices:
            nests[k] = rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension)
            fitness[k] = problem.evaluate(nests[k])

        history.add([x.copy() for x in nests], list(fitness))

    total_time = timer.stop()

    best_x, best_value = history.get_best_value()

    return ContinuousResult(
        algorithm="Cuckoo Search",
        problem=problem,
        time=total_time,
        last_x=nests,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def cuckoo_search_discrete_tsp(
    problem: DiscreteProblem,
    population_size: int = 25,
    alpha: Float = 1.0,
    discovery_rate: Float = 0.25,
    beta: Float = 1.5,
    generation: int = 100,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Cuckoo Search (CS) for TSP using Discrete Lévy Flights.
    
    For TSP, discrete Lévy flights are implemented as:
    - Step size = number of swap operations to perform
    - Larger Lévy steps = more swaps = bigger jumps in solution space
    - Maintains valid permutations throughout
    
    Args:
        problem: The TSP problem to solve.
        population_size: The number of nests (solutions) in the population.
        alpha: Step size scaling factor for Lévy flights.
        discovery_rate: Fraction of worst nests to be abandoned.
        beta: Exponent for the Lévy distribution (typically 1.5).
        generation: The number of generations to run the algorithm.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if not (0 < discovery_rate < 1):
        raise ValueError("discovery_rate must be in (0, 1)")
    if not (1 <= beta <= 2):
        raise ValueError("beta for Lévy flight must be in [1, 2]")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if not isinstance(problem, TSPFunction):
        raise ValueError("cuckoo_search_discrete_tsp requires a TSPFunction problem instance")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize nests (solutions)
    nests: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in nests])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([x.copy() for x in nests], list(fitness))

    for gen in range(generation):
        # Generate new solutions via discrete Lévy flights
        for i in range(population_size):
            # Discrete Lévy flight: determine number of swaps to perform
            num_swaps = _discrete_levy_flight(rng_wrapper, beta, scale=alpha)
            
            # Perform swaps to create new solution
            new_nest = nests[i].copy()
            for _ in range(num_swaps):
                # Random 2-opt swap
                idx1, idx2 = sorted(rng_wrapper.rng.choice(problem.dimension, size=2, replace=False))
                new_nest[idx1:idx2+1] = new_nest[idx1:idx2+1][::-1]
            
            new_fitness = problem.evaluate(new_nest)

            # Choose a random nest to compare
            j = rng_wrapper.rng.integers(0, population_size)
            if new_fitness < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fitness

        # Abandon worst nests and build new ones
        sorted_indices = np.argsort(fitness)
        num_abandon = max(1, int(discovery_rate * population_size))
        worst_indices = sorted_indices[-num_abandon:]

        for k in worst_indices:
            if rng_wrapper.random() < 0.5:
                # Generate completely new random solution
                nests[k] = problem.random_solution(rng_wrapper)
            else:
                # Create from a good solution with large perturbation
                good_idx = sorted_indices[rng_wrapper.rng.integers(0, population_size // 3)]
                large_step = _discrete_levy_flight(rng_wrapper, beta, scale=alpha * 2)
                
                nests[k] = nests[good_idx].copy()
                for _ in range(large_step):
                    idx1, idx2 = sorted(rng_wrapper.rng.choice(problem.dimension, size=2, replace=False))
                    nests[k][idx1:idx2+1] = nests[k][idx1:idx2+1][::-1]
            
            fitness[k] = problem.evaluate(nests[k])

        history.add([x.copy() for x in nests], list(fitness), f"gen={gen+1}, best={np.min(fitness):.4f}")

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Cuckoo Search (TSP)",
        problem=problem,
        time=total_time,
        last_x=nests,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def cuckoo_search_discrete_knapsack(
    problem: DiscreteProblem,
    population_size: int = 25,
    alpha: Float = 1.0,
    discovery_rate: Float = 0.25,
    beta: Float = 1.5,
    generation: int = 100,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Cuckoo Search (CS) for Knapsack using Set-based Operations.
    
    For Knapsack, discrete Lévy flights are implemented as:
    - Step size = number of items to flip (0->1 or 1->0)
    - Larger Lévy steps = more items flipped = bigger changes
    - Uses set-based thinking: add/remove multiple items at once
    
    Args:
        problem: The Knapsack problem to solve.
        population_size: The number of nests (solutions) in the population.
        alpha: Step size scaling factor for Lévy flights.
        discovery_rate: Fraction of worst nests to be abandoned.
        beta: Exponent for the Lévy distribution (typically 1.5).
        generation: The number of generations to run the algorithm.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if not (0 < discovery_rate < 1):
        raise ValueError("discovery_rate must be in (0, 1)")
    if not (1 <= beta <= 2):
        raise ValueError("beta for Lévy flight must be in [1, 2]")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if not isinstance(problem, KnapsackFunction):
        raise ValueError("cuckoo_search_discrete_knapsack requires a KnapsackFunction problem instance")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize nests (solutions)
    nests: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in nests])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([x.copy() for x in nests], list(fitness))

    for gen in range(generation):
        # Generate new solutions via discrete Lévy flights (set-based operations)
        for i in range(population_size):
            # Discrete Lévy flight: determine number of items to flip
            num_flips = _discrete_levy_flight(rng_wrapper, beta, scale=alpha)
            num_flips = min(num_flips, problem.dimension)  # Cap at max items
            
            # Select random items to flip
            new_nest = nests[i].copy()
            flip_indices = rng_wrapper.rng.choice(problem.dimension, size=num_flips, replace=False)
            
            for idx in flip_indices:
                new_nest[idx] = 1.0 - new_nest[idx]  # Flip bit
            
            new_fitness = problem.evaluate(new_nest)

            # Choose a random nest to compare
            j = rng_wrapper.rng.integers(0, population_size)
            if new_fitness < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fitness

        # Abandon worst nests and build new ones
        sorted_indices = np.argsort(fitness)
        num_abandon = max(1, int(discovery_rate * population_size))
        worst_indices = sorted_indices[-num_abandon:]

        for k in worst_indices:
            if rng_wrapper.random() < 0.5:
                # Generate completely new random solution
                nests[k] = problem.random_solution(rng_wrapper)
            else:
                # Create from a good solution with set-based modification
                good_idx = sorted_indices[rng_wrapper.rng.integers(0, population_size // 3)]
                large_step = _discrete_levy_flight(rng_wrapper, beta, scale=alpha * 2)
                large_step = min(large_step, problem.dimension)
                
                nests[k] = nests[good_idx].copy()
                flip_indices = rng_wrapper.rng.choice(problem.dimension, size=large_step, replace=False)
                for idx in flip_indices:
                    nests[k][idx] = 1.0 - nests[k][idx]
            
            fitness[k] = problem.evaluate(nests[k])

        history.add([x.copy() for x in nests], list(fitness), f"gen={gen+1}, best={-np.min(fitness):.4f}")

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Cuckoo Search (Knapsack)",
        problem=problem,
        time=total_time,
        last_x=nests,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def cuckoo_search_discrete_graphcoloring(
    problem: DiscreteProblem,
    population_size: int = 25,
    alpha: Float = 1.0,
    discovery_rate: Float = 0.25,
    beta: Float = 1.5,
    generation: int = 100,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Cuckoo Search (CS) for Graph Coloring using Discrete Lévy Flights.
    
    For Graph Coloring, discrete Lévy flights are implemented as:
    - Step size = number of vertices to re-color
    - Larger Lévy steps = more vertices re-colored = bigger changes
    - Focuses on conflict resolution and color minimization
    
    Args:
        problem: The Graph Coloring problem to solve.
        population_size: The number of nests (solutions) in the population.
        alpha: Step size scaling factor for Lévy flights.
        discovery_rate: Fraction of worst nests to be abandoned.
        beta: Exponent for the Lévy distribution (typically 1.5).
        generation: The number of generations to run the algorithm.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if not (0 < discovery_rate < 1):
        raise ValueError("discovery_rate must be in (0, 1)")
    if not (1 <= beta <= 2):
        raise ValueError("beta for Lévy flight must be in [1, 2]")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if not isinstance(problem, GraphColoringFunction):
        raise ValueError("cuckoo_search_discrete_graphcoloring requires a GraphColoringFunction problem instance")

    rng_wrapper = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    # Initialize nests (solutions)
    nests: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in nests])

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())
    history.add([x.copy() for x in nests], list(fitness))

    for gen in range(generation):
        # Generate new solutions via discrete Lévy flights
        for i in range(population_size):
            # Discrete Lévy flight: determine number of vertices to re-color
            num_recolor = _discrete_levy_flight(rng_wrapper, beta, scale=alpha)
            num_recolor = min(num_recolor, problem.dimension)
            
            # Select random vertices to re-color
            new_nest = nests[i].copy()
            recolor_vertices = rng_wrapper.rng.choice(problem.dimension, size=num_recolor, replace=False)
            
            # Determine available colors (colors currently used)
            colors_used = set(int(c) for c in new_nest)
            max_color = max(colors_used) if colors_used else 0
            
            for vertex in recolor_vertices:
                # Try to assign a color that minimizes conflicts
                # Prefer existing colors to minimize total colors used
                available_colors = list(range(max_color + 2))  # Include one new color
                
                # Randomly select a color (biased toward existing colors)
                if rng_wrapper.random() < 0.7 and colors_used:
                    # 70% chance to use existing color
                    new_nest[vertex] = float(rng_wrapper.rng.choice(list(colors_used)))
                else:
                    # 30% chance to try any color (including new)
                    new_nest[vertex] = float(rng_wrapper.rng.choice(available_colors))
            
            new_fitness = problem.evaluate(new_nest)

            # Choose a random nest to compare
            j = rng_wrapper.rng.integers(0, population_size)
            if new_fitness < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fitness

        # Abandon worst nests and build new ones
        sorted_indices = np.argsort(fitness)
        num_abandon = max(1, int(discovery_rate * population_size))
        worst_indices = sorted_indices[-num_abandon:]

        for k in worst_indices:
            if rng_wrapper.random() < 0.5:
                # Generate completely new random solution (greedy)
                nests[k] = problem.random_solution(rng_wrapper)
            else:
                # Create from a good solution with large perturbation
                good_idx = sorted_indices[rng_wrapper.rng.integers(0, population_size // 3)]
                large_step = _discrete_levy_flight(rng_wrapper, beta, scale=alpha * 2)
                large_step = min(large_step, problem.dimension)
                
                nests[k] = nests[good_idx].copy()
                recolor_vertices = rng_wrapper.rng.choice(problem.dimension, size=large_step, replace=False)
                
                colors_used = set(int(c) for c in nests[k])
                max_color = max(colors_used) if colors_used else 0
                
                for vertex in recolor_vertices:
                    nests[k][vertex] = float(rng_wrapper.rng.integers(0, max_color + 2))
            
            fitness[k] = problem.evaluate(nests[k])

        best_fitness = np.min(fitness)
        best_idx = np.argmin(fitness)
        num_colors = len(np.unique(nests[best_idx].astype(int)))
        history.add([x.copy() for x in nests], list(fitness), f"gen={gen+1}, best={best_fitness:.4f}, colors={num_colors}")

    total_time = timer.stop()
    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Cuckoo Search (Graph Coloring)",
        problem=problem,
        time=total_time,
        last_x=nests,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history=history
    )


def cuckoo_search_discrete(
    problem: DiscreteProblem,
    population_size: int = 25,
    alpha: Float = 1.0,
    discovery_rate: Float = 0.25,
    beta: Float = 1.5,
    generation: int = 100,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Cuckoo Search (CS) for discrete optimization problems.
    
    This is a dispatcher function that automatically selects the appropriate
    CS variant based on the problem type:
    - TSPFunction -> cuckoo_search_discrete_tsp (discrete Lévy flights as swaps)
    - KnapsackFunction -> cuckoo_search_discrete_knapsack (set-based item flips)
    - GraphColoringFunction -> cuckoo_search_discrete_graphcoloring (vertex re-coloring)
    
    Cuckoo Search is inspired by the brood parasitism behavior of cuckoo birds.
    For discrete problems, Lévy flights are discretized into integer step sizes
    that determine the magnitude of changes to solutions.
    
    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of nests (solutions) in the population.
        alpha: Step size scaling factor for Lévy flights.
        discovery_rate: Fraction of worst nests to be abandoned (typically 0.25).
        beta: Exponent for the Lévy distribution (typically 1.5).
        generation: The number of generations to run the algorithm.
        rng_seed: Seed for the random number generator for reproducibility.
        
    Returns:
        DiscreteResult from the appropriate CS variant.
    """
    # Dispatch to specialized CS function based on problem type
    if isinstance(problem, TSPFunction):
        return cuckoo_search_discrete_tsp(
            problem=problem,
            population_size=population_size,
            alpha=alpha,
            discovery_rate=discovery_rate,
            beta=beta,
            generation=generation,
            rng_seed=rng_seed
        )
    elif isinstance(problem, KnapsackFunction):
        return cuckoo_search_discrete_knapsack(
            problem=problem,
            population_size=population_size,
            alpha=alpha,
            discovery_rate=discovery_rate,
            beta=beta,
            generation=generation,
            rng_seed=rng_seed
        )
    elif isinstance(problem, GraphColoringFunction):
        return cuckoo_search_discrete_graphcoloring(
            problem=problem,
            population_size=population_size,
            alpha=alpha,
            discovery_rate=discovery_rate,
            beta=beta,
            generation=generation,
            rng_seed=rng_seed
        )
    else:
        # Fallback to graph coloring approach for unknown discrete problems
        return cuckoo_search_discrete_graphcoloring(
            problem=problem,
            population_size=population_size,
            alpha=alpha,
            discovery_rate=discovery_rate,
            beta=beta,
            generation=generation,
            rng_seed=rng_seed
        )