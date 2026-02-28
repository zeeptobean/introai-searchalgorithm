from util.define import *
from util.util import *
import math 
from util.result import ContinuousResult, DiscreteResult
from function.continuous_function import ContinuousProblem
from function.discrete_function import DiscreteProblem

# Mantegna's algorithm
def _levy_flight_step(rng_wrapper: RNGWrapper, dimension: int, beta: Float) -> FloatVector:
    sigma_u_num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    sigma_u_den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (sigma_u_num / sigma_u_den) ** (1 / beta)
    sigma_v = 1.0

    u = rng_wrapper.rng.normal(0, sigma_u, size=dimension)
    v = rng_wrapper.rng.normal(0, sigma_v, size=dimension)
    
    step = u / (np.abs(v) ** (1 / beta))
    return step

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

def cuckoo_search_discrete(
    problem: DiscreteProblem,
    population_size: int = 25,
    alpha: Float = 1.0,
    discovery_rate: Float = 0.25,
    step_size: Float = 1.0,
    generation: int = 100,
    rng_seed: int | None = None
) -> DiscreteResult:
    """
    Cuckoo Search (CS) for discrete optimization problems.
    
    Cuckoo Search is inspired by the brood parasitism behavior of cuckoo birds:
    - Cuckoos lay eggs in other birds' nests
    - If discovered, the host bird abandons the nest
    - Lévy flights are used for efficient exploration
    
    For discrete problems, we adapt CS by:
    - Using neighbor function instead of Lévy flights for position updates
    - Using probabilistic blending for non-permutation problems
    - Maintaining permutation validity for TSP-like problems

    Args:
        problem: The discrete optimization problem to solve.
        population_size: The number of nests (solutions) in the population.
        alpha: Step size scaling factor (controls exploration strength).
        discovery_rate: Fraction of worst nests to be abandoned (typically 0.25).
        step_size: Base step size for neighbor generation.
        generation: The number of generations to run the algorithm.
        rng_seed: Seed for the random number generator for reproducibility.
    """
    if not (0 < discovery_rate < 1):
        raise ValueError("discovery_rate must be in (0, 1)")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    rng_wrapper = RNGWrapper(rng_seed)

    timer = TimerWrapper()
    timer.start()

    nests: list[FloatVector] = [problem.random_solution(rng_wrapper) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in nests])

    is_permutation = (len(np.unique(nests[0])) == len(nests[0]) and 
                        np.allclose(np.sort(nests[0]), np.arange(len(nests[0]))))

    history = HistoryEntry(is_max_value_problem=problem.is_max_value_problem())

    history.add([x.copy() for x in nests], list(fitness))

    for gen in range(generation):
        for i in range(population_size):
            effective_step = max(1, int(step_size * alpha * (1 + rng_wrapper.random())))
            
            new_nest = problem.neighbor(nests[i], effective_step, rng_wrapper)
            new_fitness = problem.evaluate(new_nest)

            j = rng_wrapper.rng.integers(0, population_size)
            
            if new_fitness < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fitness

        sorted_indices = np.argsort(fitness)
        num_abandon = max(1, int(discovery_rate * population_size))
        worst_indices = sorted_indices[-num_abandon:]

        for k in worst_indices:
            if rng_wrapper.random() < 0.5:
                nests[k] = problem.random_solution(rng_wrapper)
            else:
                good_idx = sorted_indices[rng_wrapper.rng.integers(0, population_size // 2)]
                
                if is_permutation:
                    large_step = max(int(step_size * 2), problem.dimension // 4)
                    nests[k] = problem.neighbor(nests[good_idx], large_step, rng_wrapper)
                else:
                    random_sol = problem.random_solution(rng_wrapper)
                    blend_prob = 0.5
                    blend_mask = rng_wrapper.rng.random(problem.dimension) < blend_prob
                    nests[k] = np.where(blend_mask, random_sol, nests[good_idx])
            
            fitness[k] = problem.evaluate(nests[k])

        history.add([x.copy() for x in nests], list(fitness))

    total_time = timer.stop()

    best_x, best_value = history.get_best_value()

    return DiscreteResult(
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