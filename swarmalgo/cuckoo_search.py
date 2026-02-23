from util.define import *
from util.util import *
import math 
from util.result import ContinuousResult
from function.continuous_function import ContinuousProblem

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

    timer = TimerWrapper()
    timer.start()

    nests: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in nests])

    history_x: list[list[FloatVector]] = [[x.copy() for x in nests]]
    history_value: list[list[Float]] = [list(fitness)]

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

        history_x.append([x.copy() for x in nests])
        history_value.append(list(fitness))

    total_time = timer.stop()

    history_info: list[str | None] = [None] * len(history_x)
    best_x, best_value = get_min_2d(history_x, history_value)

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
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )