from function.discrete_function import DiscreteProblem
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

    timer = TimerWrapper()
    timer.start()

    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in population])

    history_x: list[list[FloatVector]] = [[x.copy() for x in population]]
    history_value: list[list[Float]] = [list(fitness)]

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
        history_x.append([x.copy() for x in population])
        history_value.append(list(fitness))
    total_time = timer.stop()

    history_info: list[str | None] = [None] * len(history_x)
    best_x, best_value = get_min_2d(history_x, history_value)

    return ContinuousResult(
        algorithm="Differential Evolution",
        problem=problem,
        time=total_time,
        last_x=population,
        last_value=list(fitness),
        best_x=best_x,
        best_value=best_value,
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )


def differential_evolution_discrete(
    problem: DiscreteProblem, 
    population_size: int = 20, 
    mutation_factor: Float = 0.8, 
    crossover_rate: Float = 0.7, 
    generation: int = 100, 
    rng_seed: int | None = None
) -> DiscreteResult:
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
            # 1. Mutation rời rạc
            idx_list = [idx for idx in range(population_size) if idx != i]
            a_idx = rng_wrapper.rng.choice(idx_list)
            
            swap_cnt = int(rng_wrapper.rng.integers(1, max(2, int(problem.dimension * mutation_factor))))
            mutant = problem.neighbor(population[a_idx], swap_cnt, rng_wrapper)

            # 2. Crossover for permutation problems (Order Crossover - OX)
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

            # 3. Selection (Greedy)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
        history.add([x.copy() for x in population], list(fitness))

    total_time = timer.stop()

    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Differential Evolution",
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