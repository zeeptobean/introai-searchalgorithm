from util.define import *

def differential_evolution_continuous(
    problem: ContinuousProblem, 
    population_size: int = 20, 
    mutation_factor: Float = 0.8, 
    crossover_rate: Float = 0.7, 
    generation: int = 10000, 
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

    population: list[FloatVector] = [rng_wrapper.uniform(lower_bound, upper_bound, size=problem.dimension) for _ in range(population_size)]
    fitness = np.array([problem.evaluate(x) for x in population])

    history_x: list[FloatVector] = []
    history_value: list[Float] = []
    history_info: list[str] = []

    for ite in range(generation):
        for i in range(population_size):
            # Mutation
            idx_list = [idx for idx in range(population_size) if idx != i]
            a, b, c = population[rng_wrapper.rng.choice(idx_list, 3, replace=False)]
            mutant: FloatVector = np.clip(a + mutation_factor * (b - c), lower_bound, upper_bound)

            # Crossover
            crossover_mask = rng_wrapper.rng.random(problem.dimension) < crossover_rate
            trial = np.where(crossover_mask, mutant, population[i])
            trial_fitness = problem.evaluate(trial)

            # Selection
            if trial_fitness < fitness[i]:  # Minimizing fitness
                population[i] = trial
                fitness[i] = trial_fitness

    return ContinuousResult(
        algorithm="Differential Evolution",
        objective_function=repr(problem),
        last_x=population[np.argmin(fitness)],
        last_value=np.min(fitness),
        iterations=generation,
        rng_seed=rng_wrapper.get_seed(),
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )