from util.define import *

def simulated_annealing_continuous(
    problem: ContinuousProblem, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    step_bound: Float = 0.1, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> Result:
    if rng_seed is None:
        rng_seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    current_x = rng.uniform(lower_bound, upper_bound, size=problem.dimension)
    current_energy = problem.evaluate(current_x)

    history_x: list[FloatVector] = [current_x]
    history_value: list[Float] = [current_energy]
    history_info: list[str] = [f"temp: {temp:.4f}, delta: None"]

    iteration = 1
    while temp > min_temp and iteration <= max_iteration:
        noise = rng.uniform(-step_bound, step_bound, size=problem.dimension)
        next_x = np.clip(current_x + noise, lower_bound, upper_bound)
        next_energy = problem.objective_function(next_x)
        delta = current_energy - next_energy #minimizing energy => delta > 0 is better 

        history_x.append(next_x)
        history_value.append(next_energy)
        history_info.append(f"temp: {temp:.4f}, delta: {delta:.4f}")

        if delta > 0 or rng.random() < np.exp(delta / temp):
            current_x = next_x
            current_energy = next_energy

        temp *= cooling_rate
        iteration += 1

    return Result(
        type="continuous",
        algorithm="Simulated_Annealing (geometric cooling)",
        objective_function=repr(problem),
        best_x=current_x,
        best_value=current_energy,
        iterations=iteration,
        rng_seed=rng_seed,
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )