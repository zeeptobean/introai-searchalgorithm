from function.discrete_function import DiscreteProblem
from util.define import *
from util.util import *
from util.result import ContinuousResult
from util.result import DiscreteResult
from function.continuous_function import ContinuousProblem

def simulated_annealing_continuous(
    problem: ContinuousProblem, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    step_bound: Float = 0.1, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> ContinuousResult:
    rng = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    timer = TimerWrapper()
    timer.start()

    current_x = rng.uniform(lower_bound, upper_bound, size=problem.dimension)
    current_energy = problem.evaluate(current_x)

    history_x: list[list[FloatVector]] = [[current_x]]
    history_value: list[list[Float]] = [[current_energy]]
    history_info: list[str | None] = [f"temp: {temp:.4f}, delta: None"]

    iteration = 1
    while temp > min_temp and iteration <= max_iteration:
        noise = rng.uniform(-step_bound, step_bound, size=problem.dimension)
        next_x = np.clip(current_x + noise, lower_bound, upper_bound)
        next_energy = problem.objective_function(next_x)
        delta = current_energy - next_energy #minimizing energy => delta > 0 is better 

        history_x.append([next_x])
        history_value.append([next_energy])

        if delta > 0 or rng.random() < np.exp(delta / temp):
            history_info.append(f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)")
            current_x = next_x
            current_energy = next_energy
        else:
            history_info.append(f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)")

        temp *= cooling_rate
        iteration += 1

    total_time = timer.stop()

    best_x, best_value = get_min_2d(history_x, history_value)

    return ContinuousResult(
        algorithm="Simulated Annealing (geometric cooling)",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy],
        best_x=best_x,
        best_value=best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )
"""
Temperature is scaled linearly with iteration count
"""
def simulated_annealing_linear_continuous(
    problem: ContinuousProblem, 
    max_temp: Float = 10.0, 
    step_bound: Float = 0.1, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> ContinuousResult:
    rng = RNGWrapper(rng_seed)
    lower_bound = problem.lower_bound if problem.lower_bound is not None else -100
    upper_bound = problem.upper_bound if problem.upper_bound is not None else 100

    timer = TimerWrapper()
    timer.start()

    current_x = rng.uniform(lower_bound, upper_bound, size=problem.dimension)
    current_energy = problem.evaluate(current_x)

    history_x: list[list[FloatVector]] = [[current_x]]
    history_value: list[list[Float]] = [[current_energy]]
    history_info: list[str | None] = [f"temp: {max_temp:.4f}, delta: None"]

    iteration = 0
    while iteration <= max_iteration:
        temp = max_temp * (1 - iteration/ max_iteration)
        temp = max(temp, 1e-12)     # Avoid divide by 0
        noise = rng.uniform(-step_bound, step_bound, size=problem.dimension)
        next_x = np.clip(current_x + noise, lower_bound, upper_bound)
        next_energy = problem.objective_function(next_x)
        delta = current_energy - next_energy

        history_x.append([next_x])
        history_value.append([next_energy])

        if delta > 0 or rng.random() < np.exp(delta / temp):
            history_info.append(f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)")
            current_x = next_x
            current_energy = next_energy
        else:
            history_info.append(f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)")

        iteration += 1
    total_time = timer.stop()

    best_x, best_value = get_min_2d(history_x, history_value)

    return ContinuousResult(
        algorithm="Simulated Annealing (linear cooling)",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy],
        best_x=best_x,
        best_value=best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history_x=history_x,
        history_value=history_value,
        history_info=history_info
    )

# discrete
def simulated_annealing_discrete(
    problem: DiscreteProblem, 
    temp: Float = 10.0, 
    cooling_rate: Float = 0.95, 
    min_temp: Float = 0.001, 
    step_bound: int = 2, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> DiscreteResult:
    rng = RNGWrapper(rng_seed)

    timer = TimerWrapper()
    timer.start()

    current_x = problem.random_solution(rng)
    current_energy = problem.evaluate(current_x)

    history = HistoryEntry(problem.is_max_value_problem())
    history.add([current_x], [current_energy], f"temp: {temp:.4f}, delta: None")

    iteration = 1
    while temp > min_temp and iteration <= max_iteration:
        noise = int(rng.rng.integers(1, step_bound, endpoint=True))
        next_x = problem.neighbor(current_x, noise, rng)
        next_energy = problem.evaluate(next_x)

        delta = current_energy - next_energy #minimizing energy => delta > 0 is better 

        if delta > 0 or rng.random() < np.exp(delta / temp):
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)"
            current_x = next_x
            current_energy = next_energy
        else:
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)"
        history.add([next_x], [next_energy], infostr)

        temp *= cooling_rate
        iteration += 1

    total_time = timer.stop()

    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Simulated Annealing (geometric cooling)",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy],
        best_x=best_x,
        best_value=best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history=history
    )

"""
Temperature is scaled linearly with iteration count
"""
def simulated_annealing_linear_discrete(
    problem: DiscreteProblem, 
    max_temp: Float = 10.0, 
    step_bound: int = 2, 
    max_iteration: int = 10000, 
    rng_seed: int | None = None
) -> DiscreteResult:
    rng = RNGWrapper(rng_seed)
    timer = TimerWrapper()
    timer.start()

    current_x = problem.random_solution(rng)
    current_energy = problem.evaluate(current_x)

    history = HistoryEntry(problem.is_max_value_problem())
    history.add([current_x], [current_energy], f"temp: {max_temp:.4f}, delta: None")

    iteration = 0
    while iteration <= max_iteration:
        temp = max_temp * (1 - iteration/ max_iteration)
        temp = max(temp, 1e-12)     # Avoid divide by 0
        noise = int(rng.rng.integers(1, step_bound, endpoint=True))
        next_x = problem.neighbor(current_x, noise, rng)
        next_energy = problem.evaluate(next_x)
        delta = current_energy - next_energy

        if delta > 0 or rng.random() < np.exp(delta / temp):
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (accepted)"
            current_x = next_x
            current_energy = next_energy
        else:
            infostr = f"temp: {temp:.4f}, delta: {delta:.4f} (rejected)"
        history.add([next_x], [next_energy], infostr)

        iteration += 1
    total_time = timer.stop()

    best_x, best_value = history.get_best_value()

    return DiscreteResult(
        algorithm="Simulated Annealing (linear cooling)",
        problem=problem,
        last_x=[current_x],
        last_value=[current_energy if problem.is_max_value_problem() else -current_energy],
        best_x=best_x,
        best_value=best_value if problem.is_max_value_problem() else -best_value,
        time=total_time,
        iterations=iteration,
        rng_seed=rng.get_seed(),
        history=history
    )