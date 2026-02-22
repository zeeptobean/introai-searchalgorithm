import numpy as np
from util.define import *
import function.continuous_function as contfunc
from algo.simulated_annealing import *
from algo.differential_evolution import *
from swarmalgo.cuckoo_search import *
from swarmalgo.tlbo import *
from typing import Callable

# problem = contfunc.RastriginFunction(dimension=1024)
problem = contfunc.MichalewiczFunction(dimension=10)
# result = simulated_annealing_continuous(problem, rng_seed=42, step_bound=0.5, temp=100.0, cooling_rate=0.99)
# result = differential_evolution_continuous(problem, rng_seed=42, population_size=60, generation=1500)
result = tlbo_continuous(problem, rng_seed=42, population_size=265, generation=1000)

print(f"best = {result.best_value} at x = {result.best_x} took {result.time:.4f} ms")