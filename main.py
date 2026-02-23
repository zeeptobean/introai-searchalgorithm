import json

import numpy as np
from util.define import *
import function.continuous_function as contfunc
from algo.simulated_annealing import *
from algo.differential_evolution import *
from algo.genetic import *
from swarmalgo.pso import *
from swarmalgo.cuckoo_search import *
from swarmalgo.tlbo import *
from swarmalgo.abc import *

# problem = contfunc.RastriginFunction(dimension=2)
problem = contfunc.MichalewiczFunction(dimension=2)
# result = simulated_annealing_continuous(problem, rng_seed=42, step_bound=0.5, temp=100.0, cooling_rate=0.99)
# result = differential_evolution_continuous(problem, rng_seed=42, population_size=60, generation=1500)
result = differential_evolution_continuous(problem, rng_seed=42, generation=10)

# print(result)
with open("result.dat", "wb") as f:
    f.write(result.to_binary())

with open("result.dat", "rb") as f:
    result = ContinuousResult.from_binary(f.read())

print(f"algo = {result.algorithm}, problem = {result.problem}")
print(f"best = {result.best_value} at x = {result.best_x} took {result.time:.4f} ms")