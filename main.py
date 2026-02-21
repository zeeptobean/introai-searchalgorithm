import numpy as np
from util.define import *
from function.continuous_function import *
from algo.simulated_annealing import simulated_annealing_continuous
from typing import Callable

problem = RastriginFunction(dimension=2)
result = simulated_annealing_continuous(problem, rng_seed=42, step_bound=0.5, temp=55.0 )

for i in range(len(result.history_x)):
    print(f"Iteration {i}: x = {result.history_x[i]}, value = {result.history_value[i]}, info = {result.history_info[i]}")
print(f"Best solution: x = {result.best_x}, value = {result.best_value}")
print(f"RNG Seed: {result.rng_seed}")
