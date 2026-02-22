import numpy as np
from util.define import *
from function.continuous_function import *
from algo.simulated_annealing import *
from typing import Callable

problem = RastriginFunction(dimension=2)
result = simulated_annealing_continuous(problem, rng_seed=42, step_bound=0.5, temp=100.0, cooling_rate=0.99)

print(result)