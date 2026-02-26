import json

import numpy as np
from util.define import *
import function.continuous_function as contfunc
import function.discrete_function as discfunc
from algo.simulated_annealing import *
from algo.differential_evolution import *
from algo.genetic import *
from swarmalgo.pso import *
from swarmalgo.cuckoo_search import *
from swarmalgo.tlbo import *
from swarmalgo.abc import *

# problem = contfunc.RastriginFunction(dimension=2)
# problem = contfunc.MichalewiczFunction(dimension=2)
# distance_matrix=np.array([[ 0, 48, 65, 68, 68],
# [10,  0, 22, 37, 88],
# [71, 89,  0, 13, 59],
# [66, 40, 88,  0, 89],
# [82, 38, 26, 78,  0]])
# problem = discfunc.TSPFunction(distance_matrix=distance_matrix)
# print(distance_matrix)
# problem=discfunc.KnapsackFunction(
#     weights=np.array([10, 20, 30, 40, 50]),
#     values=np.array([60, 100, 120, 240, 300]),
#     capacity=100
# )
adjacency_matrix = np.array([
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
])
problem = discfunc.GraphColoringFunction(
    adjacency_matrix=adjacency_matrix
)
# result = simulated_annealing_continuous(problem, rng_seed=42, step_bound=0.5, temp=100.0, cooling_rate=0.99)
# result = differential_evolution_continuous(problem, rng_seed=42, population_size=60, generation=1500)
# result = differential_evolution_continuous(problem, rng_seed=42, generation=10)
result = differential_evolution_discrete(problem, rng_seed=42, population_size=20, generation=50, mutation_factor=1, crossover_rate=0.7)
# result = simulated_annealing_discrete(problem, rng_seed=42, temp=100.0, cooling_rate=0.99, step_bound=1.0, max_iteration=10000)
# result = genetic_algorithm_discrete(
#     problem, 
#     rng_seed=42, 
#     population_size=50, 
#     generation=100, 
#     tournament_k=3,
#     crossover_rate=0.9,
#     mutation_rate=0.2
# )

# print(result)
# with open("result.dat", "wb") as f:
#     f.write(result.to_binary())

# with open("result.dat", "rb") as f:
#     result = ContinuousResult.from_binary(f.read())

print(f"algo = {result.algorithm}, problem = {result.problem}")
print(f"best = {result.best_value} at x = {result.best_x} took {result.time:.4f} ms")