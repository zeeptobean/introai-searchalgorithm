# IntroAI course project: Search & Nature-inspried algorithms

Testing and comparing different heuristic-based and swarms intelligence algorithm across different target functions:
- Continuous functions: Rastrigin, Rosenbrock, Sphere, Michalewicz, Styblinski-Tang, Griewank, Ackley
- Discrete optimization: TSP, Knapsack, Graph coloring
- Pathfinding problems

## Members
#### Group name: Lorem_Ipsum
- 24127003 - Vũ Trần Minh Hiếu
- 24127240 - Hoàng Đức Thịnh
- 24127270 - Trần Viết Bảo
- 24127326 - Đoàn Quốc Bảo

## Project Layout

- Core algorithms:
  - Differential Evolution (DE): [algo/differential_evolution.py](algo/differential_evolution.py)
  - Genetic algorihm (GA): [algo/genetic.py](algo/genetic.py)
  - Simulated Annealing (SA): [algo/simulated_annealing.py](algo/simulated_annealing.py)
  - BFS, DFS, A* for pathfinding: [algo/graph_search.py](algo/graph_search.py)
- Swarm algorithms:
  - Artificial Bee Colony (ABC): [swarmalgo/abc.py](swarmalgo/abc.py)
  - Ant colony optimization (ACO): [swarmalgo/aco.py](swarmalgo/aco.py)
  - Cuckoo Search (CS): [swarmalgo/cuckoo_search.py](swarmalgo/cuckoo_search.py)
  - Firefly algorihm (FA): [swarmalgo/firefly.py](swarmalgo/firefly.py)
  - Particle Swarms Optimization (PSO): [swarmalgo/pso.py](swarmalgo/pso.py)
  - Teaching-learning Based Optimization (TLBO): [swarmalgo/tlbo.py](swarmalgo/tlbo.py)
- Problem/function definitions:
  - [function/continuous_function.py](function/continuous_function.py)
  - [function/discrete_function.py](function/discrete_function.py)
  - [function/graph_problem.py](function/graph_problem.py)
- Visualization:
  - [visual.py](visual.py)
- Experiment notebooks:
  - Continuous problems: [continuous_test.ipynb](continuous_test.ipynb)
  - Discrete & pathfinding problems [discrete_and_path_test.ipynb](discrete_and_path_test.ipynb)

## Setup
0. Require Python 3.13 or newer installed
1. Create and activate a Python virtual environment.
2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run experiments with notebooks
- [continuous_test.ipynb](continuous_test.ipynb)
- [discrete_and_path_test.ipynb](discrete_and_path_test.ipynb)

