"""
Test file for discrete optimization algorithms on TSP, Knapsack, and Graph Coloring problems.
Tests 9 algorithms: SA, DE, GA, PSO, ABC, TLBO, Firefly, ACO, Cuckoo Search
"""

import numpy as np
import time
from typing import List, Dict
import function.discrete_function as discfunc
from algo.simulated_annealing import simulated_annealing_discrete
from algo.differential_evolution import differential_evolution_discrete
from algo.genetic import genetic_algorithm_discrete
from swarmalgo.pso import pso_discrete
from swarmalgo.abc import abc_discrete
from swarmalgo.tlbo import tlbo_discrete
from swarmalgo.firefly import firefly_discrete
from swarmalgo.aco import aco_discrete
from swarmalgo.cuckoo_search import cuckoo_search_discrete
from util.result import DiscreteResult


# ==================== Problem Definitions ====================

def create_tsp_problem():
    """TSP problem with 5 cities"""
    distance_matrix = np.array([
        [0, 48, 65, 68, 68],
        [10, 0, 22, 37, 88],
        [71, 89, 0, 13, 59],
        [66, 40, 88, 0, 89],
        [82, 38, 26, 78, 0]
    ])
    return discfunc.TSPFunction(distance_matrix=distance_matrix, dimension=5)


def create_graphcoloring_problem():
    """Graph Coloring problem with 20 nodes"""
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
    return discfunc.GraphColoringFunction(adjacency_matrix=adjacency_matrix, dimension=20)


def create_knapsack_problem():
    """Knapsack problem with 5 items"""
    return discfunc.KnapsackFunction(
        weights=np.array([10, 20, 30, 40, 50]),
        values=np.array([60, 100, 120, 240, 300]),
        capacity=100,
        dimension=5
    )


# ==================== Test Functions ====================

def test_algorithm(algo_name: str, algo_func, problem, **kwargs) -> Dict:
    """
    Test a single algorithm on a problem.
    
    Args:
        algo_name: Name of the algorithm
        algo_func: Algorithm function to call
        problem: Problem instance
        **kwargs: Additional parameters for the algorithm
    
    Returns:
        Dictionary with test results
    """
    print(f"  Testing {algo_name}...", end=" ")
    
    try:
        start_time = time.time()
        result: DiscreteResult = algo_func(problem, **kwargs)
        wall_time = time.time() - start_time
        
        # Extract results
        best_value = result.best_value
        iterations = result.iterations
        algo_time = result.time
        
        print(f"✓ Best: {best_value:.4f}, Iter: {iterations}, Time: {wall_time:.3f}s")
        
        return {
            "algorithm": algo_name,
            "best_value": best_value,
            "iterations": iterations,
            "algo_time_ms": algo_time,
            "wall_time_s": wall_time,
            "success": True,
            "error": None
        }
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return {
            "algorithm": algo_name,
            "best_value": None,
            "iterations": None,
            "algo_time_ms": None,
            "wall_time_s": None,
            "success": False,
            "error": str(e)
        }


def run_all_tests(problem, problem_name: str, rng_seed: int = 42) -> List[Dict]:
    """
    Run all 9 algorithms on a given problem.
    
    Args:
        problem: Problem instance
        problem_name: Name of the problem
        rng_seed: Random seed for reproducibility
    
    Returns:
        List of test results
    """
    print(f"\n{'='*80}")
    print(f"Testing on {problem_name}")
    print(f"{'='*80}")
    
    results = []
    
    # Common parameters
    common_params = {
        "rng_seed": rng_seed,
        "generation": 100,
        "population_size": 50
    }
    
    # 1. Simulated Annealing
    results.append(test_algorithm(
        "Simulated Annealing",
        simulated_annealing_discrete,
        problem,
        temp=100.0,
        cooling_rate=0.95,
        min_temp=0.001,
        max_iteration=5000,
        rng_seed=rng_seed
    ))
    
    # 2. Differential Evolution
    results.append(test_algorithm(
        "Differential Evolution",
        differential_evolution_discrete,
        problem,
        population_size=50,
        generation=100,
        mutation_factor=0.8,
        crossover_rate=0.7,
        rng_seed=rng_seed
    ))
    
    # 3. Genetic Algorithm
    results.append(test_algorithm(
        "Genetic Algorithm",
        genetic_algorithm_discrete,
        problem,
        population_size=50,
        generation=100,
        tournament_k=3,
        crossover_rate=0.9,
        mutation_rate=0.1,
        rng_seed=rng_seed
    ))
    
    # 4. Particle Swarm Optimization
    results.append(test_algorithm(
        "PSO",
        pso_discrete,
        problem,
        population_size=50,
        generation=100,
        inertia_weight=0.7,
        cognitive_coeff=1.5,
        social_coeff=1.5,
        rng_seed=rng_seed
    ))
    
    # 5. Artificial Bee Colony
    results.append(test_algorithm(
        "ABC",
        abc_discrete,
        problem,
        population_size=50,
        generation=100,
        limit=10,
        rng_seed=rng_seed
    ))
    
    # 6. Teaching-Learning-Based Optimization
    results.append(test_algorithm(
        "TLBO",
        tlbo_discrete,
        problem,
        population_size=50,
        generation=100,
        rng_seed=rng_seed
    ))
    
    # 7. Firefly Algorithm
    results.append(test_algorithm(
        "Firefly",
        firefly_discrete,
        problem,
        population_size=50,
        generation=100,
        alpha=0.5,
        beta0=1.0,
        gamma=1.0,
        rng_seed=rng_seed
    ))
    
    # 8. Ant Colony Optimization
    results.append(test_algorithm(
        "ACO",
        aco_discrete,
        problem,
        num_ants=50,
        generation=100,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5,
        rng_seed=rng_seed
    ))
    
    # 9. Cuckoo Search
    results.append(test_algorithm(
        "Cuckoo Search",
        cuckoo_search_discrete,
        problem,
        population_size=50,
        generation=100,
        discovery_rate=0.25,
        rng_seed=rng_seed
    ))
    
    return results


def print_summary(all_results: Dict[str, List[Dict]]):
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Algorithm':<25} {'TSP':<15} {'Knapsack':<15} {'GraphColor':<15}")
    print(f"{'-'*80}")
    
    # Collect algorithms
    algorithms = []
    if all_results.get("TSP"):
        algorithms = [r["algorithm"] for r in all_results["TSP"]]
    
    # Print each algorithm's results
    for algo in algorithms:
        row = f"{algo:<25}"
        
        for problem_name in ["TSP", "Knapsack", "GraphColoring"]:
            if problem_name in all_results:
                result = next((r for r in all_results[problem_name] if r["algorithm"] == algo), None)
                if result and result["success"]:
                    row += f" {result['best_value']:>13.4f} "
                else:
                    row += f" {'ERROR':>13} "
            else:
                row += f" {'-':>13} "
        
        print(row)
    
    print(f"{'='*80}\n")


# ==================== Main Test Runner ====================

def main():
    """Main test function."""
    print("\n" + "="*80)
    print(" DISCRETE OPTIMIZATION ALGORITHMS TEST SUITE")
    print("="*80)
    print("\nTesting 9 algorithms on 3 problems:")
    print("  1. Simulated Annealing (SA)")
    print("  2. Differential Evolution (DE)")
    print("  3. Genetic Algorithm (GA)")
    print("  4. Particle Swarm Optimization (PSO)")
    print("  5. Artificial Bee Colony (ABC)")
    print("  6. Teaching-Learning-Based Optimization (TLBO)")
    print("  7. Firefly Algorithm")
    print("  8. Ant Colony Optimization (ACO)")
    print("  9. Cuckoo Search")
    print("\nProblems:")
    print("  - TSP (5 cities)")
    print("  - Knapsack (5 items, capacity=100)")
    print("  - Graph Coloring (20 nodes)")
    
    # Create problems
    tsp_problem = create_tsp_problem()
    knapsack_problem = create_knapsack_problem()
    graphcoloring_problem = create_graphcoloring_problem()
    
    # Run tests
    all_results = {}
    
    # Test TSP
    all_results["TSP"] = run_all_tests(tsp_problem, "TSP (5 cities)", rng_seed=42)
    
    # Test Knapsack
    all_results["Knapsack"] = run_all_tests(knapsack_problem, "Knapsack (5 items)", rng_seed=42)
    
    # Test Graph Coloring
    all_results["GraphColoring"] = run_all_tests(graphcoloring_problem, "Graph Coloring (20 nodes)", rng_seed=42)
    
    # Print summary
    print_summary(all_results)
    
    # Print best results for each problem
    print("\n" + "="*80)
    print("BEST RESULTS")
    print("="*80)
    
    for problem_name, results in all_results.items():
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            # For minimization problems (TSP, GraphColoring) or maximization (Knapsack)
            # All are stored as minimization internally, so find min value
            best = min(successful_results, key=lambda x: x["best_value"])
            problem_type = "maximization" if problem_name == "Knapsack" else "minimization"
            print(f"\n{problem_name}: Best = {best['best_value']:.4f} ({problem_type} problem)")
            
            print(f"  Algorithm: {best['algorithm']}")
            print(f"  Iterations: {best['iterations']}")
            print(f"  Time: {best['wall_time_s']:.3f}s")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
