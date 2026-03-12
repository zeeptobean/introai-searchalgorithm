"""
Graph Search Algorithms for GridWorld Problems
Implements BFS, DFS, and A* for pathfinding in grid-based environments.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from typing import List, Tuple, Set, Optional, Dict
from collections import deque
import heapq
import time
import numpy as np
from function.graph_problem import GridWorldProblem
from util.result import SearchResult
from util.util import HistoryEntry


def bfs(problem: GridWorldProblem) -> SearchResult:
    """
    Breadth-First Search (BFS) algorithm.
    Guarantees shortest path in terms of number of steps.
    
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V) for the queue and visited set
    
    Args:
        problem: GridWorldProblem instance
    
    Returns:
        SearchResult with path, cost, and statistics
    """
    start_time = time.time()
    history = HistoryEntry(is_max_value_problem=False)
    
    # Initialize queue with start position
    queue = deque([(problem.start, [problem.start])])
    visited: Set[Tuple[int, int]] = {problem.start}
    nodes_expanded = 0
    
    while queue:
        current_pos, path = queue.popleft()
        nodes_expanded += 1
        
        # Add to history (track current path being explored)
        path_cost = problem.evaluate_path(path) if len(path) > 1 else 0.0
        history.add(
            x=[problem.path_to_vector(path)],
            value=[path_cost],
            info=f"Exploring node {current_pos}, queue_size={len(queue)}"
        )
        
        # Check if goal is reached
        if problem.is_goal(current_pos):
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            cost = problem.evaluate_path(path)
            
            return SearchResult(
                type="pathfinding",
                algorithm="BFS",
                iterations=nodes_expanded,
                rng_seed=0,
                time=elapsed_time,
                last_x=[],  # Will be set in __post_init__
                last_value=[],  # Will be set in __post_init__
                best_x=np.array([]),  # Will be set in __post_init__
                best_value=cost,
                history=history,
                problem=problem,
                path=path
            )
        
        # Explore neighbors
        for neighbor in problem.get_neighbors(current_pos):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    
    # No path found
    elapsed_time = (time.time() - start_time) * 1000
    return SearchResult(
        type="pathfinding",
        algorithm="BFS",
        iterations=nodes_expanded,
        rng_seed=0,
        time=elapsed_time,
        last_x=[],
        last_value=[],
        best_x=np.array([]),
        best_value=float('inf'),
        history=history,
        problem=problem,
        path=None
    )


def dfs(problem: GridWorldProblem, max_depth: Optional[int] = None) -> SearchResult:
    """
    Depth-First Search (DFS) algorithm.
    Does NOT guarantee shortest path, but uses less memory than BFS.
    
    Time Complexity: O(V + E) in worst case
    Space Complexity: O(h) where h is maximum depth
    
    Args:
        problem: GridWorldProblem instance
        max_depth: Optional maximum depth limit to prevent infinite loops
    
    Returns:
        SearchResult with path, cost, and statistics
    """
    start_time = time.time()
    history = HistoryEntry(is_max_value_problem=False)
    
    # Initialize stack with start position
    stack = [(problem.start, [problem.start], 0)]  # (position, path, depth)
    visited: Set[Tuple[int, int]] = set()
    nodes_expanded = 0
    
    # Set default max depth if not provided
    if max_depth is None:
        max_depth = problem.rows * problem.cols
    
    while stack:
        current_pos, path, depth = stack.pop()
        
        # Skip if already visited
        if current_pos in visited:
            continue
        
        visited.add(current_pos)
        nodes_expanded += 1
        
        # Add to history
        path_cost = problem.evaluate_path(path) if len(path) > 1 else 0.0
        history.add(
            x=[problem.path_to_vector(path)],
            value=[path_cost],
            info=f"Exploring node {current_pos}, depth={depth}, stack_size={len(stack)}"
        )
        
        # Check if goal is reached
        if problem.is_goal(current_pos):
            elapsed_time = (time.time() - start_time) * 1000
            cost = problem.evaluate_path(path)
            
            return SearchResult(
                type="pathfinding",
                algorithm="DFS",
                iterations=nodes_expanded,
                rng_seed=0,
                time=elapsed_time,
                last_x=[],
                last_value=[],
                best_x=np.array([]),
                best_value=cost,
                history=history,
                problem=problem,
                path=path
            )
        
        # Check depth limit
        if depth >= max_depth:
            continue
        
        # Explore neighbors (in reverse to maintain left-to-right priority)
        neighbors = problem.get_neighbors(current_pos)
        for neighbor in reversed(neighbors):
            if neighbor not in visited:
                new_path = path + [neighbor]
                stack.append((neighbor, new_path, depth + 1))
    
    # No path found
    elapsed_time = (time.time() - start_time) * 1000
    return SearchResult(
        type="pathfinding",
        algorithm="DFS",
        iterations=nodes_expanded,
        rng_seed=0,
        time=elapsed_time,
        last_x=[],
        last_value=[],
        best_x=np.array([]),
        best_value=float('inf'),
        history=history,
        problem=problem,
        path=None
    )


def astar(problem: GridWorldProblem, heuristic_name: str = "manhattan") -> SearchResult:
    """
    A* Search algorithm.
    Guarantees optimal path by combining actual cost g(n) and heuristic h(n).
    f(n) = g(n) + h(n)
    
    Time Complexity: O(b^d) where b is branching factor, d is depth
    Space Complexity: O(b^d) for the priority queue
    
    Args:
        problem: GridWorldProblem instance
        heuristic_name: "manhattan" or "euclidean" distance heuristic
    
    Returns:
        SearchResult with path, cost, and statistics
    """
    start_time = time.time()
    history = HistoryEntry(is_max_value_problem=False)
    # info = (g_score, h_score, f_score, queue_size) will be tracked in history for debugging
    
    # Priority queue: (f_score, counter, position, path, g_score)
    # counter ensures FIFO order for equal f_scores
    counter = 0
    start_h = problem.heuristic(problem.start, heuristic_name)
    priority_queue = [(start_h, counter, problem.start, [problem.start], 0.0)]
    
    # Track best g_score for each position
    g_scores: Dict[Tuple[int, int], float] = {problem.start: 0.0}
    nodes_expanded = 0
    
    while priority_queue:
        f_score, _, current_pos, path, g_score = heapq.heappop(priority_queue)
        nodes_expanded += 1
        
        # Add to history
        path_cost = problem.evaluate_path(path) if len(path) > 1 else 0.0
        h_score = problem.heuristic(current_pos, heuristic_name)
        history.add(
            x=[problem.path_to_vector(path)],
            value=[path_cost],
            info=f"Node {current_pos}, g={g_score:.2f}, h={h_score:.2f}, f={f_score:.2f}, queue_size={len(priority_queue)}"
        )
        
        # Check if goal is reached
        if problem.is_goal(current_pos):
            elapsed_time = (time.time() - start_time) * 1000
            cost = problem.evaluate_path(path)
            
            return SearchResult(
                type="pathfinding",
                algorithm=f"A* ({heuristic_name})",
                iterations=nodes_expanded,
                rng_seed=0,
                time=elapsed_time,
                last_x=[],
                last_value=[],
                best_x=np.array([]),
                best_value=cost,
                history=history,
                problem=problem,
                path=path
            )
        
        # Skip if we've found a better path to this position
        if current_pos in g_scores and g_score > g_scores[current_pos]:
            continue
        
        # Explore neighbors
        for neighbor in problem.get_neighbors(current_pos):
            # Calculate new g_score (actual cost from start to neighbor)
            step_cost = problem.get_path_cost(current_pos, neighbor)
            new_g_score = g_score + step_cost
            
            # Only proceed if this is a better path
            if neighbor not in g_scores or new_g_score < g_scores[neighbor]:
                g_scores[neighbor] = new_g_score
                
                # Calculate h_score (heuristic estimate to goal)
                h_score = problem.heuristic(neighbor, heuristic_name)
                
                # f_score = g_score + h_score
                new_f_score = new_g_score + h_score
                
                new_path = path + [neighbor]
                counter += 1
                heapq.heappush(priority_queue, 
                              (new_f_score, counter, neighbor, new_path, new_g_score))
    
    # No path found
    elapsed_time = (time.time() - start_time) * 1000
    return SearchResult(
        type="pathfinding",
        algorithm=f"A* ({heuristic_name})",
        iterations=nodes_expanded,
        rng_seed=0,
        time=elapsed_time,
        last_x=[],
        last_value=[],
        best_x=np.array([]),
        best_value=float('inf'),
        history=history,
        problem=problem,
        path=None
    )


# Example usage and testing
if __name__ == "__main__":
    from function.graph_problem import create_simple_grid
    
    # Create a test grid
    print("Creating test grid...")
    grid, start, goal = create_simple_grid(rows=10, cols=10, obstacle_ratio=0.2, seed=42)
    problem = GridWorldProblem(grid, start, goal)
    
    print(f"\nProblem: {problem}")
    print("\nInitial Grid:")
    print(problem.visualize_path())
    print()
    
    # Test BFS
    print("=" * 60)
    print("Running BFS...")
    bfs_result = bfs(problem)
    print(bfs_result)
    if bfs_result.success:
        print("\nBFS Path Visualization:")
        print(bfs_result.visualize_path())
    print()
    
    # Test DFS
    print("=" * 60)
    print("Running DFS...")
    dfs_result = dfs(problem)
    print(dfs_result)
    if dfs_result.success:
        print("\nDFS Path Visualization:")
        print(dfs_result.visualize_path())
    print()
    
    # Test A* with Manhattan distance
    print("=" * 60)
    print("Running A* (Manhattan)...")
    astar_manhattan_result = astar(problem, heuristic_name="manhattan")
    print(astar_manhattan_result)
    if astar_manhattan_result.success:
        print("\nA* Path Visualization (Manhattan):")
        print(astar_manhattan_result.visualize_path())
    print()
    
    # Test A* with Euclidean distance
    print("=" * 60)
    print("Running A* (Euclidean)...")
    astar_euclidean_result = astar(problem, heuristic_name="euclidean")
    print(astar_euclidean_result)
    if astar_euclidean_result.success:
        print("\nA* Path Visualization (Euclidean):")
        print(astar_euclidean_result.visualize_path())
    print()
    
    # Compare algorithms
    print("=" * 60)
    print("ALGORITHM COMPARISON:")
    print(f"{'Algorithm':<20} {'Cost':>10} {'Nodes':>8} {'Time (ms)':>12} {'Success':>10}")
    print("-" * 60)
    print(f"{'BFS':<20} {bfs_result.path_cost:>10.2f} {bfs_result.nodes_explored:>8} {bfs_result.time:>12.2f} {str(bfs_result.success):>10}")
    print(f"{'DFS':<20} {dfs_result.path_cost:>10.2f} {dfs_result.nodes_explored:>8} {dfs_result.time:>12.2f} {str(dfs_result.success):>10}")
    print(f"{'A* (Manhattan)':<20} {astar_manhattan_result.path_cost:>10.2f} {astar_manhattan_result.nodes_explored:>8} {astar_manhattan_result.time:>12.2f} {str(astar_manhattan_result.success):>10}")
    print(f"{'A* (Euclidean)':<20} {astar_euclidean_result.path_cost:>10.2f} {astar_euclidean_result.nodes_explored:>8} {astar_euclidean_result.time:>12.2f} {str(astar_euclidean_result.success):>10}")
    print()
    
    # Test JSON serialization
    print("=" * 60)
    print("Testing JSON serialization...")
    json_data = bfs_result.to_json()
    print(f"JSON keys: {list(json_data.keys())}")
    
    # Test simplified JSON
    json_simple = bfs_result.to_json_simple()
    print(f"Simplified JSON keys: {list(json_simple.keys())}")
    print(f"Serialization test: PASSED")
