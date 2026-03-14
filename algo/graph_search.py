from typing import Tuple, Set, Optional, Dict
from collections import deque
import heapq
import time
import numpy as np
from function.graph_problem import GridWorldProblem
from util.result import SearchResult
from util.util import HistoryEntry
from util.define import TimerWrapper


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
    timer = TimerWrapper()
    history = HistoryEntry(is_max_value_problem=False)
    
    # Initialize queue with start position
    queue = deque([(problem.start, [problem.start])])
    visited: Set[Tuple[int, int]] = {problem.start}
    nodes_expanded = 0
    
    timer.start()
    while queue:
        current_pos, path = queue.popleft()
        nodes_expanded += 1
        
        # Add to history (track current path being explored)
        path_cost = problem.evaluate_path(path)
        history.add(
            x=[problem.path_to_vector(path)],
            value=[path_cost],
            info=f"Exploring node {current_pos}, queue_size={len(queue)}"
        )
        
        # Check if goal is reached
        if problem.is_goal(current_pos):
            elapsed_time = timer.stop()
            cost = problem.evaluate_path(path)
            
            return SearchResult(
                algorithm="Breadth-First Search",
                short_name="BFS",
                nodes_expanded=nodes_expanded,
                time=elapsed_time,
                cost=cost,
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
    elapsed_time = timer.stop()
    return SearchResult(
        algorithm="Breadth-First Search",
        short_name="BFS",
        nodes_expanded=nodes_expanded,
        time=elapsed_time,
        cost=float('inf'),
        problem=problem,
        path=[]
    )


def dfs(problem: GridWorldProblem, input_max_depth: Optional[int] = None) -> SearchResult:
    """
    Depth-First Search (DFS) algorithm.
    Does NOT guarantee shortest path, but uses less memory than BFS.
    
    Time Complexity: O(V + E) in worst case
    Space Complexity: O(h) where h is maximum depth
    
    Args:
        problem: GridWorldProblem instance
        input_max_depth: Optional maximum depth limit to prevent infinite loops
    
    Returns:
        SearchResult with path, cost, and statistics
    """
    timer = TimerWrapper()
    timer.start()
    history = HistoryEntry(is_max_value_problem=False)
    
    # Initialize stack with start position
    stack = [(problem.start, [problem.start], 0)]  # (position, path, depth)
    visited: Set[Tuple[int, int]] = set()
    nodes_expanded = 0
    
    # Set default max depth if not provided
    max_depth = input_max_depth if input_max_depth is not None else problem.rows * problem.cols
    
    while stack:
        current_pos, path, depth = stack.pop()
        
        # Skip if already visited
        if current_pos in visited:
            continue
        
        visited.add(current_pos)
        nodes_expanded += 1
        
        # Add to history
        path_cost = problem.evaluate_path(path)
        history.add(
            x=[problem.path_to_vector(path)],
            value=[path_cost],
            info=f"Exploring node {current_pos}, depth={depth}, stack_size={len(stack)}"
        )
        
        # Check if goal is reached
        if problem.is_goal(current_pos):
            elapsed_time = timer.stop()
            cost = problem.evaluate_path(path)
            
            return SearchResult(
                algorithm="Depth-First Search",
                short_name="DFS",
                nodes_expanded=nodes_expanded,
                time=elapsed_time,
                cost=cost,
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
    elapsed_time = timer.stop()
    return SearchResult(
        algorithm="Depth-First Search",
        short_name="DFS",
        nodes_expanded=nodes_expanded,
        time=elapsed_time,
        cost=float('inf'),
        problem=problem,
        path=[]
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
    timer = TimerWrapper()
    timer.start()
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
        path_cost = problem.evaluate_path(path)
        h_score = problem.heuristic(current_pos, heuristic_name)
        history.add(
            x=[problem.path_to_vector(path)],
            value=[path_cost],
            info=f"Node {current_pos}, g={g_score:.2f}, h={h_score:.2f}, f={f_score:.2f}, queue_size={len(priority_queue)}"
        )
        
        # Check if goal is reached
        if problem.is_goal(current_pos):
            elapsed_time = timer.stop()
            cost = problem.evaluate_path(path)
            
            return SearchResult(
                algorithm=f"A* ({heuristic_name})",
                short_name="A*",
                nodes_expanded=nodes_expanded,
                time=elapsed_time,
                cost=cost,
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
    elapsed_time = timer.stop()
    return SearchResult(
        algorithm=f"A* ({heuristic_name})",
        short_name="A*",
        nodes_expanded=nodes_expanded,
        time=elapsed_time,
        cost=float('inf'),
        problem=problem,
        path=[]
    )