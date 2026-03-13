from util.define import *
from typing import Tuple, List, Set, override, Callable
import numpy as np
import numpy.typing as npt

class GridWorldProblem:
    """
    Graph problem for finding shortest path in a grid/matrix with obstacles.
    Grid values: 0 = free space, 1 = obstacle
    
    This follows the same pattern as DiscreteProblem for consistency.
    Uses path as solution vector and path cost as objective value.
    """
    def __init__(self, grid: npt.NDArray[np.int64], start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize the graph problem.
        
        Args:
            grid: 2D numpy array where 0 = passable, 1 = obstacle
            start: (row, col) starting position
            goal: (row, col) goal position
        """
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise ValueError("Grid must be a 2D numpy array")
        
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        self.start = start
        self.goal = goal
        
        # Validate start and goal positions
        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is invalid or blocked")
        if not self._is_valid_position(goal):
            raise ValueError(f"Goal position {goal} is invalid or blocked")
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not an obstacle"""
        row, col = pos
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col] == 0
        return False
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions (up, down, left, right).
        Can be extended to include diagonal movements.
        """
        row, col = pos
        neighbors = []
        
        # 4-directional movement (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Uncomment for 8-directional movement (including diagonals)
        # directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
        #               (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            new_pos = (row + dr, col + dc)
            if self._is_valid_position(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def is_goal(self, pos: Tuple[int, int]) -> bool:
        """Check if the position is the goal"""
        return pos == self.goal
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def heuristic(self, pos: Tuple[int, int], heuristic_name: str) -> float:
        """
        Heuristic function for A*.
        Returns estimated distance from pos to goal.
        """
        if heuristic_name == "manhattan":
            return self.manhattan_distance(pos, self.goal)
        elif heuristic_name == "euclidean":
            return self.euclidean_distance(pos, self.goal)
        else:
            raise ValueError("Unsupported heuristic name")
    
    def get_path_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Get the cost of moving from pos1 to pos2.
        Returns 1 for adjacent cells, sqrt(2) for diagonal moves.
        """
        dr = abs(pos1[0] - pos2[0])
        dc = abs(pos1[1] - pos2[1])
        
        if dr + dc == 1:  # Adjacent cells
            return 1.0
        elif dr == 1 and dc == 1:  # Diagonal
            return np.sqrt(2)
        else:
            raise ValueError("Positions are not neighbors")
    
    @property
    def dimension(self) -> int:
        """Return grid size as dimension (for compatibility with DiscreteProblem)"""
        return self.rows * self.cols
    
    def is_max_value_problem(self) -> bool:
        """Pathfinding minimizes cost, so return False"""
        return False
    
    def path_to_vector(self, path: List[Tuple[int, int]]) -> FloatVector:
        """Convert path (list of positions) to a vector representation"""
        if not path:
            return np.array([], dtype=float)
        # Flatten path: [row1, col1, row2, col2, ...]
        vec = np.array([coord for pos in path for coord in pos], dtype=float)
        return vec
    
    def vector_to_path(self, vec: FloatVector) -> List[Tuple[int, int]]:
        """Convert vector back to path"""
        if len(vec) == 0:
            return []
        # Reshape to pairs
        path = [(int(vec[i]), int(vec[i+1])) for i in range(0, len(vec), 2)]
        return path
    
    def evaluate_path(self, path: List[Tuple[int, int]]) -> Float:
        """Calculate path cost (used as objective value)"""
        if not path or len(path) < 2:
            return float('inf')
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self.get_path_cost(path[i], path[i+1])
        return total_cost
    
    def __repr__(self) -> str:
        return f"GridWorldProblem(grid_size={self.grid.shape}, start={self.start}, goal={self.goal})"
    
    def to_json(self) -> dict:
        """Serialize to JSON format"""
        return {
            "problem_type": "GridWorldProblem",
            "grid": self.grid.tolist(),
            "start": list(self.start),
            "goal": list(self.goal),
            "rows": self.rows,
            "cols": self.cols
        }
    
    @staticmethod
    def from_json(json_data: dict) -> 'GridWorldProblem':
        """Reconstruct GridWorldProblem from JSON"""
        grid = np.array(json_data["grid"], dtype=np.int64)
        start = tuple(json_data["start"])
        goal = tuple(json_data["goal"])
        return GridWorldProblem(grid, start, goal)
    
    def visualize_path(self, path: List[Tuple[int, int]] | None = None) -> str:
        """
        Create a string visualization of the grid and path.
        S = Start, G = Goal, # = Obstacle, . = Free space, * = Path
        """
        # Create a copy for visualization
        vis = np.empty(self.grid.shape, dtype=str)
        vis[self.grid == 0] = '.'
        vis[self.grid == 1] = '#'
        
        # Mark path
        if path:
            for pos in path:
                if pos != self.start and pos != self.goal:
                    vis[pos] = '*'
        
        # Mark start and goal
        vis[self.start] = 'S'
        vis[self.goal] = 'G'
        
        # Convert to string
        lines = []
        for row in vis:
            lines.append(' '.join(row))
        
        return '\n'.join(lines)


def create_simple_grid(rows: int = 10, cols: int = 10, obstacle_ratio: float = 0.2, 
                        seed: int | None = None) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Create a random grid with obstacles for testing.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        obstacle_ratio: Ratio of obstacles (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (grid, start_pos, goal_pos)
    """
    if seed is not None:
        np.random.seed(seed)
    
    grid = np.zeros((rows, cols), dtype=int)
    
    # Add random obstacles
    num_obstacles = int(rows * cols * obstacle_ratio)
    for _ in range(num_obstacles):
        r, c = np.random.randint(0, rows), np.random.randint(0, cols)
        grid[r, c] = 1
    
    # Ensure start and goal are not blocked
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    grid[start] = 0
    grid[goal] = 0
    
    return grid, start, goal
