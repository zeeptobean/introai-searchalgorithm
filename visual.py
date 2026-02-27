import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from function.discrete_function import *
from util.result import ContinuousResult, DiscreteResult
from util.define import *
from util.util import *

class ContinuousResultVisualizer:
    """Visualize continuous optimization results with interactive Plotly 3D/2D plots."""
    
    def __init__(self, result: ContinuousResult):
        self.result = result
        self.problem = result.problem
        
    def _create_mesh_grid(self, resolution: int):
        """Helper method to generate the function landscape data."""
        x_min, x_max = self.problem.lower_bound or -10, self.problem.upper_bound or 10
        y_min, y_max = self.problem.lower_bound or -10, self.problem.upper_bound or 10
        
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        XY = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = np.array([self.problem.evaluate(xy) for xy in XY]).reshape(X.shape)
        
        return x, y, Z

    def _add_animation_controls(self, fig, frames, interval: int):
        """Helper to add Play/Pause buttons and a timeline slider to the figure."""
        fig.frames = frames
        
        sliders = [{
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], {"frame": {"duration": interval, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": interval}}],
                    "label": f.name,
                    "method": "animate",
                }
                for f in frames
            ],
        }]

        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": interval, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": interval}}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate",
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=sliders
        )
        return fig

    def visualize_2d(self, resolution: int = 100, animate: bool = True, 
                     interval: int = 100, save_path: str | None = None):
        if self.problem.dimension != 2:
            raise ValueError("2D visualization requires dimension=2")
        
        x, y, Z = self._create_mesh_grid(resolution)
        
        # 1. Base Traces
        contour = go.Contour(z=Z, x=x, y=y, colorscale='Viridis', opacity=0.7, name='Function')
        
        best_point = go.Scatter(
            x=[self.result.best_x[0]], y=[self.result.best_x[1]],
            mode='markers', marker=dict(color='red', symbol='star', size=15),
            name=f'Best: {self.result.best_value:.4f}'
        )
        
        init_x = [p[0] for p in self.result.history_x[0]]
        init_y = [p[1] for p in self.result.history_x[0]]
        swarm = go.Scatter(
            x=init_x, y=init_y, mode='markers',
            marker=dict(color='magenta', size=8, line=dict(color='black', width=1)),
            name='Swarm'
        )
        
        fig = go.Figure(data=[contour, best_point, swarm])
        
        # 2. Animations
        if animate:
            frames = []
            for i in range(len(self.result.history_x)):
                curr_x = [p[0] for p in self.result.history_x[i]]
                curr_y = [p[1] for p in self.result.history_x[i]]
                
                frame = go.Frame(
                    data=[go.Scatter(x=curr_x, y=curr_y)],
                    traces=[2], # Target the swarm trace (index 2)
                    name=str(i)
                )
                frames.append(frame)
                
            fig = self._add_animation_controls(fig, frames, interval)

        fig.update_layout(
            title=f'{self.result.algorithm} on {self.problem}',
            xaxis_title='x₁', yaxis_title='x₂',
            width=800, height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_3d(self, resolution: int = 60, animate: bool = True,
                     interval: int = 100, save_path: str | None = None):
        if self.problem.dimension != 2:
            raise ValueError("3D visualization requires dimension=2")
        
        x, y, Z = self._create_mesh_grid(resolution)
        
        # 1. Base Traces
        surface = go.Surface(z=Z, x=x, y=y, colorscale='Viridis', opacity=0.8, name='Landscape')
        
        best_point = go.Scatter3d(
            x=[self.result.best_x[0]], y=[self.result.best_x[1]], z=[self.result.best_value],
            mode='markers', marker=dict(color='red', symbol='diamond', size=8),
            name=f'Best: {self.result.best_value:.4f}'
        )
        
        init_x = [p[0] for p in self.result.history_x[0]]
        init_y = [p[1] for p in self.result.history_x[0]]
        init_z = self.result.history_value[0]
        swarm = go.Scatter3d(
            x=init_x, y=init_y, z=init_z, mode='markers',
            marker=dict(color='magenta', size=5, line=dict(color='black', width=1)),
            name='Swarm'
        )
        
        fig = go.Figure(data=[surface, best_point, swarm])
        
        # 2. Animations
        if animate:
            frames = []
            for i in range(len(self.result.history_x)):
                curr_x = [p[0] for p in self.result.history_x[i]]
                curr_y = [p[1] for p in self.result.history_x[i]]
                curr_z = self.result.history_value[i]
                
                frame = go.Frame(
                    data=[go.Scatter3d(x=curr_x, y=curr_y, z=curr_z)],
                    traces=[2], # Target the swarm trace (index 2)
                    name=str(i)
                )
                frames.append(frame)
                
            fig = self._add_animation_controls(fig, frames, interval)

        fig.update_layout(
            title=f'{self.result.algorithm} on {self.problem}',
            scene=dict(xaxis_title='x₁', yaxis_title='x₂', zaxis_title='f(x)'),
            width=900, height=800,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def visualize_convergence(self, save_path: str | None = None):
        best_values = []
        current_best = float('inf')
        for iteration_values in self.result.history_value:
            current_best = min(current_best, min(iteration_values))
            best_values.append(current_best)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(best_values))), y=best_values,
            mode='lines', name='Current Best', line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(y=self.result.best_value, line_dash="dash", line_color="red", 
                      annotation_text=f'Final Best: {self.result.best_value:.6f}')
        
        fig.update_layout(
            title=f'Convergence - {self.result.algorithm}',
            xaxis_title='Iteration', yaxis_title='Best Function Value',
            width=800, height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig

def visualize_result(result: ContinuousResult, method: str = '3d', 
                     animate: bool = True, save_path: str | None = None, **kwargs):
    visualizer = ContinuousResultVisualizer(result)
    
    if method == '2d':
        fig = visualizer.visualize_2d(animate=animate, save_path=save_path, **kwargs)
    elif method == '3d':
        fig = visualizer.visualize_3d(animate=animate, save_path=save_path, **kwargs)
    elif method == 'convergence':
        fig = visualizer.visualize_convergence(save_path=save_path)
    else:
        raise ValueError(f"Unknown method: {method}. Use '2d', '3d', or 'convergence'")
        
    # Plotly's show() will automatically open a tab in your default web browser
    fig.show()
    return fig

def visualize_knapsack(result: 'DiscreteResult', dark_theme: bool = False) -> None:
    history = result.history
    iterations: int = len(history.history_x)
    
    if not isinstance(result.problem, KnapsackFunction):
        raise TypeError(f"Expected KnapsackFunction, got {type(result.problem)}")
    if iterations == 0:
        print("No history data to visualize.")
        return

    num_agents: int = len(history.history_x[0])
    num_items: int = len(history.history_x[0][0])

    # --- 0. Theme Configuration ---
    plotly_template: str = "plotly_dark" if dark_theme else "plotly_white"
    heatmap_colorscale: str = "Plasma" if dark_theme else "Viridis"

    # --- 1. Process Convergence Data (Fixing the "gap") ---
    best_vals: list[float] = []
    avg_vals: list[float] = []
    
    for gen in history.history_value:
        # Filter out extreme penalties (e.g., massive negative numbers for overweight knapsacks)
        valid_fitnesses: list[float] = [float(v) for v in gen if v is not None and float(v) > -1e6] 
        
        if valid_fitnesses:
            best_vals.append(max(valid_fitnesses))
            avg_vals.append(float(np.mean(valid_fitnesses)))
        else:
            best_vals.append(np.nan)
            avg_vals.append(np.nan)

    # --- 2. Process Heatmap Data & Labels ---
    heatmap_data: np.ndarray = np.zeros((num_items, iterations))
    for t in range(iterations):
        gen_x_array: np.ndarray = np.array(history.history_x[t])
        heatmap_data[:, t] = np.mean(gen_x_array, axis=0)

    # Type guard / Cast to ensure we are dealing with a Knapsack problem with weights/values
    if not hasattr(result.problem, 'weights') or not hasattr(result.problem, 'values'):
        raise TypeError("result.problem must be of type KnapsackFunction with defined weights and values.")
    
    # Format the Y-axis labels to show Item Index, Weight, and Value
    y_labels: list[str] = [
        f"#{i} (w:{w}, v:{v})" 
        for i, (w, v) in enumerate(zip(result.problem.weights, result.problem.values))
    ]

    # --- 3. Build the Plotly Figure ---
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Algorithm Convergence ({result.algorithm}, Seed: {result.rng_seed})", 
            "Item Selection Consensus Over Time (Exploration vs. Exploitation)"
        ),
        row_heights=[0.4, 0.6]
    )

    # Top Plot: Convergence Lines
    fig.add_trace(
        go.Scatter(
            x=list(range(iterations)), y=best_vals, 
            mode='lines', name='Best Value in Swarm', 
            line=dict(color='#2ecc71', width=3),
            connectgaps=True # Ensures the line continues even if an iteration is fully NaN
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(iterations)), y=avg_vals, 
            mode='lines', name='Average Swarm Value', 
            line=dict(color='#f39c12', width=2, dash='dash'),
            connectgaps=True
        ),
        row=1, col=1
    )

    # Bottom Plot: Item Selection Heatmap
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=list(range(iterations)),
            y=y_labels,
            colorscale=heatmap_colorscale, 
            zmin=0, zmax=1,      # Forces the color scale to stay locked between 0% and 100%
            xgap=1,              # Adds a distinct line between iterations
            ygap=1,              # Adds a distinct line between items
            colorbar=dict(title="Selection Rate", x=1.02),
            hovertemplate="Iteration: %{x}<br>%{y}<br>Selected by: %{z:.1%}<extra></extra>"
        ),
        row=2, col=1
    )

    # --- 4. Layout Adjustments ---
    fig.update_layout(
        height=850, # Slightly taller to make room for all 40 item labels
        title_text=f"Knapsack Metaheuristic Analysis (Run Time: {result.time:.2f}ms)",
        template=plotly_template,
        hovermode="x unified", 
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=150) # Adds padding on the left so the "Item (w: X, v: Y)" labels aren't cut off
    )

    fig.update_yaxes(title_text="Objective Value", row=1, col=1)
    
    # Optional: Reverse the Y-axis so Item 0 is at the top of the heatmap
    fig.update_yaxes(autorange="reversed", row=2, col=1) 
    fig.update_xaxes(title_text="Iteration", row=2, col=1)

    fig.show()

def visualize_graph_coloring(result: DiscreteResult) -> None:
    """
    Visualizes the result of a graph coloring optimization.
    """
    # 1. Ensure the problem contains an adjacency matrix
    if not isinstance(result.problem, GraphColoringFunction):
        raise TypeError("Result problem must be an instance of GraphColoringFunction to visualize.")

    adj_matrix = result.problem.adjacency_matrix
    
    # 2. Build the NetworkX graph and calculate layout
    G = nx.from_numpy_array(adj_matrix)
    
    # Use the RNG seed from the result to keep the layout reproducible
    pos = nx.spring_layout(G, seed=result.rng_seed, k=0.5, iterations=50) 

    # 3. Create Edge Trace
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # 4. Create Node Trace
    node_x: list[float] = []
    node_y: list[float] = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Map best_x (color indices) to a discrete qualitative palette
    color_indices = result.best_x.astype(int)
    palette = pc.qualitative.Plotly  # Built-in Plotly distinct colors
    node_colors = [palette[idx % len(palette)] for idx in color_indices]
    
    hover_texts = [f"Node {node}<br>Color Index: {color_indices[node]}" for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=hover_texts,
        text=[str(node+1) for node in G.nodes()],
        textposition="middle center",
        textfont=dict(color='white', size=10),
        marker=dict(
            showscale=False,
            color=node_colors,
            size=35,
            line=dict(width=2, color='DarkSlateGrey')
        )
    )

    # 5. Compile the Figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f'Graph Coloring Solution<br><sup>Total Colors Used: {int(result.best_value)}</sup>',
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )

    fig.show()