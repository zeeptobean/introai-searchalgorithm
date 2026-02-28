import math

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotly.subplots import make_subplots
from IPython.display import display, HTML
from function.discrete_function import *
from util.result import *
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

    def visualize_2d(self, 
            canvas_size: tuple[float, float] = (10, 8),
            mesh_resolution: int = 100,
            interval: int = 100, 
            dark_theme: bool = False, 
            on_jupyter_notebook: bool = True, 
            export_html: str | None = None, 
            export_mp4: str | None = None):
        """
        Visualizes the optimization process in 2D contour plot
        Args:
            mesh_resolution: Number of points along each axis for the surface grid.
            canvas_size: Width and height of the figure in inches.
            interval: Time in milliseconds between animation frames.
            dark_theme: Whether to use a dark theme
            on_jupyter_notebook: Whether to display the animation in a Jupyter Notebook. Set to False to display in a separate window
            export_html: If provided, saves the animation as an HTML file
            export_mp4: If provided, saves the animation as an MP4 video file
        """

        if self.problem.dimension != 2:
            raise ValueError("2D visualization requires dimension=2")
        
        # 1. Theme and Color setup
        if dark_theme:
            plt.style.use('dark_background')
            surface_colorscale = 'plasma'
            point_color = '#00FFFF'
            best_point_color = '#FFFFFF'
        else:
            plt.style.use('default')
            surface_colorscale = 'viridis'
            point_color = '#FF0000'
            best_point_color = '#FFD000'
            
        fig, ax = plt.subplots(figsize=canvas_size)
        
        # Get grid data
        x, y, Z = self._create_mesh_grid(mesh_resolution)
        
        # Strict mathematical bounds based on your mesh grid
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        
        # 2. Draw Landscape (Contour)
        # Assuming x and y are 1D arrays and Z is 2D. 
        # contourf provides filled contours similar to Plotly's default contour.
        contour = ax.contourf(x, y, Z, levels=30, cmap=surface_colorscale, alpha=0.7)
        fig.colorbar(contour, ax=ax, label='f(x)')
        
        # 3. Draw the Best Point
        ax.scatter(
            self.result.best_x[0], self.result.best_x[1],
            c=best_point_color, marker='*', s=200, edgecolors='black', linewidths=1,
            label=f'Best: {self.result.best_value:.6g}', zorder=3
        )
        
        # 4. Filter initial swarm to discard out-of-bounds points
        init_x, init_y = [], []
        for p in self.result.history.history_x[0]:
            if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max:
                init_x.append(p[0])
                init_y.append(p[1])
                
        # Draw Initial Swarm
        swarm_scatter = ax.scatter(
            init_x, init_y, c=point_color, s=40, edgecolors='black', linewidths=1,
            label='Agent', zorder=2
        )
        
        # 5. Lock the axes
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        
        # Format Title (Matplotlib handles newlines with \n instead of <br>)
        title_text = (
            f"2D Contour: {str(self.problem)}; runtime={self.result.time:.2f}ms\n"
            f"{self.result.algorithm}; rngseed={self.result.rng_seed}"
        )
        ax.set_title(title_text)
        
        # Place legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), borderaxespad=0.)
        fig.tight_layout() # Ensures legend and labels aren't cut off

        # 6. Animation Update Function
        def update(frame_idx):
            curr_x, curr_y = [], []
            
            # Filter animation frames to discard out-of-bounds points
            for p in self.result.history.history_x[frame_idx]:
                if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max:
                    curr_x.append(p[0])
                    curr_y.append(p[1])
            
            # Matplotlib scatter set_offsets expects an (N, 2) array
            if curr_x: # Check if there are points to plot
                swarm_scatter.set_offsets(np.c_[curr_x, curr_y])
            else:
                swarm_scatter.set_offsets(np.empty((0, 2)))
                
            return swarm_scatter,
        
        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, 
            frames=len(self.result.history.history_x),
            interval=interval, 
            blit=True, # Blitting optimizes rendering by only redrawing changed elements
            repeat=False
        )

        if export_mp4:
            mp4_writer = animation.FFMpegWriter(
                fps=10, 
                codec='libx264', 
                bitrate=1800, # Higher bitrate = better quality, larger file
                extra_args=['-pix_fmt', 'yuv420p'] # Ensures compatibility across most media players
            )
            ani.save(export_mp4, writer=mp4_writer)
            print(f"Animation video saved as {export_mp4}")
        
        html_string = ani.to_jshtml()
        if export_html:
            with open(export_html, 'w') as f:
                f.write(html_string)
            print(f"Animation HTML saved as {export_html}")

        if on_jupyter_notebook:
            html_anim = HTML(html_string)
            plt.close() # Prevents a static plot from showing up in Jupyter
            display(html_anim)
        else:
            plt.show()
        
        # It's important to return the 'ani' object as well. 
        # If the animation object is garbage collected, the animation will freeze.
        return fig, ani
    
    def visualize_3d(self,  
            mesh_resolution: int = 60,
            canvas_size: tuple[float, float] = (1040, 780),
            interval: int = 100, 
            dark_theme: bool = False):
        """
        Visualizes the optimization process in 3D
        Args:
            mesh_resolution: Number of points along each axis for the surface grid.
            canvas_size: Width and height of the figure in pixel.
            interval: Time in milliseconds between animation frames.
            dark_theme: Whether to use a dark theme
        """
        if self.problem.dimension != 2:
            raise ValueError("3D visualization requires dimension=2")
        
        plotly_template: str = "plotly_dark" if dark_theme else "plotly_white"
        surface_colorscale: str = "Plasma" if dark_theme else "Viridis"
        point_color = "#00FFFF" if dark_theme else "#FF0000"       
        best_point_color = "#FFFFFF" if dark_theme else "#FFD000"  
        
        x, y, Z = self._create_mesh_grid(mesh_resolution)
        
        # 1. Base Traces
        surface = go.Surface(z=Z, x=x, y=y, colorscale=surface_colorscale, opacity=0.8, name='Landscape')
        
        best_point = go.Scatter3d(
            x=[self.result.best_x[0]], y=[self.result.best_x[1]], z=[self.result.best_value],
            mode='markers', marker=dict(color=best_point_color, symbol='diamond', size=5),
            name=f'Best: {self.result.best_value:.6g}',
            hovertemplate="Best Value: %{z} at (%{x}, %{y})<extra></extra>"
        )
        
        init_x = [p[0] for p in self.result.history.history_x[0]]
        init_y = [p[1] for p in self.result.history.history_x[0]]
        init_z = self.result.history.history_value[0]
        swarm = go.Scatter3d(
            x=init_x, y=init_y, z=init_z, mode='markers',
            marker=dict(color=point_color, size=3),
            name='Agent'
        )
        
        fig = go.Figure(data=[surface, best_point, swarm])
        
        # 2. Animations
        frames = []
        for i in range(len(self.result.history.history_x)):
            curr_x = [p[0] for p in self.result.history.history_x[i]]
            curr_y = [p[1] for p in self.result.history.history_x[i]]
            curr_z = self.result.history.history_value[i]
            
            frame = go.Frame(
                data=[go.Scatter3d(x=curr_x, y=curr_y, z=curr_z)],
                traces=[2], # Target the swarm trace (index 2)
                name=str(i)
            )
            frames.append(frame)
            
        fig = self._add_animation_controls(fig, frames, interval)

        fig.update_layout(
            template=plotly_template,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.15),
            title_text=(
                f"3D Graph: {str(self.problem)}; runtime={self.result.time:.2f}ms<br>"
                f"<sup>{self.result.algorithm}; rngseed={self.result.rng_seed}</sup>"
            ),
            scene=dict(xaxis_title='x₁', yaxis_title='x₂', zaxis_title='f(x)'),
            width=canvas_size[0], height=canvas_size[1],
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        fig.show()
        return fig
    
def visualize_convergence(result: ContinuousResult | DiscreteResult, dark_theme: bool = False):
    history = result.history
    history_value = history.history_value
    iterations: int = len(history_value)
    num_agents: int = len(history_value[0]) if iterations > 0 else 0

    # --- 0. Theme Configuration ---
    plotly_template: str = "plotly_dark" if dark_theme else "plotly_white"

    # --- 1. Process Convergence Data ---
    best_vals: list[float] = []
    avg_vals: list[float] = []
    
    for gen in history_value:
        valid_fitnesses: list[float] = [float(v) for v in gen if v is not None and not math.isnan(float(v))] 
        
        if valid_fitnesses:
            # Note: Continuous optimization generally defaults to minimization
            best_vals.append(min(valid_fitnesses)) 
            avg_vals.append(float(np.mean(valid_fitnesses)))
        else:
            best_vals.append(math.nan)
            avg_vals.append(math.nan)

    # Find the First Overall Best Value
    overall_best_val = float(result.best_value)
    first_best_idx = -1
    
    if not math.isinf(overall_best_val) and not math.isnan(overall_best_val):
        # Using math.isclose since continuous values are floats
        for i, val in enumerate(best_vals):
            if math.isclose(val, overall_best_val, rel_tol=1e-9, abs_tol=1e-9):
                first_best_idx = i
                break

    # --- 2. Build the Plotly Figure ---
    fig = go.Figure()

    best_line_name = 'Best Value' if num_agents > 1 else 'Current Value'
    best_line_width = 3 if iterations < 300 else 1 
    
    fig.add_trace(
        go.Scatter(
            x=list(range(iterations)), y=best_vals, 
            mode='lines', name=best_line_name, 
            line=dict(color='#2ecc71', width=best_line_width),
            connectgaps=True 
        )
    )
    
    # Add the Star Marker for the first best value
    if first_best_idx != -1:
        fig.add_trace(
            go.Scatter(
                x=[first_best_idx], 
                y=[overall_best_val], 
                mode='markers', 
                name=f'First Best ({overall_best_val:.6g})', 
                marker=dict(symbol='star', size=16, color='gold', line=dict(color='black', width=1)),
                hovertemplate="Iteration: %{x}<br>Best Value: %{y}<extra></extra>"
            )
        )
    
    # Only render the average line if there is actually a swarm/population
    if num_agents > 1:
        fig.add_trace(
            go.Scatter(
                x=list(range(iterations)), y=avg_vals, 
                mode='lines', name='Average Value', 
                line=dict(color='#f39c12', width=2, dash='dash'),
                connectgaps=True
            )
        )

    # --- 3. Layout Adjustments ---
    fig.update_layout(
        title_text=(
            f"Convergence: {str(result.problem)}; runtime={result.time:.2f}ms<br>"
            f"<sup>{result.algorithm}; rngseed={result.rng_seed}</sup>"
        ),
        template=plotly_template,
        hovermode="x unified", 
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Iteration",
        yaxis_title="Objective Value",
        height=500  # Fixed height since there is no heatmap to expand
    )

    fig.show()

def visualize_knapsack(result: 'DiscreteResult', dark_theme: bool = False) -> None:
    smooth_transition_threshold = 300
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

    # --- 1. Process Convergence Data ---
    best_vals: list[float] = []
    avg_vals: list[float] = []
    
    for gen in history.history_value:
        valid_fitnesses: list[float] = [float(v) for v in gen if v is not None and float(v) > -1e6] 
        
        if valid_fitnesses:
            best_vals.append(max(valid_fitnesses))
            avg_vals.append(float(np.mean(valid_fitnesses)))
        else:
            best_vals.append(math.nan)
            avg_vals.append(math.nan)

    # Find the First Overall Best Value
    overall_best_val = float(result.best_value)
    if math.isinf(overall_best_val) or math.isnan(overall_best_val):
        first_best_idx = -1
    else:
        first_best_idx = best_vals.index(overall_best_val)

    # --- 2. Process Heatmap Data & Labels ---
    heatmap_data: np.ndarray = np.zeros((num_items, iterations))
    for t in range(iterations):
        gen_x_array: np.ndarray = np.array(history.history_x[t])
        heatmap_data[:, t] = np.mean(gen_x_array, axis=0)

    if not hasattr(result.problem, 'weights') or not hasattr(result.problem, 'values'):
        raise TypeError("result.problem must be of type KnapsackFunction with defined weights and values.")
    
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
            f"{result.algorithm}; rngseed={result.rng_seed}", 
            "Item selection probability"
        ),
        row_heights=[0.3, 0.7] # Gave a bit more room to the heatmap to help with many items
    )

    # Top Plot: Convergence Lines
    best_line_name = 'Best Value' if num_agents > 1 else 'Current Value'
    best_line_width = 3 if iterations < smooth_transition_threshold else 1 
    
    fig.add_trace(
        go.Scatter(
            x=list(range(iterations)), y=best_vals, 
            mode='lines', name=best_line_name, 
            line=dict(color='#2ecc71', width=best_line_width),
            connectgaps=True 
        ),
        row=1, col=1
    )
    
    # Add the Star Marker for the first best value
    if first_best_idx != -1:
        fig.add_trace(
            go.Scatter(
                x=[first_best_idx], 
                y=[overall_best_val], 
                mode='markers', 
                name=f'First Best ({overall_best_val})', 
                marker=dict(symbol='star', size=16, color='gold', line=dict(color='black', width=1)),
                hovertemplate="Iteration: %{x}<br>Best Value: %{y}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Only render the average line if there is actually a swarm
    if num_agents > 1:
        fig.add_trace(
            go.Scatter(
                x=list(range(iterations)), y=avg_vals, 
                mode='lines', name='Average Value', 
                line=dict(color='#f39c12', width=2, dash='dash'),
                connectgaps=True
            ),
            row=1, col=1
        )

    # Bottom Plot: Item Selection Heatmap
    dynamic_xgap = 1 if iterations <= smooth_transition_threshold else 0 

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=list(range(iterations)),
            y=y_labels,
            colorscale=heatmap_colorscale, 
            zmin=0, zmax=1,      
            xgap=dynamic_xgap,   
            ygap=1,              
            colorbar=dict(title="Selection Rate", x=1.02),
            hovertemplate="Iteration: %{x}<br>%{y}<br>Selected by: %{z:.1%}<extra></extra>"
        ),
        row=2, col=1
    )

    # --- 4. Layout Adjustments (Dynamic Height implemented here) ---
    # Assign at least 20 pixels of height per item so they are never squished, plus 400px for the top chart.
    dynamic_height = max(800, 400 + (num_items * 20))

    fig.update_layout(
        height=dynamic_height, 
        title_text=f"Convergence: Knapsack(capacity={result.problem.capacity}); runtime={result.time:.2f}ms",
        template=plotly_template,
        hovermode="x unified", 
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=150) 
    )

    fig.update_yaxes(title_text="Objective Value", row=1, col=1)
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