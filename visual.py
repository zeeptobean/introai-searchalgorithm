import numpy as np
import plotly.graph_objects as go
from util.result import ContinuousResult
from util.define import *

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