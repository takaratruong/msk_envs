from nicegui import ui
import plotly.graph_objects as go

from .common import Chart, idx_to_color


class Vec3Chart(Chart):
    def __init__(self,
                 fig: go.Figure,
                 plot: ui.plotly,
                 title: str,
                 x_label: str,
                 y_label: str):
        super().__init__(fig, plot)
        self.update_axes(
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

    def update(self, times, vec_values, ind_done):
        labels = ["x", "y", "z"]
        for i in range(3):
            self.fig.add_trace(go.Scatter(
                x=times,
                y=[grf[i] for grf in vec_values],
                mode='lines',
                name=f'{labels[i]}',
                line=dict(color=idx_to_color(i), width=2)
            ))

        self.add_episode_boundaries(times[ind_done])

        for i in range(3):
            self.fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=8, color=idx_to_color(i)),
                name=f'{labels[i]} marker',
                showlegend=False
            ))
        self.plot.update()

    def update_frame(self, frame_idx: int):
        for i in range(3):
            x_val = self.fig.data[i].x[frame_idx]
            y_val = self.fig.data[i].y[frame_idx]
            self.fig.data[i + 3].x = [x_val]
            self.fig.data[i + 3].y = [y_val]
        self.plot.update()
