from nicegui import ui
import plotly.graph_objects as go

from .common import Chart, idx_to_color


class RewChart(Chart):
    def __init__(self, fig: go.Figure, plot: ui.plotly,
                 title: str, x_label: str, y_label: str):
        super().__init__(fig, plot)
        self.update_axes(
            title=title,
            x_label=x_label,
            y_label=y_label
        )
        self.keys = []

    def update(self, times, rew_values, ind_done):
        self.keys = rew_values[0].keys()
        for i, key in enumerate(self.keys):
            self.fig.add_trace(go.Scatter(
                x=times,
                y=[rew[key] for rew in rew_values],
                mode='lines',
                name=key,
                line=dict(color=idx_to_color(i), width=2)
            ))
            self.fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=8, color=idx_to_color(i)),
                name=key,
                showlegend=False
            ))

        self.add_episode_boundaries(times[ind_done])
        self.plot.update()

    def update_frame(self, frame_idx: int):
        for i, key in enumerate(self.keys):
            x_val = self.fig.data[i * 2].x[frame_idx]
            y_val = self.fig.data[i * 2].y[frame_idx]
            self.fig.data[i * 2 + 1].x = [x_val]
            self.fig.data[i * 2 + 1].y = [y_val]
        self.plot.update()
