from nicegui import ui
import plotly.graph_objects as go

from .common import Chart, idx_to_color


class MuscleChart(Chart):
    def __init__(self, fig: go.Figure, plot: ui.plotly,
                 title: str, x_label: str, y_label: str):
        super().__init__(fig, plot)
        self.update_axes(
            title=title,
            x_label=x_label,
            y_label=y_label
        )
        self.keys = []

    def update(self, times, values, ind_done):
        self.keys = list(values.keys())

        # Create dropdown options based on number of keys
        n_keys = len(self.keys)

        n_shown = 6
        dropdown_buttons = []
        for start_idx in range(0, n_keys, n_shown):
            end_idx = min(start_idx + n_shown, n_keys)
            range_label = f"{start_idx}-{end_idx-1}"

            # Create visibility array for this range
            visibility = [False] * (n_keys * 2)
            for i in range(start_idx, end_idx):
                visibility[i * 2] = True      # Line trace
                visibility[i * 2 + 1] = True  # Marker trace

            dropdown_buttons.append(dict(
                label=range_label,
                method="restyle",
                args=[{"visible": visibility}]
            ))


        # Add all traces
        for i, key in enumerate(self.keys):
            self.fig.add_trace(go.Scatter(
                x=times,
                y=values[key],
                mode='lines',
                name=key,
                line=dict(color=idx_to_color(i), width=2),
                visible=(i < n_shown)
            ))
            self.fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=8, color=idx_to_color(i)),
                name=key,
                showlegend=False,
                visible=(i < n_shown)
            ))

        # Update layout with dropdown
        self.fig.update_layout(
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=1.0,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )

        self.add_episode_boundaries(times[ind_done])
        self.plot.update()

    def update_frame(self, frame_idx: int):
        for i, key in enumerate(self.keys):
            x_val = self.fig.data[i * 2].x[frame_idx]
            y_val = self.fig.data[i * 2].y[frame_idx]
            self.fig.data[i * 2 + 1].x = [x_val]
            self.fig.data[i * 2 + 1].y = [y_val]
        self.plot.update()
        return
