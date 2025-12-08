""" Shared functions for standardizing chart creation and layout """
from nicegui import ui
from plotly import graph_objects as go
import plotly.io as pio


class Chart:
    def __init__(self, fig: go.Figure, plot: ui.plotly):
        self.fig = fig
        self.plot = plot

    def update_axes(self, title: str, x_label: str, y_label: str):
        """ Set standard layout for all figures """
        self.fig.update_layout(
            title=title,
            height=400,
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50),
            title_font=dict(size=24)
        )
        self.fig.update_layout(title_x=0.5)
        self.fig.update_xaxes(title_text=x_label,
                              title_font=dict(size=20),
                              showline=True,
                              showticklabels=True,
                              linecolor='black',
                              linewidth=2.5,
                              ticks='outside',
                              mirror='allticks',
                              tickwidth=2.5,
                              tickcolor='black')
        self.fig.update_yaxes(title_text=y_label,
                              title_font=dict(size=20),
                              showline=True,
                              linecolor='black',
                              linewidth=2.5,
                              ticks='outside',
                              mirror='allticks',
                              tickwidth=2.5,
                              tickcolor='black')
        self.fig.update_layout(
            legend=dict(
                itemsizing='constant',
                itemwidth=30,
            )
        )

    def set_x_range(self, x_min: float, x_max: float):
        self.fig.update_layout(xaxis=dict(range=[x_min, x_max]))
        return

    def set_y_range(self, y_min: float, y_max: float):
        self.fig.update_layout(yaxis=dict(range=[y_min, y_max]))
        return

    def fit_x_around_frame(self, frame_idx: int, window: int = 100):
        """ Only show a window around current time """
        x_values = self.fig.data[0].x  # assuming first trace has the x data :(
        n_frames = len(x_values)

        start = max(0, (frame_idx // window) * window)
        end = min(n_frames - 1, start + window)
        x_min = x_values[start]
        x_max = x_values[end]

        # Check if we need to change at all
        x_min_curr, x_max_curr = self.fig.layout.xaxis.range or (None, None)
        if x_min_curr == x_min and x_max_curr == x_max:
            return

        self.set_x_range(x_min, x_max)
        return

    def max_x_range(self):
        x_values = self.fig.data[0].x
        self.set_x_range(min(x_values), max(x_values))

    def add_episode_boundaries(self, times_done: list[int]):
        """ Add vertical lines to indicate episode boundaries """
        # for idx in times_done:
        #     self.fig.add_vline(x=idx, line_width=2, line_dash="dash",
        #                        line_color="gray")
        return  # not working yet

    def reset(self):
        """ clear chart """
        self.fig.data = []
        self.plot.update()

    def hide(self):
        self.plot.visible = False

    def show(self):
        self.plot.visible = True

    def update_chart(self, times, values, ind_done):
        """
        Update chart with new data
        *to implement in child classes*
         """
        raise NotImplementedError

    def update_frame(self, frame_idx):
        """
        Optionally update chart to highlight current frame
        *to implement in child classes*
        """
        raise NotImplementedError


def idx_to_color(idx: int) -> str:
    default_colors = pio.templates[pio.templates.default].layout.colorway
    return default_colors[idx % len(default_colors)]
