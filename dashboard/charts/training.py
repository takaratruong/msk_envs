from nicegui import ui
import plotly.graph_objects as go
import numpy as np

from .common import Chart, idx_to_color


class TrainingChart(Chart):
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

    def update(self, times, values, ind_done):
        # data goes mean_reward, std_reward, min_reward, max_reward
        # first plot the mean, with std as a band
        values = np.array(values)
        mean_rewards = values[:, 0]
        std_rewards = values[:, 1]
        st_err_rewards = values[:, 2]
        min_rewards = values[:, 3]
        max_rewards = values[:, 4]
        range_rewards = max_rewards - min_rewards
        self.fig.add_trace(go.Scatter(
            x=times,
            y=mean_rewards,
            error_y=dict(
                type='data',
                array=st_err_rewards,
                visible=True
            ),
            mode='lines',
            name='Mean Reward',
            line=dict(color=idx_to_color(0), width=2)
        ))
        self.plot.update()

    def update_frame(self, frame_idx: int):
        pass
