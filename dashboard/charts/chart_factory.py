from .vec import Vec3Chart
from .rewards import RewChart
from .muscle import MuscleChart
from .training import TrainingChart

import plotly.graph_objects as go
from nicegui import ui

from enum import Enum


class ChartType(Enum):
    VEC = "vec"
    REW = "rew"
    MUSCLE = "muscle"
    TRAINING = "training"


class ChartFactory:
    @staticmethod
    def create_chart(chart_type: ChartType, title: str, x_label: str, y_label: str):
        fig = go.Figure()
        plot = ui.plotly(fig).style('flex: 1')
        if chart_type == ChartType.VEC:
            chart = Vec3Chart(fig, plot, title, x_label, y_label)
        elif chart_type == ChartType.REW:
            chart = RewChart(fig, plot, title, x_label, y_label)
        elif chart_type == ChartType.MUSCLE:
            chart = MuscleChart(fig, plot, title, x_label, y_label)
        elif chart_type == ChartType.TRAINING:
            chart = TrainingChart(fig, plot, title, x_label, y_label)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        return chart
