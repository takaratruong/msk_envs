from dataclasses import dataclass


@dataclass
class DEPConfig:
    use_dep: bool = True
    dep_horizon: int = 8
    dep_init_steps: int = 256
    dep_p: float = 0.0037
    dep_buffer_size: int = 200
    dep_bias_rate: float = 0.002
    dep_kappa: float = 1000.0
    dep_tau: int = 40
    dep_s4avg: float = 2.0
    dep_regularization: float = 32
    dep_time_dist: int = 5
