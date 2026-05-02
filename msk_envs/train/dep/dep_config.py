from dataclasses import dataclass


@dataclass
class DEPConfig:
    use_dep: bool = True
    """ Whether to use DEP """
    dep_horizon: int = 8
    """ Number of steps for DEP to perform when active """
    dep_p: float = 0.0037
    """ Probably of DEP being active at every step """
    dep_buffer_size: int = 60
    dep_bias_rate: float = 0.0066
    dep_kappa: float = 1000.0
    dep_tau: int = 12
    dep_s4avg: float = 1.0
    dep_regularization: float = 32
    dep_time_dist: int = 2
