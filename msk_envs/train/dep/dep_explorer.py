import random
import torch
from . import DEP, DEPConfig


class DEPExplorer:
    def __init__(
            self,
            dep_config: DEPConfig,
            num_muscles: int,
            num_worlds: int,
            device: torch.device,
            use_dep: bool
    ):
        self.dep = DEP(
            n_motors=num_muscles,
            n_envs=num_worlds,
            buffer_size=dep_config.dep_buffer_size,
            bias_rate=dep_config.dep_bias_rate,
            kappa=dep_config.dep_kappa,
            tau=dep_config.dep_tau,
            s4avg=dep_config.dep_s4avg,
            regularization=dep_config.dep_regularization,
            time_dist=dep_config.dep_time_dist,
            with_learning=True,
            device=device
        )
        self.in_dep = False
        self.in_dep_counter = 0
        self.dep_horizon = dep_config.dep_horizon
        self.dep_p = dep_config.dep_p
        self.use_dep = use_dep

    def explore(self, muscle_states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self.use_dep:
            return actions

        modified_actions = actions.clone()

        # important for dep to keep learning, even if we don't use the actions
        dep_actions = self.dep.step(muscle_states)

        # Randomly decide whether to activate DEP
        if not self.in_dep and random.random() < self.dep_p:
            self.in_dep = True
            self.in_dep_counter = 0

        # In dep: replace muscle actions with DEP actions
        if self.in_dep:
            num_muscles = self.dep.num_motors
            modified_actions[:, :num_muscles] = dep_actions
            self.in_dep_counter += 1

        # Toggle off dep after horizon
        if self.in_dep_counter >= self.dep_horizon:
            self.in_dep = False
            self.in_dep_counter = 0

        return modified_actions
