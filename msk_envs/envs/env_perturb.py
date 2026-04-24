import torch

from msk_envs.utils.reward_lib import joint_penalty, \
    actuator_sq_penalty, metabolic_penalty, activation_square_penalty, has_fallen, alive_bonus, derivative_sq_penalty
from .env_base import MSKEnv
from .env_config import EnvConfig


class PerturbEnv(MSKEnv):
    """ Represents an env where the agent is constantly pushed """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            live_render: bool,
            requires_visuals: bool,
            cuda_graph: bool,
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            live_render=live_render,
            cuda_graph=cuda_graph
        )

        # How long perturbations last
        self.perturbation_range = (0.1, 0.3)      # Duration of perturbations
        self.perturbation_frequency = (0.25, 1.0)  # How often to wait between perturbations
        # Whether to apply perturbations this step
        self.perturbation_enabled = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        # Timers to track perturbation durations and wait times
        self.timer_target_duration = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.timer = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        # Standard deviation of force to apply
        self.force_std = 2.0
        return

    def sample_range(self, range_tuple: tuple) -> torch.Tensor:
        return torch.rand(self.num_worlds, device=self.device) * (range_tuple[1] - range_tuple[0]) + range_tuple[0]

    def _pre_step(self) -> None:
        # If any world are currently applying perturbations *and* have exceeded their perturbation duration
        worlds_done = (self.perturbation_enabled & (self.timer >= self.timer_target_duration))
        if torch.any(worlds_done):
            # Disable perturbations, reset timer
            self.perturbation_enabled[worlds_done] = False
            self.timer[worlds_done] = 0.0
            # Sample time to wait until next perturbation
            wait_times = self.sample_range(self.perturbation_frequency)
            self.timer_target_duration[worlds_done] = wait_times[worlds_done]

        # If any worlds are not currently applying perturbations *and* have exceeded their wait time
        worlds_start = (~self.perturbation_enabled & (self.timer >= self.timer_target_duration))
        if torch.any(worlds_start):
            # Enable perturbations, reset timer
            self.perturbation_enabled[worlds_start] = True
            self.timer[worlds_start] = 0.0
            # Sample perturbation durations
            perturb_durations = self.sample_range(self.perturbation_range)
            self.timer_target_duration[worlds_start] = perturb_durations[worlds_start]

            # Sample a new random external force for these worlds
            num_perturb = torch.sum(worlds_start).item()
            force_dir = torch.zeros((num_perturb, 3), device=self.device)
            force_magnitudes = torch.randn(num_perturb, device=self.device) * self.force_std
            force_directions = torch.randn((num_perturb, 3), device=self.device)
            force_directions = force_directions / torch.norm(force_directions, dim=1, keepdim=True)
            force_dir += force_directions * force_magnitudes.unsqueeze(1)
            # external_forces = 20 * force_dir
            # self.body_user_forces[worlds_start, self.root_id, 3:6] = external_forces

        # Make sure we reset forces for worlds not applying perturbations
        no_perturb_mask = ~self.perturbation_enabled
        self.body_user_forces[no_perturb_mask, :, :] = 0.0

        # Increment timers
        self.timer += self.delta_t
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
         5. External forces applied
        """
        obs = torch.cat([
            self.time.view(self.num_worlds, 1),
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            self.body_user_forces.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_alive = alive_bonus(self._get_terminated())
        # rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)
        # rew_actuator_dot = derivative_sq_penalty(self.actuator_activations_dot, self.num_actuators)
        # rew_fatigue = fatigue_penalty(self.muscle_activations, self.num_muscles)
        # rew_metabolic = metabolic_penalty(self.muscle_powers, self.num_muscles)

        # Joint passive penalty
        squared_penalties = True
        rew_spring = joint_penalty(self.ufrc_spring, squared=squared_penalties)
        rew_damper = joint_penalty(self.ufrc_damper, squared=squared_penalties)
        rew_limit = joint_penalty(self.ufrc_limit, squared=squared_penalties)
        rew_muscle_passive = joint_penalty(self.ufrc_muscle_passive, squared=squared_penalties)

        self.reward_dict = {
            "rew_alive": rew_alive,
            "rew_spring": rew_spring.detach(),
            "rew_damper": rew_damper.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
            # "rew_actuator": rew_actuator.detach(),
            # "rew_actuator_dot": rew_actuator_dot.detach(),
            # "rew_fatigue": rew_fatigue.detach(),
            # "rew_metabolic": rew_metabolic.detach(),
        }

    def _get_terminated(self):
        fallen = has_fallen(self.root_pos, self.head_pos, self.head_rot, self.head_offset)
        terminated = fallen.float()
        return terminated.detach()
