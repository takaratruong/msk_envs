import torch
import math
import msk_warp
import warp as wp
import os

from .env_config import EnvConfig
from msk_envs.utils.pose import parse_starting_pose, get_swap_left_right_data
from msk_envs.utils.muscle_props import parse_starting_activations
from msk_envs.utils.contact_params import parse_contact_params
from msk_envs.utils.scene_settings import SceneSettings
from msk_envs.utils.transforms import get_position_from_transform, get_rotation_from_transform
from msk_envs.utils.global_params import UP_IDX, SIDE_IDX, FWD_IDX, build_axis


class MSKEnv:
    """ Superclass for MSK environments """

    def _setup_model(self, env_config: EnvConfig) -> None:
        """ Modify model parameters here. """
        # Muscles activation and fiber dynamic
        msk_warp.set_activation_type(self.m, env_config.muscle_activation_dynamics)
        msk_warp.set_contraction_type(self.m, env_config.muscle_contraction_dynamics)
        muscle_metadata = msk_warp.muscle_metadata(self.m)
        for mm in muscle_metadata:
            mm.max_isometric_force *= env_config.muscle_multiplier
            mm.activation_time_const = env_config.muscle_activation_time_const
            mm.deactivation_time_const = env_config.muscle_deactivation_time_const
            mm.activation_dynamics_smoothing = env_config.muscle_activation_dynamics_smoothing
            mm.fiber_damping = env_config.muscle_fiber_damping
            mm.min_activation = env_config.muscle_min_activation
            mm.max_activation = env_config.muscle_max_activation
            mm.v_max = env_config.muscle_v_max

        # Collider properties
        if env_config.use_specified_contact_params:
            geom_stiffness = msk_warp.collider_stiffness(self.m)
            geom_dissipation = msk_warp.collider_dissipation(self.m)
            geom_priority = msk_warp.collider_priority(self.m)
            geom_friction = msk_warp.collider_friction(self.m)
            geom_transition_velocity = msk_warp.collider_transition_velocity(self.m)

            contact_params_path = os.path.join(self.curr_path, env_config.contact_params_path)
            contact_params = parse_contact_params(contact_params_path, self.collider_id_lookup)
            for params in contact_params:
                geom_id = params.geom_id
                geom_stiffness[geom_id] = params.stiffness
                geom_dissipation[geom_id] = params.dissipation
                geom_priority[geom_id] = params.priority
                geom_friction[geom_id][0] = params.static_friction
                geom_friction[geom_id][1] = params.dynamic_friction
                geom_friction[geom_id][2] = params.viscous_friction
                geom_transition_velocity[geom_id] = params.transition_velocity

        # Armature
        dof_start = 6 if self.root_free else 0
        msk_warp.armature(self.m)[dof_start:] = env_config.armature

        msk_warp.set_implicit_damping(self.m, env_config.use_implicit_damping)

        # Integrator type
        msk_warp.set_integrator_accuracy(self.m, env_config.integrator_accuracy)
        msk_warp.set_integrator_use_inf_norm(self.m, env_config.integrator_use_inf_norm)
        # Toggle drag forces
        msk_warp.set_drag_enabled(self.m, env_config.enable_drag)

        msk_warp.reinitialize_model(self.m, self.d)
        return

    def _setup_cuda_graphs(self):
        if self.cuda_graph:
            assert torch.cuda.is_available()
            # Step graph
            with wp.ScopedCapture() as capture:
                msk_warp.step(self.m, self.d)
            self.step_graph = capture.graph

            # Reset graph: call after resetting any the worlds
            with wp.ScopedCapture() as capture:
                msk_warp.reset(self.m, self.d)
            self.reset_graph = capture.graph

            # Post-step graph: computes things like joint torques. Needed for rewards, observations, etc.
            with wp.ScopedCapture() as capture:
                msk_warp.post(self.m, self.d)
            self.post_graph = capture.graph

            # Analytics graph: anything else needed for analytics
            with wp.ScopedCapture() as capture:
                msk_warp.compute_muscle_moments(self.m, self.d)
                msk_warp.compute_net_joint_moments(self.m, self.d)
            self.analytics_graph = capture.graph

            # Forward kinematics graph, useful for motion tracking
            with wp.ScopedCapture() as capture:
                msk_warp.fk(self.m, self.d)
            self.fk_graph = capture.graph
        return

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool,
            debug: bool = False
    ):
        self.num_worlds = num_envs
        self.device = device
        self.debug = debug

        # Load model
        self.curr_path = os.path.abspath(os.path.dirname(__file__))
        self.model_path = os.path.join(self.curr_path, env_config.model_path)
        function_path = os.path.join(
            self.curr_path, env_config.muscle_function_path) if env_config.use_function_based_path else None
        load_result = msk_warp.load_model(
            model_path=self.model_path,
            n_worlds=num_envs,
            integrator=env_config.integrator,
            requires_visuals=requires_visuals,
            polynomial_data_path=function_path
        )
        self.m, self.d = load_result.model, load_result.data
        self.root_free = load_result.root_free
        # Store all convenient lookups
        self.body_id_lookup = load_result.body_id_lookup
        self.dof_id_lookup = load_result.dof_id_lookup
        self.qpos_id_lookup = load_result.qpos_id_lookup
        self.limit_id_lookup = load_result.limit_id_lookup
        self.muscle_id_lookup = load_result.muscle_id_lookup
        self.actuator_id_lookup = load_result.actuator_id_lookup
        self.collider_id_lookup = load_result.collider_id_lookup
        self.visuals = load_result.mesh_load_results
        self._setup_model(env_config)

        # Model properties
        self.num_qpos = msk_warp.get_num_qpos(self.m)
        self.num_dofs = msk_warp.get_num_dofs(self.m)
        self.num_bodies = msk_warp.get_num_bodies(self.m)
        self.num_muscles = msk_warp.get_num_muscles(self.m)
        self.num_actuators = msk_warp.get_num_actuators(self.m)
        self.num_colliders = msk_warp.get_num_colliders(self.m)
        # [num_envs, num_bodies]
        self.body_mass = msk_warp.body_mass(self.m)
        self.total_mass = self.body_mass.sum()
        self.gravity = msk_warp.gravity(self.m)

        # Data properties. The following are all references
        # [num_envs]
        self.time = msk_warp.time(self.d)
        # [num_envs, num_muscles]
        self.muscle_activations = msk_warp.muscle_activations(self.d)
        self.muscle_excitations = msk_warp.muscle_excitations(self.d)
        self.muscle_fiber_lengths = msk_warp.muscle_fiber_lengths(self.d)
        self.muscle_fiber_velocities = msk_warp.muscle_fiber_velocities(self.d)
        self.muscle_powers = msk_warp.muscle_powers(self.d)
        # [num_envs, num_actuators]
        self.actuator_activations = msk_warp.actuator_activations(self.d)
        self.actuator_activations_dot = msk_warp.actuator_activations_dot(self.d)
        self.actuator_excitations = msk_warp.actuator_excitations(self.d)
        # [num_envs, num_bodies, 7], ignore ground
        self.body_transforms = msk_warp.body_transforms(self.d)
        # [num_envs, num_bodies, 3], ignore ground
        self.body_positions = msk_warp.body_com_positions(self.d)
        # [num_envs, num_bodies, 4] (w, x, y, z)
        self.body_rotations = get_rotation_from_transform(self.body_transforms)
        # [num_envs, num_bodies, 6] (ang, lin)
        self.body_velocities = msk_warp.body_velocities(self.d)
        self.body_accelerations = msk_warp.body_accelerations(self.d)
        # [num_envs, num_bodies, 6] (frc, trq)
        self.body_user_forces = msk_warp.body_user_forces(self.d)
        # [num_envs, num_qpos]
        self.joint_positions = msk_warp.joint_positions(self.d)
        # [num_envs, num_dofs]
        self.joint_velocities = msk_warp.joint_velocities(self.d)
        # [num_envs, num_dofs]
        self.ufrc_spring = msk_warp.ufrc_spring(self.d)
        self.ufrc_damper = msk_warp.ufrc_damper(self.d)
        self.ufrc_limit = msk_warp.ufrc_limit(self.d)
        self.ufrc_muscle_passive = msk_warp.ufrc_muscle_passive(self.d)
        # [num_envs, num_colliders]
        self.collider_forces = msk_warp.collider_forces(self.d)
        # self.collider_self_forces = msk_warp.collider_self_forces(self.d)
        self.body_self_collision_forces = msk_warp.body_self_collisions(self.d)

        # [num_envs, 3]
        self.grf = msk_warp.grf(self.d)

        # [num_envs, num_visuals, 3]
        self.visual_transforms = msk_warp.get_visual_transforms(self.d)
        self.visual_positions = get_position_from_transform(self.visual_transforms)
        # [num_envs, num_visuals, 4]
        self.visual_rotations = get_rotation_from_transform(self.visual_transforms)

        # RL Environment metadata
        self.action_range = (-1.0, 1.0)
        self.max_episode_duration = env_config.max_episode_duration
        self.delta_t = env_config.delta_t
        self.reset_tensor = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)

        # Simulation steps required to reach env step. if adaptive, just step to desired time
        #  otherwise step in increments of [delta_t_sim]
        self.is_adaptive = msk_warp.is_adaptive(env_config.integrator)
        if self.is_adaptive:
            self.delta_t_sim = self.delta_t
            self.sim_steps_per_env_step = 1
        else:
            self.delta_t_sim = env_config.delta_t_sim
            self.sim_steps_per_env_step = math.ceil(self.delta_t / self.delta_t_sim)

        # Starting position, load from file
        start_pose_path = os.path.join(self.curr_path, env_config.starting_pose_path)
        q, qv = parse_starting_pose(
            start_pose_path, self.qpos_id_lookup, self.dof_id_lookup, self.num_qpos, self.num_dofs)
        assert len(q) == self.num_qpos and len(qv) == self.num_dofs
        q_torch = torch.tensor(q, dtype=torch.float32, device=device)
        qv_torch = torch.tensor(qv, dtype=torch.float32, device=device)
        self.start_pose_base = q_torch.unsqueeze(0)
        self.start_velocity_base = qv_torch.unsqueeze(0)
        # Repeat for all envs
        self.start_pose = q_torch.unsqueeze(0).repeat(num_envs, 1)
        self.start_velocity = qv_torch.unsqueeze(0).repeat(num_envs, 1)
        # Pose noise/reset settings
        self.noise_start = env_config.noise_start
        self.q_noise = env_config.q_noise
        self.qv_noise = env_config.qv_noise
        # Pre-compute left-right swap data
        self.swap_lr = env_config.swap_lr
        if self.swap_lr:
            self.swap_lr_data = get_swap_left_right_data(self.m, self.body_id_lookup)
        else:
            self.swap_lr_data = []

        # Starting muscle activations
        self.noise_act_start = False
        if env_config.use_prescribed_starting_activations:
            start_activations_path = os.path.join(self.curr_path, env_config.starting_activations_path)
            start_activations = parse_starting_activations(
                start_activations_path, self.muscle_id_lookup, env_config.default_activation)
            self.start_activations = torch.tensor(start_activations, device=device).unsqueeze(0).repeat(num_envs, 1)
        else:
            if env_config.default_activation == -1.0:
                self.start_activations = torch.rand((num_envs, self.num_muscles), device=device)
                self.noise_act_start = True
            else:
                self.start_activations = torch.ones(
                    (num_envs, self.num_muscles), device=device) * env_config.default_activation

        # Rewards storage
        self.reward_dict = {}
        self.reward_lambdas = env_config.reward_lambdas

        self.render = live_render
        if live_render:
            self.renderer = msk_warp.create_renderer(
                load_result=load_result,
                renderer_type=msk_warp.RendererType.OPENGL,
                draw_visuals=True,
                draw_beams=True,
                draw_body_mass=False,
                draw_colliders=False,
                draw_muscles=True,
                draw_sites=False,
            )
            if self.renderer.viewer_type == msk_warp.RendererType.TILED:
                self.renderer.setup_tiled_renderer(list(range(min(num_envs, 4))))

        # CUDA Graphs
        self.cuda_graph = cuda_graph
        self._setup_cuda_graphs()

        # Pre-compute useful body IDs and offsets
        self.ground_id = self.lookup_body_id("ground")
        self.root_id = self.lookup_body_id("pelvis")
        self.head_id = self.lookup_body_id("head")
        head_offset = torch.zeros(3, device=self.device)

        # Head isn't its own body, compute offset from torso
        if self.head_id == -1:
            self.head_id = self.lookup_body_id("torso")
            head_offset = torch.tensor(build_axis(axis=UP_IDX, scale=0.215), device=self.device)

        self.root_pos = self.body_positions[:, self.root_id]
        self.root_rot = self.body_rotations[:, self.root_id]
        self.head_pos = self.body_positions[:, self.head_id]
        self.head_rot = self.body_rotations[:, self.head_id]
        self.head_offset = head_offset.unsqueeze(0).repeat(num_envs, 1)

        # Finally, set initial pose
        reset_ind = torch.ones_like(self.reset_tensor, dtype=torch.bool)
        self.noise_start_pose(reset_ind.ravel())
        return

    def noise_start_pose(self, reset_mask: torch.Tensor) -> None:
        """
        Re-noise the starting pose and velocity for envs where reset_mask is 1
        Note: takes effect on next reset or init
        """
        # Repeat for all envs
        q = self.start_pose_base.repeat(self.num_worlds, 1)
        qv = self.start_velocity_base.repeat(self.num_worlds, 1)

        # Noise starting pose
        if self.noise_start:
            q[7:] += torch.randn_like(q[7:]) * self.q_noise
            qv += torch.randn_like(qv) * self.qv_noise

        if self.swap_lr:
            # Determine which envs to swap the starting pose
            swap_mask = (torch.rand(self.num_worlds, device=q.device) > 0.5)
            q_old, qv_old = q.clone(), qv.clone()
            for swap_pair in self.swap_lr_data:
                rq, lq, nq = swap_pair.start_qpos_r, swap_pair.start_qpos_l, swap_pair.num_qpos
                q[swap_mask, rq:rq + nq] = q_old[swap_mask, lq:lq + nq]
                q[swap_mask, lq:lq + nq] = q_old[swap_mask, rq:rq + nq]

                rv, lv, nv = swap_pair.start_dof_r, swap_pair.start_dof_l, swap_pair.num_dof
                qv[swap_mask, rv:rv + nv] = qv_old[swap_mask, lv:lv + nv]
                qv[swap_mask, lv:lv + nv] = qv_old[swap_mask, rv:rv + nv]

        self.start_pose[reset_mask, :] = q[reset_mask, :]
        self.start_velocity[reset_mask, :] = qv[reset_mask, :]

        # Noise starting muscle activations
        if self.noise_act_start:
            random_acts = torch.rand((self.num_worlds, self.num_muscles), device=self.device)
            self.start_activations[reset_mask, :] = random_acts[reset_mask, :]
        return

    def num_obs(self) -> int:
        return self._get_obs().shape[1]

    def num_actions(self) -> int:
        return self._get_actions().shape[1]

    def get_time(self) -> torch.Tensor:
        return self.time.detach().clone()

    def get_joint_passive_torques(self, include_passive_muscle: bool = True) -> torch.Tensor:
        """ Returns the net torques from the passive elements of each joint """
        passive_joint_torques = torch.abs(self.ufrc_spring) + torch.abs(self.ufrc_damper) + torch.abs(self.ufrc_limit)
        if include_passive_muscle:
            passive_joint_torques += torch.abs(self.ufrc_muscle_passive)
        return passive_joint_torques

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        """ Hook for additional reset behavior in subclasses. Occurs before sim reset """
        return

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        """ Hook for additional reset behavior in subclasses. Occurs after sim reset """
        return

    def _reset_sim(self):
        # Reset time, starting pose, muscle and actuator activations
        reset_mask = self.reset_tensor.squeeze(-1).bool()
        if reset_mask.any():
            self.time[reset_mask] = 0.0
            self.joint_positions[reset_mask, :] = self.start_pose[reset_mask, :]
            self.joint_velocities[reset_mask, :] = self.start_velocity[reset_mask, :]
            self.muscle_activations[reset_mask, :] = self.start_activations[reset_mask, :]
            self.actuator_activations[reset_mask, :] = 0.5

        # Reset sim
        msk_warp.set_reset(self.d, self.reset_tensor)
        if self.cuda_graph:
            wp.capture_launch(self.reset_graph)
            wp.synchronize()
        else:
            msk_warp.reset(self.m, self.d)
        return

    # The following are environment-specific and need to be implemented
    def _get_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_actions(self) -> torch.Tensor:
        actions = torch.cat([
            self.muscle_excitations,
            self.actuator_excitations
        ], dim=1)
        return actions.detach().clone()

    def _set_actions(self, raw_action) -> None:
        self._set_muscle_excitations(raw_action[:, :self.num_muscles])
        self._set_actuator_excitations(raw_action[:, self.num_muscles:])
        return

    def _pre_step(self) -> None:
        """ Hook for any pre-step computations """
        return

    def _compute_raw_reward_dict(self):
        """ Guarantee to only run once per step """
        raise NotImplementedError

    def _get_terminated(self):
        raise NotImplementedError

    # Rest is standard
    def get_blank_actions(self) -> torch.Tensor:
        """ Returns actions of all 0 in correct shape """
        return torch.zeros_like(self._get_actions())

    def get_random_actions(self) -> torch.Tensor:
        """ Returns randomly uniform actions in correct shape """
        return (torch.rand_like(self._get_actions()) *
                self.action_range[1] * 2 - self.action_range[1])

    def get_scaled_reward_dict(self) -> dict:
        reward_dict = self.reward_dict
        scaled_reward_dict = {}
        for key, raw_value in reward_dict.items():
            lambda_key = key.replace("rew_", "lambda_")
            lambda_value = self.reward_lambdas[lambda_key]
            scaled_reward_dict[key] = lambda_value * raw_value
        return scaled_reward_dict

    def get_rewards(self) -> torch.Tensor:
        scaled_rew_dict = self.get_scaled_reward_dict()
        total_rewards = torch.zeros(self.num_worlds, device=self.device)
        for key, value in scaled_rew_dict.items():
            total_rewards += value
        return total_rewards.detach()

    def _get_truncated(self):
        truncated = (self.get_time() >= self.max_episode_duration).float()
        return truncated.detach()

    def _set_muscle_excitations(self, raw_action) -> None:
        # Clamp to [-1, 1], then map to [0, 1]
        clamped_action = torch.clamp(raw_action, -1.0, 1.0)
        excitation = (clamped_action + 1.0) / 2.0
        self.muscle_excitations.copy_(excitation)
        return

    def _set_actuator_excitations(self, raw_action) -> None:
        # Clamp to [-1, 1], then map to [0, 1]
        clamped_action = torch.clamp(raw_action, -1.0, 1.0)
        excitations = (clamped_action + 1.0) / 2.0
        self.actuator_excitations.copy_(excitations)
        return

    def _perform_reset(self, reset_mask: torch.Tensor):
        """ Internal reset call, resets only envs where reset_mask is 1 """
        self.reset_tensor.copy_(reset_mask)
        self.noise_start_pose(reset_mask.squeeze(-1).bool())
        self._upon_reset_pre_sim(reset_mask.squeeze(-1).bool())
        self._reset_sim()
        self._upon_reset_post_sim(reset_mask.squeeze(-1).bool())
        self.reset_tensor.fill_(0.0)

    # The following impl of step is kinda jank, but we need this separation for logging
    def pre_step(self, actions) -> None:
        self._pre_step()
        self._set_actions(actions)
        if self.debug:
            assert not torch.isnan(actions).any(), "Actions contain NaN!"
        return

    def launch_step(self):
        if self.cuda_graph:
            for _ in range(self.sim_steps_per_env_step):
                msk_warp.increment_next_time(self.m, self.d, self.delta_t_sim)
                wp.capture_launch(self.step_graph)
            wp.capture_launch(self.post_graph)
            wp.synchronize()
        else:
            for _ in range(self.sim_steps_per_env_step):
                msk_warp.increment_next_time(self.m, self.d, self.delta_t_sim)
                msk_warp.step(self.m, self.d)
            msk_warp.post(self.m, self.d)
        return

    def rl_step(self):
        # Only compute reward dict once per step
        self._compute_raw_reward_dict()

        final_obs = self._get_obs()
        rew = self.get_rewards()
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # Reset any worlds that are done
        done = torch.clamp(terminated + truncated, 0.0, 1.0).unsqueeze(-1)
        if done.any():
            self._perform_reset(done)

        # Training requires the observation *after* the reset (for next action)
        obs = self._get_obs()
        # Return raw reward terms for logging
        info = {
            "final_observation": final_obs,
            "raw_rewards": self.reward_dict,
            "scaled_rewards": self.get_scaled_reward_dict(),
        }

        if self.debug:
            assert not torch.isnan(obs).any(), "Observations contain NaN!"
            assert not torch.isnan(rew).any(), "Rewards contain NaN!"

        if self.render and hasattr(self.renderer, 'meshes') and len(
                self.renderer.meshes) > 0:
            self.renderer.render(self.m, self.d)

        return obs, rew, terminated, truncated, info

    def step(self, actions):
        """ External step call """
        self.pre_step(actions)
        self.launch_step()
        return self.rl_step()

    def reset(self):
        """ External reset call, resets all envs """
        self._perform_reset(reset_mask=torch.ones_like(self.reset_tensor))
        obs = self._get_obs()
        return obs

    def fk(self):
        """ Forward kinematics only (only position dependent) """
        if self.cuda_graph:
            wp.capture_launch(self.fk_graph)
        else:
            msk_warp.fk(self.m, self.d)

    def lookup_body_id(self, body_name: str) -> int:
        return self.body_id_lookup[body_name] if body_name in self.body_id_lookup else -1

    def scene_settings(self) -> SceneSettings:
        """ Override to provide custom scene settings for viewer/renderer """
        return SceneSettings()
