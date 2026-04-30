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
from msk_envs.envs.perturber import Perturber


class MSKEnv:
    """ Superclass for MSK environments """

    def _setup_model(self, env_config: EnvConfig) -> None:
        """ Modify model parameters here. """

        upper_body_muscles = [
            "coracobrachialis_r", "deltoideus_ant_r", "deltoideus_med_r", "deltoideus_post_r", "latissimus_sup_r",
            "latissimus_med_r", "latissimus_inf_r", "pectoralis_major_sup_r", "pectoralis_major_med_r",
            "pectoralis_major_inf_r", "teres_major_r", "teres_minor_r", "infraspinatus_inf_r", "infraspinatus_sup_r",
            "subscapularis_sup_r", "subscapularis_med_r", "subscapularis_inf_r", "supraspinatus_post_r",
            "supraspinatus_ant_r", "triceps_long_r", "biceps_long_r", "biceps_brevis_r", "coracobrachialis_l",
            "deltoideus_ant_l", "deltoideus_med_l", "deltoideus_post_l", "latissimus_sup_l", "latissimus_med_l",
            "latissimus_inf_l", "pectoralis_major_sup_l", "pectoralis_major_med_l", "pectoralis_major_inf_l",
            "teres_major_l", "teres_minor_l", "infraspinatus_inf_l", "infraspinatus_sup_l", "subscapularis_sup_l",
            "subscapularis_med_l", "subscapularis_inf_l", "supraspinatus_post_l", "supraspinatus_ant_l",
            "triceps_long_l", "biceps_long_l", "biceps_brevis_l",
        ]
        upper_body_muscles_id = [self.lookup_muscle_id(muscle_name) for muscle_name in upper_body_muscles]
        upper_body_muscles_id = [muscle_id for muscle_id in upper_body_muscles_id if muscle_id != -1]

        # Muscles activation and fiber dynamic
        msk_warp.set_activation_type(self.m, env_config.muscle_activation_dynamics)
        msk_warp.set_contraction_type(self.m, env_config.muscle_contraction_dynamics)
        muscle_metadata = msk_warp.muscle_metadata(self.m)
        muscle_idx_to_name = {idx: name for name, idx in self.load_result.muscle_id_lookup.items()}
        for i, mm in enumerate(muscle_metadata):
            muscle_name = muscle_idx_to_name[i]

            # Custom muscle multiplier
            if muscle_name in env_config.custom_muscle_multipliers:
                custom_multiplier = env_config.custom_muscle_multipliers[muscle_name]
                mm.max_isometric_force *= custom_multiplier
            else:
                mm.max_isometric_force *= env_config.muscle_multiplier

            mm.activation_time_const = env_config.muscle_activation_time_const
            mm.deactivation_time_const = env_config.muscle_deactivation_time_const
            mm.activation_dynamics_smoothing = env_config.muscle_activation_dynamics_smoothing
            mm.fiber_damping = env_config.muscle_fiber_damping
            mm.min_activation = env_config.muscle_min_activation
            mm.max_activation = env_config.muscle_max_activation
            mm.active_force_width_scale = env_config.muscle_active_force_width_scale
            mm.v_max = env_config.muscle_v_max

            if env_config.ignore_short_elastic_tendons:
                mm.ignore_tendon_compliance = (
                        mm.ignore_tendon_compliance or mm.tendon_slack_length < mm.optimal_fiber_length)

            # "Disable" passive muscle forces for upper body
            if i in upper_body_muscles_id:
                mm.strain_at_one_norm_force = 3.0
                mm.stiffness_at_low_force = 0.0

            if env_config.disable_muscle_passive_forces:
                mm.strain_at_one_norm_force = 3.0
                mm.stiffness_at_low_force = 0.0

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
        msk_warp.set_use_linear_stop(self.m, env_config.use_linear_stop)

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

            # FK graph: forward kinematics (positions only)
            with wp.ScopedCapture() as capture:
                msk_warp.fk(self.m, self.d)
            self.fk_graph = capture.graph

            # Reset graph: call after resetting any the worlds
            with wp.ScopedCapture() as capture:
                msk_warp.reset(self.m, self.d)
            self.reset_graph = capture.graph

            # Post-step graph: computes things like muscle passive forces, needed for rewards
            with wp.ScopedCapture() as capture:
                msk_warp.compute_muscle_passive_forces(self.m, self.d)
            self.post_graph = capture.graph

            # Analytics graph: anything else needed for analytics
            with wp.ScopedCapture() as capture:
                msk_warp.compute_muscle_moments(self.m, self.d)
                msk_warp.compute_net_joint_moments(self.m, self.d)
                msk_warp.compute_muscle_force_breakdown(self.m, self.d)
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
        self.load_result = load_result
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
        # [num_envs, num_colliders, 3]
        self.collider_positions = get_position_from_transform(msk_warp.get_collider_transforms(self.d))
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
        self.noise_root = env_config.noise_root
        self.enforce_ground_contact = env_config.enforce_ground_contact
        # Pre-compute left-right swap data
        self.swap_lr = env_config.swap_lr
        if self.swap_lr:
            self.swap_lr_data = get_swap_left_right_data(self.qpos_id_lookup, self.dof_id_lookup)
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

        # Set initial pose
        # reset_ind = torch.ones_like(self.reset_tensor, dtype=torch.bool)
        # self.noise_start_pose(reset_ind.ravel())

        # Set up random perturber
        self.perturber = Perturber(
            num_envs=num_envs,
            device=self.device,
            perturbation_duration=env_config.perturbation_duration,
            perturbation_frequency=env_config.perturbation_frequency,
            force_std=env_config.force_std,
            delta_t=self.delta_t,
            enabled=env_config.apply_perturbations,
        )
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
            if self.noise_root:
                q += torch.randn_like(q) * self.q_noise
            else:
                q[:, 7:] += torch.randn_like(q[:, 7:]) * self.q_noise
            qv += torch.randn_like(qv) * self.qv_noise

        # Swap left/right sides
        if self.swap_lr:
            # Randomly choose to swap left and right
            swap_mask = (torch.rand(self.num_worlds, device=q.device) > 0.5)
            q_old, qv_old = q.clone(), qv.clone()
            for swap_pair in self.swap_lr_data:
                rq, lq = swap_pair.qpos_r, swap_pair.qpos_l
                q[swap_mask, rq] = q_old[swap_mask, lq]
                q[swap_mask, lq] = q_old[swap_mask, rq]

                rdq, ldq = swap_pair.dof_r, swap_pair.dof_l
                qv[swap_mask, rdq] = qv_old[swap_mask, ldq]
                qv[swap_mask, ldq] = qv_old[swap_mask, rdq]

        # Set the new starting pose
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

        # Run forward kinematics to ensure contact with ground
        if self.enforce_ground_contact and self.root_free and self.collider_positions.size(1) > 1:
            self.fk()
            collider_heights = self.collider_positions[:, 1:, UP_IDX]
            lowest_collider_height = collider_heights.min(dim=1).values
            adjustment = -lowest_collider_height
            root_height_q_id = self.qpos_id_lookup["pelvis_ty"] if self.root_free else None
            self.joint_positions[reset_mask, root_height_q_id] += adjustment[reset_mask]

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

        # We don't let the mtp motor extend (negative). Map [-1, 1] to [0.5, 1]
        if "mtp_angle_r_motor" in self.actuator_id_lookup:
            mtp_angle_r_motor_id = self.actuator_id_lookup["mtp_angle_r_motor"]
            excitations[..., mtp_angle_r_motor_id] = excitations[..., mtp_angle_r_motor_id] * 0.5 + 0.5
        if "mtp_angle_l_motor" in self.actuator_id_lookup:
            mtp_angle_l_motor_id = self.actuator_id_lookup["mtp_angle_l_motor"]
            excitations[..., mtp_angle_l_motor_id] = excitations[..., mtp_angle_l_motor_id] * 0.5 + 0.5

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

        # Set actions
        self._set_actions(actions)

        if self.debug:
            assert not torch.isnan(actions).any(), "Actions contain NaN!"

        # Apply perturbations
        self.perturber.apply(self.root_id, self.body_user_forces)
        return

    def launch_sim_step(self):
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
        self.launch_sim_step()
        self.update_metrics()
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
            wp.synchronize()
        else:
            msk_warp.fk(self.m, self.d)

    def lookup_body_id(self, body_name: str) -> int:
        return self.body_id_lookup[body_name] if body_name in self.body_id_lookup else -1

    def lookup_dof_id(self, dof_name: str) -> int:
        return self.dof_id_lookup[dof_name] if dof_name in self.dof_id_lookup else -1

    def lookup_muscle_id(self, muscle_name: str) -> int:
        return self.muscle_id_lookup[muscle_name] if muscle_name in self.muscle_id_lookup else -1

    def scene_settings(self) -> SceneSettings:
        """ Override to provide custom scene settings for viewer/renderer """
        return SceneSettings()

    def additional_metrics(self) -> dict:
        """ Additional metrics to log during training """
        return {}

    def update_metrics(self) -> None:
        """ Hook to update metrics """
        return
