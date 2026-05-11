import torch
import msk_warp
import warp as wp

from msk_envs.utils.global_params import FWD_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class BlockStartSprintingEnv(LanesEnv):
    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            live_render=live_render,
            cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, 1.0),
        )
        return

    def _add_colliders(self, env_config: EnvConfig) -> None:
        colliders = self.load_result.colliders
        starting_block1 = msk_warp.UserGeomData(
            name="starting_block1",
            body_name=msk_warp.GROUND,
            geom_type=msk_warp.GeomType.CAPSULE,
            transform=wp.transform(wp.vec3(-0.3, 0.0, 0.1), wp.quat_identity(dtype=float)),
            size=wp.vec3(0.1, 0.05, 0.1),
            priority=9,
        )
        starting_block2 = msk_warp.UserGeomData(
            name="starting_block2",
            body_name=msk_warp.GROUND,
            geom_type=msk_warp.GeomType.CAPSULE,
            transform=wp.transform(wp.vec3(-0.73, 0.0, -0.1), wp.quat_identity(dtype=float)),
            size=wp.vec3(0.1, 0.05, 0.1),
            priority=9,
        )
        colliders.append(msk_warp.convert_user_collider(starting_block1))
        colliders.append(msk_warp.convert_user_collider(starting_block2))
        return
