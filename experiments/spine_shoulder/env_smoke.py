"""Full-pipeline smoke test: build the ACTUAL StoneCourse env with the
spineshoulder model and step it with random actions. This is the authoritative
correctness+stability check (exercises ModelInitializer, contact params, obs,
reset, muscle dynamics, and the adaptive integrator exactly as training will).
"""
import numpy as np
import torch
import warp as wp

from msk_envs.envs.env_config import EnvConfigSprinterSpineShoulder
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import DerivedEnv

# StoneCourse task overrides (mirror StoneCourseConfig.__post_init__)
cfg = EnvConfigSprinterSpineShoulder()
cfg.env_variant = DerivedEnv.STONE_COURSE
cfg.starting_pose_path = "../msk_models/sprinter/poses/starting_pose_run.yaml"
cfg.delta_t = 1.0 / 30.0
cfg.max_episode_duration = 12.0
# StoneCourse reward lambdas (mirror StoneCourseConfig defaults)
cfg.reward_lambdas = {"lambda_vel": 1e-1, "lambda_alive": 1e-2,
                      "lambda_act": -1e-4, "lambda_metabolic": 0.0}

device = torch.device("cuda:0")
N = 8
print("building env...", flush=True)
env = EnvFactory.create_env(num_envs=N, env_config=cfg, requires_visuals=False,
                            cuda_graph=False, device=device)
print(f"env built. num_qpos={env.num_qpos} num_dofs={env.num_dofs} "
      f"num_muscles={env.num_muscles}", flush=True)
obs = env.reset()
obs_t = obs if isinstance(obs, torch.Tensor) else obs[0]
print(f"obs shape: {tuple(obs_t.shape)}  finite={torch.isfinite(obs_t).all().item()}", flush=True)
n_act = env.num_muscles + env.num_actuators
print(f"action dim (muscles+act): {n_act}", flush=True)

print("stepping 30 control steps with random actions...", flush=True)
for i in range(30):
    a = torch.rand((N, n_act), device=device)
    out = env.step(a)
    if i % 10 == 0:
        obs_i = out[0] if isinstance(out, (tuple, list)) else out
        fin = torch.isfinite(obs_i).all().item() if isinstance(obs_i, torch.Tensor) else True
        print(f"  step {i}: obs finite={fin}", flush=True)

qpos = wp.to_torch(env.d.qpos)
print(f"FINAL qpos finite across {N} envs: {torch.isfinite(qpos).all().item()}", flush=True)
print("ENV SMOKE:", "PASS" if torch.isfinite(qpos).all().item() else "FAIL", flush=True)
