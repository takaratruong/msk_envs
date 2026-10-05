import torch, warp as wp
from msk_envs.envs.env_config import EnvConfigSprinter, DerivedEnv
from msk_envs.envs.env_factory import EnvFactory
cfg = EnvConfigSprinter()
cfg.model_path = "../msk_models/sprinter/sprinter_model_sym.osim"
cfg.env_variant = DerivedEnv.STONE_COURSE
cfg.starting_pose_path = "../msk_models/sprinter/poses/starting_pose_run.yaml"
cfg.delta_t = 1.0/30.0; cfg.max_episode_duration = 12.0
cfg.reward_lambdas = {"lambda_vel":1e-1,"lambda_alive":1e-2,"lambda_act":-1e-4,"lambda_metabolic":0.0}
env = EnvFactory.create_env(num_envs=8, env_config=cfg, requires_visuals=False, cuda_graph=False, device=torch.device("cuda:0"))
print(f"BASE env built nq={env.num_qpos}", flush=True)
env.reset()
n = env.num_muscles + env.num_actuators
import time; t0=time.time()
for i in range(30):
    env.step(torch.rand((8,n),device="cuda:0"))
    if i%10==0: print(f"  base step {i} t={time.time()-t0:.1f}s", flush=True)
q=wp.to_torch(env.d.qpos)
print("BASE SMOKE:", "PASS" if torch.isfinite(q).all().item() else "FAIL", f"total={time.time()-t0:.1f}s", flush=True)
