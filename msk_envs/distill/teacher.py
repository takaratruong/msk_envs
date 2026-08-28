import torch
import bolt
from msk_envs.train.nets.deterministic_policy import load_policy
from msk_envs.distill.dof_utils import joint_dof_indices

class Teacher:
    def __init__(self, ckpt_path, env, device):
        self.policy = load_policy(ckpt_path); self.policy.to(device)
        self.names, self.idx = joint_dof_indices(env)
        self.idx_t = torch.tensor(self.idx, device=device, dtype=torch.long)

    @torch.no_grad()
    def action(self, obs):
        return self.policy(obs)                 # muscle excitations, (n_envs, num_muscles)

    def torque_label(self, env):
        ufrc_muscle = bolt.ufrc_muscle(env.d)   # (n_envs, num_dofs)
        return ufrc_muscle.index_select(1, self.idx_t).clone()   # (n_envs, 25)

if __name__ == "__main__":
    import torch
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinter
    from msk_envs.distill.dof_utils import joint_dof_indices

    dev = torch.device("cuda")
    env = EnvFactory.create_env(num_envs=64, env_config=EnvConfigSprinter(),
                                requires_visuals=False, cuda_graph=True, device=dev)
    CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
    t = Teacher(CKPT, env, dev)

    # Verify joint DoF count
    names, idx = joint_dof_indices(env)
    print(f"Found {len(names)} non-root joint DoFs: {names}")

    obs = env.reset()
    for _ in range(20):
        a = t.action(obs)
        # Step physics without computing rewards
        env.pre_sim_step(a)
        env.launch_sim_step()
        obs = env._get_obs()

    lab = t.torque_label(env)
    print(f"label shape {tuple(lab.shape)} mean|tau|={lab.abs().mean().item():.3f}")
    assert lab.shape[1] == 25 and torch.isfinite(lab).all()
    print("Smoke test PASSED")
