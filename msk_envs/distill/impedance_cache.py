"""Task RL-T2 — Offline impedance oracle (amortized teacher impedance for the RL reward).

WHY AN OFFLINE ORACLE
---------------------
The PD-RL env (`SprinterPDRLEnv`) rewards the policy for emitting per-joint gains (kp,kd)
close to the TEACHER's extracted impedance at the current state (the "muscle-as-constraint"
signal). The ground-truth impedance comes from `extract_impedance` (distill/impedance.py),
which finite-differences the muscle net joint torque `ufrc_muscle` via `forward.fwd`.

Two hard facts force amortization:
  1. COST: `extract_impedance` does 25 joints x 4 forward.fwd calls (=100 fwd) PER STATE. That
     is far too slow to run inside the RL reward loop (which runs every env step over hundreds
     of parallel worlds).
  2. MUSCLES REQUIRED: `extract_impedance` only works on a MUSCLE-DRIVEN env where the teacher
     is active (it reads `ufrc_muscle`). The PD-RL env runs with muscles OFF, so the ground-truth
     impedance is simply NOT AVAILABLE at RL time.

Solution: build the oracle ONCE, OFFLINE, on a teacher rollout of the muscle twin env
(`EnvConfigSprinterTorquePD`, muscles active). We collect (obs_359, kp[25], kd[25]) over a few
thousand teacher-driven states and fit a small MLP  obs(359) -> (log kp[25], log kd[25]).
At RL time we just call `oracle(obs)` — a single cheap MLP forward per step.

OBS DOMAIN MATCH (verified): the muscle-rollout env `EnvConfigSprinterTorquePD` and the RL env
`EnvConfigSprinterTorquePD_RL` share the SAME 359-dim obs (`env._get_obs()`). So the oracle is
keyed on the FULL 359 obs — no adapter for the oracle's INPUT. (The teacher's muscle policy still
needs the 336-col adapter `build_teacher_obs_cols` to be *driven*; that only affects how we roll
the teacher, not the oracle's input space.)

COST ACCOUNTING
---------------
  * OFFLINE (this script, run once):  N_states x (25 joints x 4 fwd) forward passes of the muscle
    model + a short MLP fit. With ~256 envs x tens of steps this is thousands of labeled states in
    a couple of minutes on one GPU. Done ONCE; the .pt is cached (gitignored under models/).
  * RL-TIME (per env step): ONE OracleMLP forward (2 hidden layers) over (num_worlds, 359).
    Negligible next to the physics step. No forward.fwd, no muscle model — works with muscles OFF.

TARGET / EPS CONVENTION
-----------------------
Targets are log(kp.clamp_min(EPS)) and log(kd.clamp_min(EPS)) with EPS = ORACLE_EPS (below).
The RL reward MUST use the SAME EPS when it logs the policy's gains, so the two log-spaces align.
"""
import os
import time
import argparse

import torch
import torch.nn as nn

# NOTE: extract_impedance / Teacher / env are imported LAZILY inside build_oracle so that merely
# importing this module (e.g. from env_pd_rl.py to grab OracleMLP/ORACLE_PATH/ORACLE_EPS) does not
# pull in the whole env stack or require a GPU.

ORACLE_PATH = "/home/ubuntu/msk_envs/models/dagger_pd/impedance_oracle.pt"
CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"

ORACLE_EPS = 1e-3   # floor before log(); shared by the oracle targets AND the RL reward term.
N_DOF = 25


class OracleMLP(nn.Module):
    """obs(n_obs=359) -> (log kp[25], log kd[25]) concatenated as a 50-vector.

    Predicts impedance in LOG space (kp is O(1e3), kd is O(1); log tames the dynamic range so a
    plain MSE weights both blocks comparably). Downstream consumers split the 50-vector at N_DOF.
    """

    def __init__(self, n_obs, n_dof=N_DOF, hidden=(256, 256), device="cuda"):
        super().__init__()
        dims = [n_obs, *hidden]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b, device=device), nn.ReLU()]
        layers += [nn.Linear(dims[-1], 2 * n_dof, device=device)]
        self.net = nn.Sequential(*layers)
        self.n_dof = n_dof

    def forward(self, obs):
        out = self.net(obs)                       # (n, 50)
        return out[:, : self.n_dof], out[:, self.n_dof:]   # (log_kp[25], log_kd[25])


def load_oracle(path=ORACLE_PATH, device="cuda"):
    """Load a saved oracle. Returns (model_in_eval_mode, meta_dict). Raises if the file is absent."""
    ckpt = torch.load(path, map_location=device)
    model = OracleMLP(ckpt["n_obs"], n_dof=ckpt.get("n_dof", N_DOF),
                      hidden=tuple(ckpt.get("hidden", (256, 256))), device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


@torch.no_grad()
def collect_teacher_impedance(env, teacher, teacher_cols, n_musc, n_steps, warmup=10):
    """Roll the teacher on the muscle twin env; every step record (obs_359, kp[25], kd[25]).

    Reuses `teacher_probe_step` (advance under muscles, PD buffers zeroed) then `extract_impedance`
    at the teacher-driven state — the exact labeling pattern used by dagger_pd.
    Returns (O[N,359], KP[N,25], KD[N,25]) on `env`'s device.
    """
    from msk_envs.distill.dagger_pd import teacher_probe_step
    from msk_envs.distill.impedance import extract_impedance

    env.reset()
    for _ in range(warmup):
        teacher_probe_step(env, teacher, teacher_cols, n_musc)

    O, KP, KD = [], [], []
    for _ in range(n_steps):
        teacher_probe_step(env, teacher, teacher_cols, n_musc)
        obs = env._get_obs()                         # (n, 359) — oracle's input domain
        _, kp, kd = extract_impedance(env, teacher)  # (n, 25) each, names order
        O.append(obs.detach().clone())
        KP.append(kp.detach().clone())
        KD.append(kd.detach().clone())
    return torch.cat(O), torch.cat(KP), torch.cat(KD)


def fit_oracle(model, O, log_tgt, epochs, device, lr=1e-3, mb=4096):
    """80/20 train/val split, Adam, MSE in log space on the 50-dim target. Returns final val MSE."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = O.shape[0]
    idx = torch.randperm(n, device=device)
    n_val = max(1, n // 5)
    vi, ti = idx[:n_val], idx[n_val:]
    for _ in range(epochs):
        perm = ti[torch.randperm(ti.shape[0], device=device)]
        for b in perm.split(mb):
            opt.zero_grad()
            lkp, lkd = model(O[b])
            pred = torch.cat([lkp, lkd], dim=1)
            loss = ((pred - log_tgt[b]) ** 2).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        lkp, lkd = model(O[vi])
        pred = torch.cat([lkp, lkd], dim=1)
        val = ((pred - log_tgt[vi]) ** 2).mean().item()
    return val


def build_oracle(n_envs=256, n_steps=32, warmup=10, epochs=40, out_path=ORACLE_PATH,
                 device="cuda"):
    """Build + save the offline impedance oracle. Returns (model, meta, stats)."""
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinterTorquePD
    from msk_envs.distill.teacher import Teacher
    from msk_envs.distill.dof_utils import joint_dof_indices, build_teacher_obs_cols

    dev = torch.device(device)
    # distill/impedance envs ship without reward lambdas; provide blank ones (no reward used here).
    from msk_envs.distill.dagger_pd import BLANK_REWARD_LAMBDAS
    cfg = EnvConfigSprinterTorquePD()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    env = EnvFactory.create_env(num_envs=n_envs, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)

    names, _ = joint_dof_indices(env)
    teacher = Teacher(CKPT, env, dev)
    assert names == teacher.names, "names order mismatch vs teacher"
    teacher_cols = build_teacher_obs_cols(env, dev)
    n_musc = env.num_muscles

    t0 = time.time()
    O, KP, KD = collect_teacher_impedance(env, teacher, teacher_cols, n_musc, n_steps, warmup)
    t_collect = time.time() - t0
    n_obs = O.shape[1]
    n_states = O.shape[0]

    # LOG-space targets (shared EPS convention with the RL reward).
    log_kp = KP.clamp_min(ORACLE_EPS).log()
    log_kd = KD.clamp_min(ORACLE_EPS).log()
    log_tgt = torch.cat([log_kp, log_kd], dim=1)   # (N, 50)

    model = OracleMLP(n_obs, n_dof=N_DOF, hidden=(256, 256), device=dev)
    t1 = time.time()
    val = fit_oracle(model, O, log_tgt, epochs, dev)
    t_fit = time.time() - t1

    # Baseline (predict-the-mean) val MSE for context on fit quality.
    with torch.no_grad():
        mean_tgt = log_tgt.mean(dim=0, keepdim=True)
        baseline_mse = ((log_tgt - mean_tgt) ** 2).mean().item()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta = {
        "state_dict": model.state_dict(),
        "n_obs": n_obs,
        "n_dof": N_DOF,
        "hidden": (256, 256),
        "eps": ORACLE_EPS,
        "names": names,
        "target": "concat(log kp[25], log kd[25]); inputs are RAW 359 env._get_obs()",
        "fit": {"val_mse_log": val, "baseline_mse_log": baseline_mse,
                "n_states": n_states, "n_envs": n_envs, "n_steps": n_steps,
                "epochs": epochs, "collect_s": t_collect, "fit_s": t_fit},
    }
    torch.save(meta, out_path)
    stats = meta["fit"]
    print(f"SAVED {out_path}  n_states={n_states} n_obs={n_obs} "
          f"val_mse_log={val:.4f} baseline_mse_log={baseline_mse:.4f} "
          f"collect={t_collect:.1f}s fit={t_fit:.1f}s", flush=True)
    print(f"  kp range [{KP.min().item():.2f},{KP.max().item():.1f}] mean={KP.mean().item():.2f} | "
          f"kd range [{KD.min().item():.3f},{KD.max().item():.2f}] mean={KD.mean().item():.3f}",
          flush=True)
    return model, meta, stats


def _smoke():
    """Build (or load) the oracle, then roll the PD-RL env and print rew_impedance stats,
    including a gain-mismatch-response demo. Confirms finite, negative, responsive, no NaN."""
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinterTorquePD_RL

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build the oracle if missing, else load and report cached fit.
    if os.path.exists(ORACLE_PATH):
        _, meta = load_oracle(ORACLE_PATH, dev)
        print(f"LOADED cached oracle {ORACLE_PATH}  fit={meta.get('fit')}", flush=True)
    else:
        print("No cached oracle; building offline...", flush=True)
        _, meta, _ = build_oracle(device=str(dev))

    # 2) Build the PD-RL env with lambda_impedance>0 so the term is active in the total reward.
    cfg = EnvConfigSprinterTorquePD_RL()
    cfg.reward_lambdas = dict(cfg.reward_lambdas)
    cfg.reward_lambdas["lambda_impedance"] = 0.01
    env = EnvFactory.create_env(num_envs=64, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)
    assert env.num_actions() == 75

    obs = env.reset()
    n_steps = 20
    vals = []
    any_nan = False
    for i in range(n_steps):
        actions = env.get_random_actions()
        obs, rew, term, trunc, info = env.step(actions)
        r = info["raw_rewards"]["rew_impedance"]
        any_nan = any_nan or bool(torch.isnan(r).any())
        vals.append(r.detach())
        if torch.isnan(rew).any():
            any_nan = True

    stack = torch.stack(vals)  # (n_steps, num_worlds)
    print(f"RESULT rew_impedance over {n_steps} steps: "
          f"mean={stack.mean().item():.3f} min={stack.min().item():.3f} "
          f"max={stack.max().item():.3f} finite={bool(torch.isfinite(stack).all())} "
          f"any_nan={any_nan}", flush=True)
    print(f"RESULT scaled (lambda=0.01) mean contribution={0.01*stack.mean().item():.5f}",
          flush=True)

    # 3) Gain-mismatch-response demo: recompute rew_impedance with the CURRENT (student) gains,
    #    then artificially inflate the policy's kp by 10x and recompute. Reward MUST drop.
    env._compute_raw_reward_dict()
    r_base = env.reward_dict["rew_impedance"].clone()
    kp_save = env._last_kp.clone()
    env._last_kp.mul_(10.0)          # inflate emitted stiffness -> larger log-space distance
    env._compute_raw_reward_dict()
    r_infl = env.reward_dict["rew_impedance"].clone()
    env._last_kp.copy_(kp_save)      # restore
    dropped = bool((r_infl < r_base - 1e-6).float().mean().item() > 0.9)
    print(f"RESULT gain_mismatch_demo base_mean={r_base.mean().item():.3f} "
          f"inflated10x_mean={r_infl.mean().item():.3f} "
          f"reward_dropped_frac={(r_infl < r_base).float().mean().item():.3f} "
          f"responds={dropped}", flush=True)
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the oracle even if cached")
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--n_steps", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    if args.rebuild and os.path.exists(ORACLE_PATH):
        os.remove(ORACLE_PATH)
    if args.rebuild:
        build_oracle(n_envs=args.n_envs, n_steps=args.n_steps, epochs=args.epochs)
    _smoke()


if __name__ == "__main__":
    main()
