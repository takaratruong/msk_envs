# Muscle → Torque DAgger (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distill the trained muscle sprint policy into a torque-actuated student policy via DAgger, and show the student reproduces the sprint gait in rollout.

**Architecture:** One OpenSim model carries BOTH the 136 muscles AND 25 coordinate-actuators (verified: Bolt's loader handles `nz = 2*nmuscle + nactuator`). The muscle **teacher** (existing checkpoint) is driven through `muscle_excitations` and its net joint torque is read from `ufrc_muscle`; the torque **student** (new MLP) is driven through `actuator_excitations`. Because both actuation systems live in the same env/state, DAgger needs no two-env state syncing: at each student-visited state we apply the teacher's muscle excitations, read `ufrc_muscle` as the torque label, then continue the student rollout. Labels are aggregated and the student is refit by MSE each round.

**Tech Stack:** Python 3.11, PyTorch (CUDA 13), Bolt (Warp), OpenSim 4.6, existing msk_envs TD3 infra (`DeterministicPolicy`, `load_policy`, `LocomotionEnv`/`EnvFactory`).

**Spec:** `docs/superpowers/specs/2026-08-28-muscle-to-torque-dagger-design.md`

## Global Constraints

- **Actuator torque map (verified in `bolt/_src/smooth_actuator.py`):** `torque = (activation − 0.5) × 2 × optimal_force`, excitation ∈ [0,1] → activation via first-order lag. For Phase A set `activation_time_constant = 0` so applied torque = commanded torque.
- **Excitation inversion (student target → excitation to write):** `excitation = clamp(τ_target / (2·optimal_force) + 0.5, 0, 1)`.
- **`optimal_force` per actuator** MUST be sized from observed teacher torque: `optimal_force[dof] = 1.2 × max|ufrc_muscle[dof]|` over a teacher rollout batch. Never guessed.
- **DoF split:** student controls the 25 non-root joint DoFs; the 6 pelvis root DoFs (`pelvis_tilt/list/rotation/tx/ty/tz`) are the unactuated floating base and are excluded from the student action and the torque label.
- **Teacher = existing checkpoint** `models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt` (TD3 `DeterministicPolicy`, sprinter model, 136 muscles).
- **GPU:** Bolt needs CUDA → run with `dangerouslyDisableSandbox: true`; pick an idle GPU via `nvidia-smi` and set `CUDA_VISIBLE_DEVICES`. GPUs shared with co-tenants.
- **Env activation:** `source /home/ubuntu/miniconda3/etc/profile.d/conda.sh; conda activate bolt`.
- **All new module code lives in `msk_envs/msk_envs/distill/`; edits to existing files are additive only.**

**User decisions (already made):**
- "Torque G1 is the target; muscles are a prior" — muscle model never runs on hardware; it is the teacher.
- "we should make a dagger" — DAgger loop, not plain BC (BC is the ablation baseline only).
- "2 is probably best" — rollout fidelity is the PRIMARY success criterion; torque-MSE / DAgger-vs-BC / naturalness are diagnostics.
- "put C on the back burner" — Phase C (full G1 tree) is out of scope; teacher stays the sprinter.
- "we have a working sprinting motion right now. so we can use this" — use the sprint checkpoint as teacher.

---

### Task 0: Build the sprinter torque-twin model (muscles + actuators in one .osim)

**Goal:** Produce `sprinter_torque.osim` — the sprinter model with 25 `ActivationCoordinateActuator`s added (muscles retained), `optimal_force` per actuator sized from teacher torques, and register a Bolt env config for it.

**Files:**
- Create: `/home/ubuntu/bolt/add_sprinter_torque_actuators.py`
- Create (output): `/home/ubuntu/bolt/data/models/sprinter/sprinter_torque.osim`
- Create (output): `/home/ubuntu/msk_envs/msk_envs/msk_models/sprinter/sprinter_torque.osim` (copy for env)
- Create (output): `/home/ubuntu/msk_envs/msk_envs/msk_models/sprinter/sprinter_torque_optforce.json` (name→optimal_force map, for the DAgger loop)
- Modify: `/home/ubuntu/msk_envs/msk_envs/envs/env_config.py` (add `EnvConfigSprinterTorque` + union entry)

**Acceptance Criteria:**
- [ ] Script computes per-DoF max |`ufrc_muscle`| from a teacher rollout (≥512 env-steps) and writes `optimal_force = 1.2 × that` for each of the 25 joint DoFs.
- [ ] Output `.osim` loads in Bolt with `num_muscles == 136` AND `num_actuators == 25`.
- [ ] `env-config:sprintertorque` resolves and an env constructs without error.

**Verify:** `python add_sprinter_torque_actuators.py` → prints `WROTE ... | muscles=136 actuators=25` and per-DoF optimal_force table.

**Steps:**

- [ ] **Step 1: Collect teacher torque ranges (data-gathering pass).**

Write `add_sprinter_torque_actuators.py`. First section rolls out the teacher to size actuators. Reuse the env + policy exactly as the codebase does:

```python
import os, sys, json, torch, numpy as np, opensim as osim
sys.path.insert(0, "/home/ubuntu/msk_envs")
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinter
from msk_envs.train.nets.deterministic_policy import load_policy

CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
device = torch.device("cuda")

def collect_teacher_torque_max(n_envs=256, n_steps=256):
    env = EnvFactory.create_env(num_envs=n_envs, env_config=EnvConfigSprinter(),
                                requires_visuals=False, cuda_graph=True, device=device)
    policy = load_policy(CKPT); policy.to(device)
    obs = env.reset()
    max_abs = torch.zeros(env.num_dofs, device=device)
    with torch.no_grad():
        for _ in range(n_steps):
            actions = env.get_blank_actions()
            actions[:, :env.num_muscles] = policy(obs)          # teacher drives muscles
            _, obs = env.step(actions)                          # NOTE: if step signature differs, see deploy.py
            max_abs = torch.maximum(max_abs, env.ufrc_muscle.abs().max(dim=0).values)
    return max_abs.cpu().numpy(), env
```

- [ ] **Step 2: Run to verify the teacher-torque pass works and inspect ranges.**

Run: `cd /home/ubuntu/bolt && CUDA_VISIBLE_DEVICES=<idle> python -c "import add_sprinter_torque_actuators as a; m,e=a.collect_teacher_torque_max(); print('per-dof max|tau|', m)"` (sandbox off)
Expected: prints a length-31 array; the 25 joint DoFs have non-trivial magnitudes (N·m up to hundreds), the 6 root DoFs also present (excluded next step). If `env.step` errors, mirror the exact call in `deploy.py` (`finished, obs = sim.step(actions)` uses `LoggedSim`; the raw env returns `(terminated, obs)` — adapt to whichever `EnvFactory` env exposes).

- [ ] **Step 3: Identify the 25 joint DoFs (exclude 6 root) via the model coordinates.**

```python
ROOT = {"pelvis_tilt","pelvis_list","pelvis_rotation","pelvis_tx","pelvis_ty","pelvis_tz"}

def joint_coords(model):
    cs = model.getCoordinateSet()
    names = [cs.get(i).getName() for i in range(cs.getSize())]
    return [n for n in names if n not in ROOT]   # 25 names, model order
```

- [ ] **Step 4: Add ActivationCoordinateActuators sized from teacher torque; dump optforce json.**

```python
def build(out_osim="/home/ubuntu/bolt/data/models/sprinter/sprinter_torque.osim"):
    max_abs, env = collect_teacher_torque_max()
    src = "/home/ubuntu/bolt/data/models/sprinter/sprinter_model.osim"
    model = osim.Model(src); model.initSystem()
    joints = joint_coords(model)
    dof_lu = env.dof_id_lookup                # coord name -> dof index used by ufrc_muscle
    of_map = {}
    for name in joints:
        act = osim.ActivationCoordinateActuator()
        act.setName(f"act_{name}")
        act.setCoordinate(model.getCoordinateSet().get(name))
        of = float(1.2 * max_abs[dof_lu[name]]); of = max(of, 1.0)   # floor so zero-torque DoFs still actuate
        act.setOptimalForce(of)
        act.set_activation_time_constant(0.0)     # Phase A: instantaneous
        act.set_default_activation(0.5)           # 0.5 -> zero torque per (act-0.5)*2 map
        model.addForce(act); of_map[name] = of
        print(f"  {name:22s} optimal_force={of:.1f}")
    model.finalizeConnections(); model.printToXML(out_osim)
    json.dump(of_map, open(out_osim.replace(".osim","_optforce.json"),"w"), indent=2)
    print("WROTE", out_osim)
    m2 = osim.Model(out_osim); m2.initSystem()
    n_mus = m2.getMuscles().getSize(); n_act = m2.getActuators().getSize() - n_mus
    print(f"reload muscles={n_mus} actuators={n_act}")

if __name__ == "__main__":
    build()
```

- [ ] **Step 5: Run the build, copy into msk_models, verify counts in Bolt.**

Run: `cd /home/ubuntu/bolt && CUDA_VISIBLE_DEVICES=<idle> python add_sprinter_torque_actuators.py` (sandbox off)
Then copy: `cp data/models/sprinter/sprinter_torque.osim data/models/sprinter/sprinter_torque_optforce.json /home/ubuntu/msk_envs/msk_envs/msk_models/sprinter/`
Bolt count check:
```python
import bolt; from bolt import bindings
lr = bindings.load_model(model_path="data/models/sprinter/sprinter_torque.osim", n_worlds=4,
                         integrator=bolt.IntegratorType.EULER_ADAPTIVE, requires_visuals=False,
                         muscle_fn_path="data/models/sprinter/sprinter_model_fn.xml")
print("muscles", bolt.get_num_muscles(lr.model), "actuators", bolt.get_num_actuators(lr.model))
```
Expected: `muscles 136 actuators 25`.

- [ ] **Step 6: Add env config (additive), register subcommand.**

In `msk_envs/msk_envs/envs/env_config.py`, after `EnvConfigSprinter`:
```python
@dataclass
class EnvConfigSprinterTorque(EnvConfigSprinter):
    """Sprinter skeleton carrying BOTH muscles and 25 coordinate-actuators.
    Teacher drives muscles; torque student drives actuators. See distill/ + spec."""
    model_path: str = "../msk_models/sprinter/sprinter_torque.osim"
```
Add to `EnvConfigUnion`: `Annotated[EnvConfigSprinterTorque, tyro.conf.subcommand(name="sprintertorque")],`

- [ ] **Step 7: Commit.**

```bash
cd /home/ubuntu/msk_envs
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add -A
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(distill): sprinter torque-twin model + env config (Task 0)"
```

```json:metadata
{"files": ["/home/ubuntu/bolt/add_sprinter_torque_actuators.py", "/home/ubuntu/msk_envs/msk_envs/envs/env_config.py"], "verifyCommand": "cd /home/ubuntu/bolt && python add_sprinter_torque_actuators.py", "acceptanceCriteria": ["output osim loads with 136 muscles and 25 actuators", "optimal_force sized as 1.2x teacher max|ufrc_muscle| per dof", "env-config:sprintertorque resolves"], "modelTier": "standard"}
```

---

### Task 1: Teacher wrapper — muscle policy + torque label readout

**Goal:** `distill/teacher.py` loads the muscle checkpoint and, given an env at some state, returns the teacher's muscle excitations AND the resulting 25-DoF torque label read from `ufrc_muscle`.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/__init__.py` (empty)
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/teacher.py`
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/dof_utils.py` (shared DoF index helpers)

**Acceptance Criteria:**
- [ ] `Teacher.action(obs)` returns muscle excitations matching `env.num_muscles`.
- [ ] `Teacher.torque_label(env)` returns a `(n_envs, 25)` tensor sliced to the joint DoFs from `env.ufrc_muscle` using `dof_utils`.
- [ ] Smoke script: teacher-driven rollout produces finite, non-zero torque labels with the expected shape.

**Verify:** `CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.teacher` → prints `label shape (N, 25) mean|tau|=<finite>`.

**Steps:**

- [ ] **Step 1: DoF index helper.**

`distill/dof_utils.py`:
```python
ROOT_DOFS = ("pelvis_tilt","pelvis_list","pelvis_rotation","pelvis_tx","pelvis_ty","pelvis_tz")

def joint_dof_indices(env):
    """Indices into the nv-dim ufrc/dof arrays for the 25 non-root joint DoFs, in a stable order."""
    lu = env.dof_id_lookup                      # name -> index
    names = [n for n in lu if n not in ROOT_DOFS]
    names.sort(key=lambda n: lu[n])             # stable: sim dof order
    idx = [lu[n] for n in names]
    return names, idx
```

- [ ] **Step 2: Teacher class.**

`distill/teacher.py`:
```python
import torch
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
        return env.ufrc_muscle.index_select(1, self.idx_t).clone()   # (n_envs, 25)
```

- [ ] **Step 3: Smoke `__main__`.**

```python
if __name__ == "__main__":
    import torch
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinterTorque
    dev = torch.device("cuda")
    env = EnvFactory.create_env(num_envs=64, env_config=EnvConfigSprinterTorque(),
                                requires_visuals=False, cuda_graph=True, device=dev)
    CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
    t = Teacher(CKPT, env, dev)
    obs = env.reset()
    for _ in range(20):
        a = env.get_blank_actions()
        a[:, :env.num_muscles] = t.action(obs)
        _, obs = env.step(a)
    lab = t.torque_label(env)
    print(f"label shape {tuple(lab.shape)} mean|tau|={lab.abs().mean().item():.3f}")
    assert lab.shape[1] == 25 and torch.isfinite(lab).all()
```

- [ ] **Step 4: Run smoke test.**

Run: `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.teacher` (sandbox off)
Expected: `label shape (64, 25) mean|tau|=<finite non-zero>`, no assertion error.

- [ ] **Step 5: Commit.**

```bash
cd /home/ubuntu/msk_envs
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add msk_envs/distill/
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(distill): teacher wrapper + dof utils (Task 1)"
```

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/teacher.py", "/home/ubuntu/msk_envs/msk_envs/distill/dof_utils.py", "/home/ubuntu/msk_envs/msk_envs/distill/__init__.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.teacher", "acceptanceCriteria": ["Teacher.action returns num_muscles-dim excitations", "torque_label returns (n_envs,25) finite tensor sliced from ufrc_muscle"], "modelTier": "standard"}
```

---

### Task 2: Student policy — obs → 25 torques, applied via actuators

**Goal:** `distill/student.py` — an MLP mapping obs → 25 torque targets, with helpers to convert torques→excitations (via the verified inversion) and write them into the env's `actuator_excitations`, plus save/load.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/student.py`

**Acceptance Criteria:**
- [ ] `Student(obs)` returns `(n_envs, 25)` torque targets.
- [ ] `torque_to_excitation(tau, optimal_force)` implements `clamp(tau/(2·of)+0.5, 0, 1)` and round-trips: `excitation→torque` recovers input tau within clamp range.
- [ ] `apply(env, tau)` writes into the actuator slice of the env action and steps without error.

**Verify:** `python -m msk_envs.distill.student` → prints `roundtrip max err <1e-5` and `applied step ok`.

**Steps:**

- [ ] **Step 1: Student MLP + conversion.**

```python
import torch, torch.nn as nn

class Student(nn.Module):
    def __init__(self, n_obs, n_dof=25, hidden=(256,256), device="cuda"):
        super().__init__()
        dims = [n_obs, *hidden]; layers=[]
        for a,b in zip(dims, dims[1:]):
            layers += [nn.Linear(a,b,device=device), nn.ReLU()]
        layers += [nn.Linear(dims[-1], n_dof, device=device)]
        self.net = nn.Sequential(*layers); self.n_dof = n_dof

    def forward(self, obs):
        return self.net(obs)                    # (n_envs, 25) torque targets (N·m)

def torque_to_excitation(tau, optimal_force):
    # inverse of torque = (act-0.5)*2*of ; excitation==activation when tau=0 lag
    return torch.clamp(tau / (2.0 * optimal_force) + 0.5, 0.0, 1.0)
```

- [ ] **Step 2: Apply-to-env helper (uses actuator slice).**

```python
def apply(env, tau, optimal_force, base_action=None):
    """Write student torques into env actuator excitations and step. Muscle slice left at blank (min activation)."""
    a = base_action if base_action is not None else env.get_blank_actions()
    exc = torque_to_excitation(tau, optimal_force)             # [0,1]
    # env maps raw_action [-1,1] -> excitation (a+1)/2 for actuators; pre-invert to raw:
    raw = exc * 2.0 - 1.0
    a[:, env.num_muscles:] = raw
    return env.step(a)
```

Note: `_set_actuator_excitations` maps raw `[-1,1]`→`[0,1]` (confirmed in env_base). We pre-invert so the final written excitation equals `torque_to_excitation`.

- [ ] **Step 3: Smoke `__main__` — roundtrip + apply.**

```python
if __name__ == "__main__":
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinterTorque
    dev = torch.device("cuda")
    env = EnvFactory.create_env(num_envs=16, env_config=EnvConfigSprinterTorque(),
                                requires_visuals=False, cuda_graph=True, device=dev)
    of = torch.full((25,), 100.0, device=dev)
    tau = torch.randn(16,25, device=dev) * 20.0
    exc = torque_to_excitation(tau, of); tau_rt = (exc-0.5)*2.0*of
    valid = tau.abs() < of                       # only unclamped entries round-trip
    err = (tau_rt[valid]-tau[valid]).abs().max().item()
    print(f"roundtrip max err {err:.2e}")
    obs = env.reset()
    s = Student(obs.shape[1], device=dev)
    apply(env, s(obs), of); print("applied step ok")
```

- [ ] **Step 4: Run smoke test.**

Run: `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.student` (sandbox off)
Expected: `roundtrip max err <1e-5` then `applied step ok`.

- [ ] **Step 5: Commit.**

```bash
cd /home/ubuntu/msk_envs
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add msk_envs/distill/student.py
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(distill): torque student + torque<->excitation conversion (Task 2)"
```

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/student.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.student", "acceptanceCriteria": ["Student(obs) returns (n_envs,25)", "torque_to_excitation round-trips within clamp", "apply() steps env without error"], "modelTier": "standard"}
```

---

### Task 3: DAgger loop — the orchestrator

**Goal:** `distill/dagger.py` runs the DAgger loop: β-mixed rollouts (teacher muscles vs student torques) in the SAME env, collect `(obs, τ_teacher)` at student-visited states, aggregate, refit student by MSE each round; save student + a metrics log.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/dagger.py`
- Create (output): `/home/ubuntu/msk_envs/models/dagger_sprint/student.pt`, `metrics.json`

**Acceptance Criteria:**
- [ ] Runs ≥8 DAgger rounds without error; buffer grows each round; per-round val MSE is logged.
- [ ] Val torque MSE decreases from round 0 to final (final < 0.6× round-0).
- [ ] Saves `student.pt` (state_dict + obs dim + optimal_force vector + names) and `metrics.json`.

**Verify:** `CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.dagger --rounds 8` → prints per-round val MSE, final line `SAVED student.pt`, and final val < 0.6× round-0.

**Steps:**

- [ ] **Step 1: Config + setup (env, teacher, student, optimal_force from json).**

```python
import os, json, argparse, torch
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorque
from msk_envs.distill.teacher import Teacher
from msk_envs.distill.student import Student, torque_to_excitation
from msk_envs.distill.dof_utils import joint_dof_indices

CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
OPTFORCE = "/home/ubuntu/msk_envs/msk_envs/msk_models/sprinter/sprinter_torque_optforce.json"

def read_optimal_force(names, device):
    of_map = json.load(open(OPTFORCE))          # name -> optimal_force (written in Task 0)
    return torch.tensor([of_map[n] for n in names], device=device)
```

- [ ] **Step 2: One DAgger round.**

```python
def rollout_and_label(env, teacher, student, of, beta, horizon, device):
    """Roll out horizon steps. Every step: apply teacher muscles to READ the torque label at the
    current state, record (obs, label). Then advance the controlled trajectory: with prob beta keep
    the teacher's resulting state, else take a student torque step (on-student-distribution data)."""
    obs = env.reset(); O, T = [], []
    for _ in range(horizon):
        a = env.get_blank_actions()
        a[:, :env.num_muscles] = teacher.action(obs)      # teacher muscles -> ufrc_muscle
        _, obs_teacher = env.step(a)
        O.append(obs.detach().clone()); T.append(teacher.torque_label(env))
        if torch.rand(()) < beta:
            obs = obs_teacher
        else:
            with torch.no_grad(): tau = student(obs)
            a2 = env.get_blank_actions()
            a2[:, env.num_muscles:] = torque_to_excitation(tau, of) * 2.0 - 1.0
            _, obs = env.step(a2)
    return torch.cat(O), torch.cat(T)
```
Rationale for β: round 0 uses β=1 (pure teacher) to seed; anneal β→0 over rounds so later data is on-student-distribution (the DAgger correction).

- [ ] **Step 3: Aggregate + fit.**

```python
def fit(student, buf_O, buf_T, epochs, device):
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    n = buf_O.shape[0]; idx = torch.randperm(n, device=device)
    n_val = n//5; vi, ti = idx[:n_val], idx[n_val:]
    for _ in range(epochs):
        for mb in ti.split(4096):
            opt.zero_grad()
            loss = ((student(buf_O[mb]) - buf_T[mb])**2).mean()
            loss.backward(); opt.step()
    with torch.no_grad():
        val = ((student(buf_O[vi]) - buf_T[vi])**2).mean().item()
    return val
```

- [ ] **Step 4: Main loop, β schedule, save.**

```python
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n_envs", type=int, default=256); ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5); args = ap.parse_args()
    dev = torch.device("cuda")
    env = EnvFactory.create_env(num_envs=args.n_envs, env_config=EnvConfigSprinterTorque(),
                                requires_visuals=False, cuda_graph=True, device=dev)
    names, _ = joint_dof_indices(env)
    teacher = Teacher(CKPT, env, dev)
    n_obs = env.reset().shape[1]
    student = Student(n_obs, n_dof=25, device=dev)
    of = read_optimal_force(names, dev)
    bufO = bufT = None; metrics=[]
    for r in range(args.rounds):
        beta = 1.0 if r==0 else max(0.0, 1.0 - r/(args.rounds-1))
        O,T = rollout_and_label(env, teacher, student, of, beta, args.horizon, dev)
        bufO = O if bufO is None else torch.cat([bufO,O]); bufT = T if bufT is None else torch.cat([bufT,T])
        val = fit(student, bufO, bufT, args.epochs, dev)
        print(f"round {r} beta={beta:.2f} buf={bufO.shape[0]} val_mse={val:.4f}")
        metrics.append({"round":r,"beta":beta,"buf":int(bufO.shape[0]),"val_mse":val})
    out = "/home/ubuntu/msk_envs/models/dagger_sprint"; os.makedirs(out, exist_ok=True)
    torch.save({"state_dict":student.state_dict(),"n_obs":n_obs,"optimal_force":of.cpu(),"names":names}, f"{out}/student.pt")
    json.dump(metrics, open(f"{out}/metrics.json","w"), indent=2)
    print(f"SAVED student.pt  val0={metrics[0]['val_mse']:.4f} valF={metrics[-1]['val_mse']:.4f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run 8 rounds, confirm MSE drop.**

Run: `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.dagger --rounds 8` (sandbox off)
Expected: 8 `round r ... val_mse=...` lines with final `val_mse` < 0.6 × round-0, then `SAVED student.pt`. If MSE plateaus high, raise `--epochs`/`--horizon` (real tuning knobs) and record the values used in metrics.json.

- [ ] **Step 6: Commit.**

```bash
cd /home/ubuntu/msk_envs
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add msk_envs/distill/dagger.py
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(distill): DAgger loop, teacher-labeled torque distillation (Task 3)"
```

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/dagger.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.dagger --rounds 8", "acceptanceCriteria": ["8 rounds run, buffer grows each round", "final val_mse < 0.6x round-0 val_mse", "saves student.pt + metrics.json"], "modelTier": "frontier"}
```

---

### Task 4: Rollout-fidelity evaluation (PRIMARY success criterion) + BC ablation

**Goal:** `distill/evaluate.py` runs the trained student (torque control) and measures gait reproduction vs the teacher — upright duration, forward-speed tracking, CoM drift — and runs a plain-BC baseline (teacher-states-only) for the DAgger-vs-BC diagnostic.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation ("2 is probably best" = rollout fidelity is the success bar). It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured — specifically the teacher/dagger/bc metrics table and the PHASE_A_PASS/FAIL line from a real GPU run.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/evaluate.py`
- Create (output): `/home/ubuntu/msk_envs/models/dagger_sprint/eval.json`

**Acceptance Criteria:**
- [ ] Reports, over ≥256 envs for a full episode: mean upright fraction, mean forward-speed error vs command, mean CoM lateral drift — for BOTH student and teacher.
- [ ] PRIMARY: student `upright_fraction ≥ 0.7` AND `speed_err ≤ 1.5×` teacher's — printed as `PHASE_A_PASS` / `PHASE_A_FAIL`.
- [ ] Diagnostic: prints DAgger student vs BC student on the same metrics.

**Verify:** `CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.evaluate` → prints teacher/dagger/bc metrics table and a `PHASE_A_PASS` or `PHASE_A_FAIL` line.

**Steps:**

- [ ] **Step 1: Rollout+metrics helper (teacher and student share it).**

```python
import torch, json
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorque
from msk_envs.distill.teacher import Teacher
from msk_envs.distill.student import Student, torque_to_excitation

def run_metrics(env, controller, is_teacher, of=None, steps=None):
    """controller: obs->(muscle excitations if teacher, else torque targets). Returns gait metrics."""
    obs = env.reset(); steps = steps or int(env.max_episode_duration/env.delta_t)
    upright = torch.zeros(env.num_worlds, device=obs.device)
    speed_err = torch.zeros(env.num_worlds, device=obs.device)
    drift = torch.zeros(env.num_worlds, device=obs.device)
    alive = torch.ones(env.num_worlds, device=obs.device)
    for _ in range(steps):
        a = env.get_blank_actions()
        if is_teacher:
            a[:, :env.num_muscles] = controller(obs)
        else:
            with torch.no_grad(): tau = controller(obs)
            a[:, env.num_muscles:] = torque_to_excitation(tau, of)*2.0-1.0
        term, obs = env.step(a)
        alive = alive * (1.0 - term.float())
        upright += alive
        speed_err += (env._root_vel_xy() - env.command_vel).norm(dim=1) * alive
        drift += env.root_pos[:, 2].abs() * alive          # lateral (SIDE) drift proxy
    denom = upright.clamp(min=1)
    return {"upright_fraction": (upright/steps).mean().item(),
            "speed_err": (speed_err/denom).mean().item(),
            "drift": (drift/denom).mean().item()}
```
Note: `_root_vel_xy`, `command_vel`, `root_pos`, `num_worlds` confirmed in `env_locomotion.py`/`env_base.py`. The SprinterTorque env wraps the sprint task so these attributes exist.

- [ ] **Step 2: Load DAgger student.**

```python
def load_student(path, dev):
    ck = torch.load(path, map_location=dev)
    s = Student(ck["n_obs"], n_dof=25, device=dev); s.load_state_dict(ck["state_dict"])
    return s, ck["optimal_force"].to(dev)
```

- [ ] **Step 3: BC ablation — student trained on teacher-only states, comparable sample budget.**

```python
def train_bc(env, teacher, n_obs, dev, n_batches=25, horizon=128):
    """Collect teacher-state data only (beta=1 always), fit once — the DAgger-vs-BC control."""
    s = Student(n_obs, 25, device=dev); opt = torch.optim.Adam(s.parameters(), 1e-3)
    O=[];T=[]; obs=env.reset()
    for _ in range(horizon):
        a=env.get_blank_actions(); a[:,:env.num_muscles]=teacher.action(obs); _,obs=env.step(a)
        O.append(obs.clone()); T.append(teacher.torque_label(env))
    O=torch.cat(O);T=torch.cat(T)
    for _ in range(n_batches):
        for mb in torch.randperm(O.shape[0], device=dev).split(4096):
            opt.zero_grad(); (((s(O[mb])-T[mb])**2).mean()).backward(); opt.step()
    return s
```

- [ ] **Step 4: Main — assemble table, PHASE_A verdict.**

```python
def main():
    dev=torch.device("cuda")
    env=EnvFactory.create_env(num_envs=256, env_config=EnvConfigSprinterTorque(),
                              requires_visuals=False, cuda_graph=True, device=dev)
    CKPT="/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
    teacher=Teacher(CKPT, env, dev); n_obs=env.reset().shape[1]
    m_teacher=run_metrics(env, teacher.action, True)
    student, of=load_student("/home/ubuntu/msk_envs/models/dagger_sprint/student.pt", dev)
    m_student=run_metrics(env, student, False, of=of)
    bc=train_bc(env, teacher, n_obs, dev)
    m_bc=run_metrics(env, bc, False, of=of)
    for name,m in [("teacher",m_teacher),("dagger",m_student),("bc",m_bc)]:
        print(f"{name:8s} upright={m['upright_fraction']:.3f} speed_err={m['speed_err']:.3f} drift={m['drift']:.3f}")
    ok = m_student["upright_fraction"]>=0.7 and m_student["speed_err"]<=1.5*m_teacher["speed_err"]
    print("PHASE_A_PASS" if ok else "PHASE_A_FAIL")
    json.dump({"teacher":m_teacher,"dagger":m_student,"bc":m_bc,"pass":ok},
              open("/home/ubuntu/msk_envs/models/dagger_sprint/eval.json","w"), indent=2)

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run evaluation.**

Run: `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.evaluate` (sandbox off)
Expected: 3-row table (teacher/dagger/bc) + `PHASE_A_PASS` or `PHASE_A_FAIL`. A FAIL is a legitimate scientific outcome (motivates Phase B) — record it, do not fudge thresholds.

- [ ] **Step 6: Commit.**

```bash
cd /home/ubuntu/msk_envs
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add msk_envs/distill/evaluate.py
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(distill): rollout-fidelity eval + BC ablation (Task 4)"
```

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/evaluate.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.evaluate", "acceptanceCriteria": ["reports upright/speed_err/drift for teacher+dagger+bc over 256 envs", "prints PHASE_A_PASS or PHASE_A_FAIL from upright>=0.7 and speed_err<=1.5x teacher", "DAgger-vs-BC comparison printed"], "modelTier": "frontier", "userGate": true, "tags": ["user-gate"], "requireEvidenceTokens": [["teacher"], ["dagger"], ["bc"]]}
```

---

### Task 5: Render the distilled student on the dashboard (naturalness diagnostic)

**Goal:** Render the torque student's rollout to video and add it to the Firefox dashboard next to the muscle teacher, for side-by-side naturalness comparison.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/record.py`
- Modify: `/home/ubuntu/bolt_baselines/dashboard.html` (add a "Distilled Torque Student" card)
- Create (output): `/home/ubuntu/bolt_baselines/videos/dagger_student.mp4`

**Acceptance Criteria:**
- [ ] A single-env student rollout (torque control) is recorded and rendered to `dagger_student.mp4`.
- [ ] Dashboard shows a "Distilled Torque Student" card; opens in Firefox on display `:1`.

**Verify:** `ls -la /home/ubuntu/bolt_baselines/videos/dagger_student.mp4` shows a non-empty file, and a screenshot of `:1` shows the new card.

**Steps:**

- [ ] **Step 1: Record student rollout via LoggedSim (mirror deploy.py).**

`distill/record.py` — copy `deploy.py`'s LoggedSim setup verbatim, but use `EnvConfigSprinterTorque`, load the student, and fill actions with the student torque path:
```python
import torch
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorque
from msk_envs.train.deploy import LoggedSim   # if not importable, replicate the LoggedSim block from deploy.py
from msk_envs.distill.student import Student, torque_to_excitation

def main():
    dev=torch.device("cuda")
    env=EnvFactory.create_env(num_envs=1, env_config=EnvConfigSprinterTorque(),
                              requires_visuals=True, cuda_graph=True, device=dev)
    ck=torch.load("/home/ubuntu/msk_envs/models/dagger_sprint/student.pt", map_location=dev)
    s=Student(ck["n_obs"],25,device=dev); s.load_state_dict(ck["state_dict"]); of=ck["optimal_force"].to(dev)
    sim=LoggedSim(env, dev, delta_t_log=1/30.0); obs=sim.reset()
    steps=int(env.max_episode_duration/env.delta_t)
    for _ in range(steps):
        a=env.get_blank_actions()
        with torch.no_grad(): a[:, env.num_muscles:]=torque_to_excitation(s(obs), of)*2.0-1.0
        fin,obs=sim.step(a)
        if fin.all(): break
    sim.save_animation("dashboard/trajectories/dagger_student","1", use_gzip=True)

if __name__=="__main__": main()
```
Note: if `LoggedSim` is not importable from `deploy`, copy the `LoggedSim` construction lines out of `deploy.py` into `record.py` (they are self-contained there).

- [ ] **Step 2: Render frames → mp4 with the existing renderer.**

```bash
cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.record   # writes trajectory json.gz
cd /home/ubuntu/bolt
xvfb-run -a -s "-screen 0 1200x1000x24" python render_trajectory.py \
  /home/ubuntu/msk_envs/dashboard/trajectories/dagger_student_1.json.gz /tmp/claude/student_frames
ffmpeg -y -framerate 30 -i /tmp/claude/student_frames/frame_%04d.png -pix_fmt yuv420p \
  /home/ubuntu/bolt_baselines/videos/dagger_student.mp4
```
(Confirm the exact trajectory filename `save_animation` produces; adjust the path passed to `render_trajectory.py`.)

- [ ] **Step 3: Add dashboard card (additive edit).**

In `/home/ubuntu/bolt_baselines/dashboard.html`, add a card after the G1 card:
```html
<div class="card">
  <h2>Distilled Torque Student <span style="color:#8b97a5;font-weight:normal;font-size:12px">· DAgger from sprint teacher</span></h2>
  <video src="videos/dagger_student.mp4" controls autoplay loop muted></video>
  <div class="meta">torque-actuated student (25 coordinate-actuators) distilled from the 136-muscle sprint policy via DAgger — naturalness comparison vs the muscle teacher</div>
</div>
```

- [ ] **Step 4: Open in Firefox on :1 (snap binary directly; gdm xauth).**

```bash
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority HOME=/home/ubuntu
rm -f ~/.mozilla/firefox/*/lock ~/.mozilla/firefox/*/.parentlock 2>/dev/null
LD_LIBRARY_PATH=/snap/firefox/current/usr/lib/firefox:/snap/firefox/current/usr/lib/firefox/gtk3 \
  setsid /snap/firefox/current/usr/lib/firefox/firefox --new-window "file:///home/ubuntu/bolt_baselines/dashboard.html" &
sleep 8
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority import -window root /tmp/claude/dash.png
```
(Both Firefox launch and screenshot need sandbox off. Read `/tmp/claude/dash.png` to confirm the card renders.)

- [ ] **Step 5: Commit.**

```bash
cd /home/ubuntu/msk_envs
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add -A
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(distill): render distilled student + dashboard card (Task 5)"
```

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/record.py", "/home/ubuntu/bolt_baselines/dashboard.html"], "verifyCommand": "ls -la /home/ubuntu/bolt_baselines/videos/dagger_student.mp4", "acceptanceCriteria": ["student rollout rendered to mp4", "dashboard shows student card and opens in Firefox on :1"], "modelTier": "standard"}
```

---

## Notes on ordering & dependencies

- Task 0 → 1 → 2 → 3 → 4 → 5 is strictly sequential: each depends on the prior (model → teacher → student → loop → eval → render).
- Tasks 0–3 are the build; Task 4 is the PRIMARY verdict (user gate); Task 5 is the qualitative diagnostic.
- A `PHASE_A_FAIL` at Task 4 does not block Task 5 — the render is still informative and motivates Phase B.
