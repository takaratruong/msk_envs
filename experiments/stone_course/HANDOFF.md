# Claude handoff

Use the clean worktree at `/home/ubuntu/msk_envs-stone-course` on branch
`experiment/stone-course`. Its fork remote is
`https://github.com/takaratruong/msk_envs`.

The original exploratory checkout at `/home/ubuntu/msk_envs` is intentionally
left untouched because it also contains unrelated distillation, walking, and G1
changes. Its older finite-course result remains available at:

- experiment: `stonecourse_walkingheel_2026-08-31_19-41`;
- checkpoints: `/home/ubuntu/msk_envs/models/stonecourse_walkingheel_2026-08-31_19-41/`;
- trajectories: `/home/ubuntu/msk_envs/dashboard/trajectories/stonecourse_walkingheel_2026-08-31_19-41/`; and
- dashboard render: `/home/ubuntu/bolt_baselines/dashboard.html`.

The active run is rendered automatically by
`/home/ubuntu/bolt_baselines/auto_render_walksteps.sh`. Cron invokes it every
minute; it skips unchanged trajectories, publishes the newest rollout at
`videos/stonecourse_walkingheel_latest.mp4`, and updates
`stonecourse_training_status.json`. The top dashboard card polls that status
when served over HTTP and reloads the stable video path when opened via
`file://`.

The flat endless-course run was stopped after producing checkpoint 35000:

- experiment: `stonecourse_recycled_interior_2026-09-01_00-13`;
- worktree: `/home/ubuntu/msk_envs-stone-course`;
- log: `models/stonecourse_recycled_interior_launch.log`; and
- final retained checkpoint:
  `models/stonecourse_recycled_interior_2026-09-01_00-13/stonecourse_recycled_interior_2026-09-01_00-13_35000.pt`.

The active forward-3D curriculum runs in tmux session
`stonecourse_allsteps_forward` on physical GPU 7:

- experiment: `stonecourse_allsteps_forward_2026-09-01_02-32`;
- worktree: `/home/ubuntu/msk_envs-stone-course`;
- log: `models/stonecourse_allsteps_forward_launch.log`;
- warm start: `models/stonecourse_allsteps_forward_warmstart_35000.pt`; and
- source checkpoint: the flat endless-course checkpoint 35000 above.

Attach with `tmux attach -t stonecourse_allsteps_forward`. It starts at radial
distance 0.65–0.80 m with zero elevation/yaw/tilt, then competence promotions
expand distance to 1.50 m, elevation to ±50°, forward-biased yaw to ±20°, and
slab roll/pitch to ±20°. Reward is capped at the paper's walking speed of
1.35 m/s. The first corrected live window at iteration 35100 averaged 1.24 m/s
and 3.56 m of forward progress; it has not promoted yet.

The renderer now prefers this experiment once its first evaluation is written,
then falls back to the flat endless course and the original finite course.

Commit `1fabeb5` is the focused checkpoint of the behavior that launched the
old finite run. Commit `d8f15b8` introduced the five-slab recycled flat course.
Commit `7c7f943` adds the ALLSTEPS-inspired forward 3D curriculum, tilted-local
landing validation, 20 terrain observations, tests, and runbook. Commit
`ac69ccf` fixes warm-start normalization so all legacy gait inputs retain their
learned statistics. Both current commits are pushed to the fork.

Before changing behavior, read [README.md](README.md) and run the focused tests.
