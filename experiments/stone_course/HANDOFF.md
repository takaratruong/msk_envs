# Claude handoff

Use the clean worktree at `/home/ubuntu/msk_envs-stone-course` on branch
`experiment/stone-course`. Its fork remote is
`https://github.com/takaratruong/msk_envs`.

The original exploratory checkout at `/home/ubuntu/msk_envs` is intentionally
left untouched because it also contains unrelated distillation, walking, and G1
changes. The active pre-refactor training run remains there:

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

The endless-course retrain is active in tmux session
`stonecourse_recycled_interior` on physical GPU 5:

- experiment: `stonecourse_recycled_interior_2026-09-01_00-13`;
- worktree: `/home/ubuntu/msk_envs-stone-course`;
- log: `models/stonecourse_recycled_interior_launch.log`; and
- warm start: finite-course checkpoint 26000 from the run above.

Attach with `tmux attach -t stonecourse_recycled_interior`. The renderer scans
both worktrees and will switch the dashboard to this experiment when its first
evaluation trajectory appears at iteration 27000.

Commit `1fabeb5` is the focused checkpoint of the behavior that launched the
old finite run. Commit `d8f15b8` preserves the 344-value policy observation but
replaces the finite 24-slab course with five recycled per-world slabs, rejects
edge-pivot touchdowns, removes final-slab termination, and adds a success-driven
spacing curriculum with checkpoint/evaluation synchronization.

Before changing behavior, read [README.md](README.md) and run the focused tests.
