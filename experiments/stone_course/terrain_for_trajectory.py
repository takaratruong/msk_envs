"""Fit the natural-terrain surface through a rollout's actual slab tops.

The foothold rollout (rollout_on_footholds.py) walks a fixed course sampled
from the curriculum distribution, plus deterministic continuation slabs past
its end. This script reads the slab layout back out of the saved trajectory -
the layout the walker physically stepped on, launch-slab snap included - and
fits one natural heightfield through every slab top with the same
fit_terrain() machinery as the terrain visualization.

That single fit is saved twice, guaranteed identical:
- an .npz grid for render_trajectory_natural_terrain.py, which draws it as a
  mesh under the walker in place of the slab boxes; and
- the hero image, with the 14 curriculum footholds in red/orange and the
  continuation contacts in gray.

Usage (bolt conda env, repo root):
    MPLCONFIGDIR=/tmp/mpl python experiments/stone_course/terrain_for_trajectory.py \
        dashboard/trajectories/foothold_rollout/foothold_seed29_0.json.gz \
        --seed 129 \
        --out-npz dashboard/trajectories/foothold_rollout/terrain_seed29.npz \
        --out-image /home/ubuntu/bolt_baselines/images/curriculum_terrain_hero_mid.png
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "stone_course"))

from terrain_from_curriculum import (  # noqa: E402
    draw_footholds,
    fit_terrain,
    shaded_surface,
)


def slab_tops_from_trajectory(traj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(course_tops, all_tops) in viz coords (x fwd, lateral, top height).

    course_tops are the first frame's slabs - the fixed visualized course as
    installed after the launch snap. all_tops adds every continuation slab
    position that ever appeared while the walker kept going.
    """
    with gzip.open(traj_path, "rt") as stream:
        frames = json.load(stream)

    def viz(collider):
        x, y, z = collider["pos"]
        return (x, z, y + collider["scale"][1])

    course, seen = [], {}
    for collider in frames[0]["colliders"]:
        if collider["name"].startswith("stone_"):
            course.append(viz(collider))
    for frame in frames:
        for collider in frame.get("colliders", []):
            if collider["name"].startswith("stone_"):
                point = viz(collider)
                seen[tuple(round(v, 4) for v in point)] = point
    all_tops = sorted(seen.values(), key=lambda p: p[0])
    return np.array(course), np.array(all_tops)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--seed", type=int, default=129,
                        help="heightfield seed (foothold seed + 100 by convention)")
    parser.add_argument("--cell", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=3.0,
                        help="terrain apron beyond the contacts (m)")
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--out-image", type=Path, required=True)
    args = parser.parse_args()

    course, all_tops = slab_tops_from_trajectory(args.trajectory)
    print(f"course slabs: {len(course)}, total slab positions: {len(all_tops)}")

    gx, gy, Z = fit_terrain(
        all_tops, cell=args.cell, margin=args.margin, seed=args.seed
    )
    ix = np.clip(np.rint((all_tops[:, 0] - gx[0]) / args.cell).astype(int), 0, len(gx) - 1)
    iy = np.clip(np.rint((all_tops[:, 1] - gy[0]) / args.cell).astype(int), 0, len(gy) - 1)
    residual = np.abs(all_tops[:, 2] - Z[iy, ix]).max()
    print(f"max |terrain - slab top| at contacts: {residual * 100:.2f} cm")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz, gx=gx, gy=gy, Z=Z,
        course_footholds=course, all_footholds=all_tops, seed=args.seed,
    )
    print("wrote", args.out_npz)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 7), facecolor="#0e1116")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0e1116")
    shaded_surface(ax, gx, gy, Z, stride=2, zoom=2.1)
    continuation = all_tops[len(course):]
    if len(continuation):
        ax.scatter(continuation[:, 0], continuation[:, 1], continuation[:, 2] + 0.03,
                   c="#c7cdd4", s=18, marker="s", depthshade=False, zorder=11,
                   edgecolors="#6a7480", linewidths=0.5)
    draw_footholds(ax, course, size=55, path=True)
    # A long course renders as a thin ribbon from the stock stage view; look
    # more nearly down the course axis so it fills the frame.
    ax.view_init(elev=28, azim=-49)
    ax.set_axis_off()
    ax.set_title(
        "Mid-stage curriculum as natural terrain - red: the 14 sampled "
        "footholds, gray: continuation contacts; the rollout video walks "
        "this exact surface",
        color="#e6e6e6", fontsize=13,
    )
    args.out_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_image, dpi=140, facecolor="#0e1116", bbox_inches="tight")
    print("wrote", args.out_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
