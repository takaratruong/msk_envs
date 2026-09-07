"""Reverse terrain generation: curriculum parameters -> footholds -> terrain.

g1-terrain-depth fits natural terrain TO a motion's foot contacts. This runs
the pipeline the other way for visualization: sample foothold placements from
the stone-course curriculum distribution (StoneCourseSpec.sample_positions is
the foothold sampler - slab tops are the intended contacts), then fit a
natural PFNN-style heightfield through those footholds using the same fBm +
RBF-correction machinery. The result shows the "equivalent natural terrain"
a curriculum stage corresponds to, with the footholds guaranteed on-surface.

Usage (bolt conda env, repo root):
    MPLCONFIGDIR=/tmp/mpl python experiments/stone_course/terrain_from_curriculum.py \
        --out-dir /home/ubuntu/bolt_baselines/images
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, "/home/ubuntu/g1-terrain-depth")

import torch  # noqa: E402

from msk_envs.envs.env_stone_course import StoneCourseSpec  # noqa: E402


# ---- lifted from g1_terrain_builder.terrain_gen (pure numpy, no deps) ------
def natural_heightfield(nx, ny, seed, cell=0.04, amp=0.16, style="mixed"):
    rng = np.random.default_rng(int(seed))
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    radial = np.sqrt(fx ** 2 + fy ** 2)
    radial[0, 0] = 1e-6
    beta = rng.uniform(2.2, 3.2)
    spec = radial ** (-beta / 2.0)
    spec *= np.exp(-(radial / rng.uniform(0.12, 0.22)) ** 2)
    phase = rng.uniform(0, 2 * np.pi, (ny, nx))
    field = np.fft.ifft2(spec * np.exp(1j * phase)).real
    field -= field.min()
    if field.max() > 1e-9:
        field /= field.max()
    h = field * amp * rng.uniform(0.7, 1.3)
    if style in ("slope", "mixed") and rng.random() < 0.6:
        gx = rng.uniform(-0.05, 0.05)
        gy = rng.uniform(-0.05, 0.05)
        xs = np.arange(nx)[None, :] * cell
        ys = np.arange(ny)[:, None] * cell
        h = h + gx * xs + gy * ys
    h -= h.min()
    return h


def rbf_correction(contacts_xy, residual, gx, gy, L=0.35):
    X, Y = np.meshgrid(gx, gy)
    GP = np.stack([X.ravel(), Y.ravel()], -1)
    d2 = ((GP[:, None, 0] - contacts_xy[None, :, 0]) ** 2
          + (GP[:, None, 1] - contacts_xy[None, :, 1]) ** 2)
    w = np.exp(-d2 / (2.0 * L * L))
    return ((w @ residual) / (w.sum(1) + 1e-12)).reshape(len(gy), len(gx))


# ---- curriculum stage -> footholds -> fitted terrain ------------------------
def curriculum_spec() -> StoneCourseSpec:
    return StoneCourseSpec(
        num_stones=14,
        step_length_range=(0.40, 1.50),
        lateral_jitter=0.10,
        slab_size=(0.36, 0.10, 0.36),
        top_height=0.45,
        top_height_range=(0.20, 1.05),
        elevation_angle_max_degrees=50.0,
        yaw_angle_max_degrees=20.0,
        surface_tilt_max_degrees=20.0,
        lookahead=4,
        alternating_lateral_offset=0.12,
    )


def sample_footholds(spec, stage, seed):
    """Slab centers/tops from the curriculum distribution at one stage.

    Returns (N,3) foothold points: x forward, y lateral, z = slab top height,
    already scaled by the stage's platform height_scale.
    """
    g = torch.Generator().manual_seed(seed)
    positions = spec.sample_positions(
        1, "cpu", g,
        step_length_max=stage["step_length_max"],
        elevation_angle_max_degrees=stage["elevation_deg"],
        yaw_angle_max_degrees=stage["yaw_deg"],
        height_scale=stage.get("height_scale", 1.0),
    )[0].numpy()
    # env axes: x fwd, y up, z lateral -> visualization: (x, lateral, up)
    tops = positions[:, 1] + spec.half_extents[1]
    return np.stack([positions[:, 0], positions[:, 2], tops], -1)


def fit_terrain(footholds, cell=0.05, margin=1.2, seed=0, amp=0.30, L=0.35, passes=12):
    """Natural heightfield RBF-fitted through the footholds (z at x,y).

    The normalized-RBF correction is a smoother, not an interpolant, so one
    pass leaves centimetre-to-decimetre gaps at the contacts (and a narrow L
    turns them into visible spikes). Iterating the correction on the
    remaining contact residual converges to an exact fit while keeping the
    wide, natural bump shape of L~0.35.
    """
    xy = footholds[:, :2]
    xmin, ymin = xy.min(0) - margin
    xmax, ymax = xy.max(0) + margin
    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    ny = int(np.ceil((ymax - ymin) / cell)) + 1
    gx = xmin + np.arange(nx) * cell
    gy = ymin + np.arange(ny) * cell
    Z = natural_heightfield(nx, ny, seed, cell=cell, amp=amp)
    ix = np.clip(np.rint((xy[:, 0] - xmin) / cell).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((xy[:, 1] - ymin) / cell).astype(int), 0, ny - 1)
    for _ in range(passes):
        resid = footholds[:, 2] - Z[iy, ix]
        Z = Z + rbf_correction(xy, resid, gx, gy, L=L)
    return gx, gy, Z


STAGES = [
    ("stage0_start", dict(step_length_max=0.70, elevation_deg=0.0,
                          yaw_deg=0.0, height_scale=0.10),
     "curriculum floor: 0.40-0.70 m gaps, flat, platform 10%"),
    ("stage1_fullheight", dict(step_length_max=0.70, elevation_deg=0.0,
                               yaw_deg=0.0, height_scale=1.0),
     "full height, easy gaps: 0.40-0.70 m, flat"),
    ("stage2_mid", dict(step_length_max=1.10, elevation_deg=20.0,
                        yaw_deg=8.0, height_scale=1.0),
     "mid terrain: gaps to 1.10 m, elevation ±20°, yaw ±8°"),
    ("stage3_max", dict(step_length_max=1.50, elevation_deg=50.0,
                        yaw_deg=20.0, height_scale=1.0),
     "max difficulty: gaps to 1.50 m, elevation ±50°, yaw ±20°"),
]


def shaded_surface(ax, gx, gy, Z, stride=1, zoom=1.0):
    """Light-source-shaded terrain surface (metric axes, no z exaggeration)."""
    from matplotlib import cm
    from matplotlib.colors import LightSource

    X, Y = np.meshgrid(gx, gy)
    # Push the color range up so the terrain cmap's low "water" band never
    # shows: the lowest ground stays green rather than reading as lakes.
    zmin, zmax = Z.min(), Z.max()
    span = max(zmax - zmin, 1e-6)
    ls = LightSource(azdeg=315, altdeg=50)
    rgb = ls.shade(Z, cmap=cm.terrain, blend_mode="soft", vert_exag=2.0,
                   vmin=zmin - 0.45 * span, vmax=zmax + 0.10 * span)
    ax.plot_surface(X, Y, Z, facecolors=rgb, linewidth=0, antialiased=True,
                    rstride=stride, cstride=stride, shade=False)
    # True metric proportions so slopes look walkable, not alpine.
    ax.set_box_aspect((np.ptp(gx), np.ptp(gy), max(np.ptp(Z), 0.3)), zoom=zoom)


def draw_footholds(ax, fh, size=45, path=False):
    ax.computed_zorder = False  # keep contact markers visible on the surface
    if path:
        ax.plot(fh[:, 0], fh[:, 1], fh[:, 2] + 0.03, c="#ffb347", lw=1.6,
                alpha=0.95, zorder=11)
    ax.scatter(fh[:, 0], fh[:, 1], fh[:, 2] + 0.03, c="#ff3355", s=size,
               marker="s", depthshade=False, edgecolors="#ffd7de",
               linewidths=0.7, zorder=12)


def render(spec, out_dir: Path, seed=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(22, 6.6), facecolor="#0e1116")
    for i, (name, stage, caption) in enumerate(STAGES):
        fh = sample_footholds(spec, stage, seed=seed + i)
        gx, gy, Z = fit_terrain(fh, seed=seed + 100 + i)
        on_surf = np.abs(fh[:, 2] - Z[
            np.clip(np.rint((fh[:, 1] - gy[0]) / (gy[1] - gy[0])).astype(int), 0, len(gy) - 1),
            np.clip(np.rint((fh[:, 0] - gx[0]) / (gx[1] - gx[0])).astype(int), 0, len(gx) - 1),
        ]).max()
        print(f"{name}: max |terrain - foothold| at contacts = {on_surf * 100:.2f} cm")
        ax = fig.add_subplot(1, 4, i + 1, projection="3d", facecolor="#0e1116")
        shaded_surface(ax, gx, gy, Z, stride=2, zoom=1.35)
        draw_footholds(ax, fh, size=40)
        ax.set_title(caption, color="#e6e6e6", fontsize=10, pad=0)
        ax.view_init(elev=38, azim=-60)
        ax.set_axis_off()
    fig.suptitle(
        "Equivalent natural terrain fitted through curriculum footholds "
        "(red squares = slab-top contacts sampled from the training distribution)",
        color="#8b97a5", fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / "curriculum_terrain_stages.png"
    fig.savefig(out, dpi=130, facecolor="#0e1116", bbox_inches="tight")
    print("wrote", out)

    # A single large hero view of the max stage.
    fh = sample_footholds(spec, STAGES[3][1], seed=seed + 3)
    gx, gy, Z = fit_terrain(fh, seed=seed + 103, cell=0.04)
    fig2 = plt.figure(figsize=(15, 8), facecolor="#0e1116")
    ax = fig2.add_subplot(111, projection="3d", facecolor="#0e1116")
    shaded_surface(ax, gx, gy, Z, stride=1, zoom=1.38)
    draw_footholds(ax, fh, size=60, path=True)
    ax.view_init(elev=32, azim=-64)
    ax.set_axis_off()
    ax.set_title("Max-difficulty curriculum as natural terrain - foothold path in orange",
                 color="#e6e6e6", fontsize=13)
    out2 = out_dir / "curriculum_terrain_hero.png"
    fig2.savefig(out2, dpi=140, facecolor="#0e1116", bbox_inches="tight")
    print("wrote", out2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "dashboard" / "terrain_viz")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render(curriculum_spec(), args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
