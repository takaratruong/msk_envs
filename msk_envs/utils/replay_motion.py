import bolt
import torch
import warp as wp
from msk_envs.utils.parse_mot import parse_mot

fk_graph = None


def fk(m, d):
    global fk_graph
    cuda = torch.cuda.is_available()

    if cuda and fk_graph is None:
        with wp.ScopedCapture() as capture:
            bolt.fk(m, d)
        fk_graph = capture.graph

    if cuda:
        wp.capture_launch(fk_graph)
        wp.synchronize()
    else:
        bolt.fk(m, d)
    return


def check_limits(
        load_result,
        joint_positions,
):
    limit_id_lookup = load_result.limit_id_lookup
    qpos_id_lookup = load_result.qpos_id_lookup

    for coord_name, (coord_lo, coord_hi) in limit_id_lookup.items():
        coord_qpos_id = qpos_id_lookup[coord_name]
        coord_value = joint_positions[0, coord_qpos_id].item()
        if coord_value < coord_lo or coord_value > coord_hi:
            print(
                f"Warning: Coordinate '{coord_name}' out of limits: "
                f"{coord_value:.4f} not in [{coord_lo:.4f}, {coord_hi:.4f}]"
            )


def main():
    motion_file = "msk_envs/msk_models/gt/motions/maxVerticalJump_3step.mot"
    model_path = "msk_envs/msk_models/gt/gt_model.osim"
    validate_limits = True
    load_result = bolt.load_model(
        model_path=model_path,
        n_worlds=1,
        integrator=bolt.IntegratorType.EULER_ADAPTIVE,
        requires_visuals=True,
        muscle_fn_path=None
    )
    m, d = load_result.model, load_result.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    renderer = bolt.create_renderer(
        load_result=load_result,
        renderer_type=bolt.RendererType.TILED,
        draw_visuals=True,
        draw_colliders=False,
        draw_muscles=True,
        draw_body_mass=False,
        draw_beams=True,
        draw_sites=True,
    )

    motion = parse_mot(motion_file, load_result, in_degrees=False)
    motion = torch.tensor(motion, device=device)
    ref_time, ref_frames = motion[:, 0], motion[:, 1:]
    print("Motion duration:", ref_time[-1] - ref_time[0])

    # resample the trajectory onto a uniform grid
    frame_dt = 1.0 / 10.0
    target_times = torch.arange(ref_time[0].item(), ref_time[-1].item(), frame_dt)

    joint_positions = bolt.joint_positions(d)
    for t in target_times:
        idx = torch.argmin(torch.abs(ref_time - t))  # Nearest reference frame

        joint_positions[0, :] = ref_frames[idx, :]
        fk(m, d)
        renderer.render(m, d)

        if validate_limits:
            check_limits(load_result, joint_positions)

    return


if __name__ == "__main__":
    main()
