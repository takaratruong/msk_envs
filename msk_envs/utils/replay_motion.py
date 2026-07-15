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


def main():
    motion_file = "/home/marth/Downloads/Data/GT_motionData/maxVerticalJump_3step/IK/maxvert_3step_1_1_segment_0_ik.mot"
    model_path = "/home/marth/Documents/msk_envs/msk_envs/msk_models/rajagopal/RajagopalLaiUhlrich2023.osim"
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
        draw_muscles=False,
        draw_body_mass=False,
        draw_beams=True,
        draw_sites=False,
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

    return


if __name__ == "__main__":
    main()
