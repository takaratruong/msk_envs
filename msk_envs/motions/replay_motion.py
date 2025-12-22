import msk_warp
from msk_envs.utils.parse_mot import parse_mot
import torch


def main():
    motion_file = "msk_envs/motions/pred_sprint.mot"
    model_path = "msk_envs/msk_models/model_motor_arms_foot_contact.osim"
    load_result = msk_warp.load_model(model_path, 1)
    m, d = load_result.model, load_result.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    renderer = msk_warp.create_renderer(
        load_result=load_result,
        renderer_type=msk_warp.RendererType.TILED,
        draw_visuals=True,
        draw_colliders=True,
        draw_muscles=True
    )

    data, col_names = parse_mot(motion_file, model_path)
    ref_motion = torch.tensor(data, device=device)
    ref_time, ref_frames = ref_motion[0, :], ref_motion[1:, :]

    joint_positions = msk_warp.joint_positions(d)
    body_id_lookup = load_result.body_id_lookup
    id_to_body = {v: k for k, v in body_id_lookup.items()}
    for i in range(data.shape[1]):
        joint_positions[0, :] = ref_frames[:, i]
        msk_warp.fk(m, d)

        renderer.render(m, d)
    return


if __name__ == "__main__":
    main()
