import bolt
import torch


def main():
    model_path = "msk_envs/msk_models/athlete15.osim"
    function_path = "msk_envs/msk_models/athlete15paths_minimal.xml"
    load_result = bolt.load_model(
        model_path=model_path,
        n_worlds=1,
        integrator=bolt.IntegratorType.EULER_ADAPTIVE,
        requires_visuals=True,
        polynomial_data_path=function_path
    )
    m, d = load_result.model, load_result.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    renderer = bolt.create_renderer(
        load_result=load_result,
        renderer_type=bolt.RendererType.TILED,
        draw_visuals=True,
        draw_colliders=True,
        draw_muscles=True,
        draw_body_mass=False,
        draw_beams=True,
        draw_sites=False,
    )

    motion = torch.load("corrected_motion.pt")
    motion = torch.tensor(motion, device=device)
    ref_time, ref_frames = motion[:, 0], motion[:, 1:]
    num_frames = len(motion)

    joint_positions = bolt.joint_positions(d)
    for i in range(num_frames):
        joint_positions[0, :] = ref_frames[i, :]
        bolt.fk(m, d)
        renderer.render(m, d)
    return


if __name__ == "__main__":
    main()
