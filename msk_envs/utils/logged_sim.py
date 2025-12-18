from msk_envs.envs.env_base import MSKEnv
from msk_envs.utils.checkpoint_parser import parse_frame
from msk_envs.utils.checkpoint_outputter import create_animation_json, create_pdf_output

import os
import json
import torch


class LoggedSim:
    def __init__(
            self,
            envs: MSKEnv,
            max_episode_length: int,
            device: torch.device,
    ):
        self.envs = envs
        self.worlds_to_save = list(range(envs.num_worlds))
        n_worlds = envs.num_worlds
        n_worlds_to_save = len(self.worlds_to_save)
        assert n_worlds_to_save <= n_worlds
        assert min(self.worlds_to_save) >= 0
        assert max(self.worlds_to_save) < n_worlds

        # Build storage for things to track
        max_ep_len = max_episode_length
        self.finished = torch.zeros((n_worlds,),
                                    dtype=torch.bool, device=device)
        self.rewards = torch.zeros((n_worlds_to_save, max_ep_len),
                                   dtype=torch.float32, device=device)
        self.frame_data = [[] for _ in range(n_worlds_to_save)]
        self.episode_length = torch.zeros((n_worlds,),
                                          dtype=torch.int32, device=device)

        self.n_worlds = n_worlds
        self.n_worlds_to_save = n_worlds_to_save
        self.max_episode_length = max_ep_len
        self.curr_step = 0
        self.device = device

    def add_to_log(self):
        # Track rewards
        rew = self.envs.get_rewards()
        ind_not_finished = torch.where(self.finished == 0)[0]
        if len(ind_not_finished) == 0:
            return
        self.rewards[ind_not_finished, self.curr_step] = rew[ind_not_finished]
        self.episode_length[ind_not_finished] += 1

        times = self.envs.get_time()
        reward_dict = self.envs.get_scaled_reward_dict()

        for i in range(len(self.worlds_to_save)):
            idx_world = self.worlds_to_save[i]
            if self.finished[idx_world]:
                continue

            reward_data = {k: v[idx_world].item() for k, v in reward_dict.items()}
            frame = parse_frame(
                m=self.envs.m,
                d=self.envs.d,
                dof_id_lookup=self.envs.dof_id_lookup,
                muscle_id_lookup=self.envs.muscle_id_lookup,
                actuator_id_lookup=self.envs.actuator_id_lookup,
                visual_load_results=self.envs.visuals,
                world_id=idx_world,
                frame_time=times[idx_world].item(),
                reward_data=reward_data,
            )
            self.frame_data[i].append(frame)
        self.curr_step += 1
        return

    def step(self, actions: torch.Tensor):
        if self.finished.all():
            return True, None

        obs, rew, terminated, truncated, _ = self.envs.step(actions)
        done = (terminated + truncated).bool()
        self.finished = self.finished | done

        self.add_to_log()
        return False, obs

    def reset(self):
        self.curr_step = 0
        self.finished[:] = 0
        self.rewards[:] = 0
        self.episode_length[:] = 0
        self.frame_data = [[] for _ in range(self.n_worlds_to_save)]
        obs = self.envs.reset()
        return obs

    def get_rewards_mean(self):
        return self.rewards.sum(dim=1).float().mean()

    def get_episode_length_mean(self):
        return self.episode_length.float().mean()

    def get_obs(self):
        return self.envs._get_obs()

    def save_frame_data(self, out_folder: str, base_filename: str):
        """ Save the raw frame data as json files """
        os.makedirs(out_folder, exist_ok=True)
        for idx_world in self.worlds_to_save:
            out_file = os.path.join(out_folder,
                                    f"{base_filename}_{idx_world}.json")
            frame_data = self.frame_data[idx_world]
            frame_data = [frame.to_data_dict() for frame in frame_data]
            with open(out_file, 'w') as f:
                json.dump(frame_data, f)
            print("Saved frame data to", out_file)
        return

    def save_animation(self, out_folder: str, base_filename):
        """ Create the animation-ready json files """
        for idx_world in self.worlds_to_save:
            out_file = os.path.join(out_folder,
                                    f"{base_filename}_{idx_world}.json")
            frame_data = self.frame_data[idx_world]
            create_animation_json(frame_data, out_file)
            print("Saved animation to", out_file)
        return

    def save_analytics(self, out_folder: str, base_filename: str):
        # make a pdf with the plots
        for i in range(self.n_worlds_to_save):
            idx_world = self.worlds_to_save[i]
            out_file = os.path.join(out_folder,
                                    f"{base_filename}_{idx_world}.pdf")
            create_pdf_output(self.frame_data[i], out_file)
            print("Saved analytics to", out_file)
        return


def setup_axes(axs, title, xlabel, ylabel):
    axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
