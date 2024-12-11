from typing import Dict
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.policy.base_policy import BasePolicy

import numpy as np
import torch
import tqdm
from termcolor import cprint
from diffusion_policy_3d.env import KortexEnv

from diffusion_policy_3d.common.pytorch_util import dict_apply
import diffusion_policy_3d.common.logger_util as logger_util

from scipy.spatial.transform import Rotation as R

def quaternion_to_6d(q: np.ndarray) -> np.ndarray:
    """
    Using 6 DoF as representation of rotation
    https://arxiv.org/pdf/1812.07035
    """
    # Convert quaternion to rotation matrix
    r = R.from_quat(q)
    rot_matrix = r.as_matrix()  # 3x3 matrix

    # Take the first two columns
    m1 = rot_matrix[:, 0]
    m2 = rot_matrix[:, 1]

    # Flatten to 6D vector
    rot_6d = np.concatenate([m1, m2], axis=-1)
    return rot_6d

class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

class KortexRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        
        # n_action_steps = 1

        # Define the environment function
        def env_fn():
            return MultiStepWrapper(
                KortexEnv(
                    task_name=task_name,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_returns = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), 
                                    desc=f"Eval in Kortex {self.task_name} Env",
                                    leave=False, 
                                    mininterval=self.tqdm_interval_sec):

            # Start rollout
            obs = env.reset()
            policy.reset()

            done = False
            reward_sum = 0.0
            actual_step_count = 0

            while not done:
                # Since obs is batched over n_obs_steps, we keep all observations
                np_obs_dict = dict(obs)
                
                # print(f"obs: {obs}")
                
                # print(f"shape of agent_pos: {obs['agent_pos'].shape}")
                # print(f"shape of point_cloud: {obs['point_cloud'].shape}")

                # Device transfer
                obs_dict = dict_apply(np_obs_dict,
                                    lambda x: torch.from_numpy(x).to(
                                        device=device, dtype=dtype))

                # Run policy
                with torch.no_grad():
                    obs_dict_input = {}

                    # Process agent_pos
                    # Shape: (1, n_obs_steps, 10)
                    agent_pos = obs_dict['agent_pos'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = agent_pos

                    # Process point_cloud
                    # Shape: (1, n_obs_steps, N, 6)
                    point_cloud = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['point_cloud'] = point_cloud

                    # Call policy to get action
                    action_dict = policy.predict_action(obs_dict_input)
                    
                    # print(f"agent_pos: {agent_pos}")
                    # print(f"point_cloud: {point_cloud}")
                    print(f"action_dict: {action_dict}")

                # Device transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().cpu().numpy())

                action = np_action_dict['action'].squeeze(0)

                # Generate action sequence
                action_sequence = np.tile(action, (self.n_action_steps, 1))  # Shape (n_action_steps, action_dim)
                
                # print(f"action_sequence: {action_sequence}")

                # Step environment
                obs, reward, done, info = env.step(action_sequence)
                # obs = env.get_observation()
                reward_sum += reward
                actual_step_count += self.n_action_steps

            all_returns.append(reward_sum)

        returns_mean = np.mean(all_returns)

        # Log results
        log_data = dict()
        log_data['mean_returns'] = returns_mean

        cprint(f"Mean returns: {returns_mean:.3f}", 'green')

        # Clear out any residual data
        _ = env.reset()
        # Clear memory
        del env

        return log_data

    # def run(self, policy: BasePolicy):
    #     device = policy.device
    #     dtype = policy.dtype
    #     env = self.env

    #     all_returns = []

    #     # Start rollout
    #     _ = env.reset()
    #     policy.reset()

    #     done = False
    #     reward_sum = 0.0
    #     actual_step_count = 0
        
    #     states_npy = np.load("/home/rls/Documents/RobotIL/data/kortex_data/pour/episode_25/abs_ee_pose.npy")
    #     button_npy = np.load("/home/rls/Documents/RobotIL/data/kortex_data/pour/episode_25/controller_btn_states.npy")
        
    #     states_npy = np.array([
    #         np.concatenate([
    #             action[:3],  # Keep position (x,y,z)
    #             quaternion_to_6d(action[3:])  # Convert quaternion to 6D rotation
    #         ])
    #         for action in states_npy
    #     ])
        
    #     action_values = np.array([
    #         np.concatenate([
    #             states_npy[i],  # Position and rotation (9D)
    #             [button_npy[i, 1]]  # Add button state (1D)
    #         ])
    #         for i in range(len(states_npy))
    #     ])

    #     for step_idx in range(1, len(states_npy)):
    #         print(f"-----------------step_idx: {step_idx}---------------------")
    #         point_cloud_previous = np.load(f"/home/rls/Documents/RobotIL/data/kortex_data/pour/episode_25/pcd/pcd_{step_idx-1}.npy")
    #         point_cloud_current = np.load(f"/home/rls/Documents/RobotIL/data/kortex_data/pour/episode_25/pcd/pcd_{step_idx}.npy")
    #         pcd_croped_pre = env.process_point_cloud(point_cloud_previous)
    #         pcd_croped_cur = env.process_point_cloud(point_cloud_current)
    #         # Combine two point clouds in a new batch dimension
    #         pcd_croped = np.stack([pcd_croped_pre, pcd_croped_cur], axis=0)
            
    #         state_value = np.stack([action_values[step_idx - 1], action_values[step_idx]], axis=0)
            
    #         obs = {'agent_pos': state_value, 'point_cloud': pcd_croped}
            
    #         print(f"obs: {obs}")
            
    #         # Since obs is batched over n_obs_steps, we keep all observations
    #         np_obs_dict = dict(obs)

    #         # Device transfer
    #         obs_dict = dict_apply(np_obs_dict,
    #                             lambda x: torch.from_numpy(x).to(
    #                                 device=device, dtype=dtype))

    #         # Run policy
    #         with torch.no_grad():
    #             obs_dict_input = {}

    #             # Process agent_pos
    #             # Shape: (1, n_obs_steps, 10)
    #             agent_pos = obs_dict['agent_pos'].unsqueeze(0)
    #             obs_dict_input['agent_pos'] = agent_pos

    #             # Process point_cloud
    #             # Shape: (1, n_obs_steps, N, 6)
    #             point_cloud = obs_dict['point_cloud'].unsqueeze(0)
    #             obs_dict_input['point_cloud'] = point_cloud

    #             # Call policy to get action
    #             action_dict = policy.predict_action(obs_dict_input)
                
    #             # print(f"agent_pos: {agent_pos}")
    #             # print(f"point_cloud: {point_cloud}")
    #             # print(f"action_dict: {action_dict}")

    #         # Device transfer
    #         np_action_dict = dict_apply(action_dict,
    #                                     lambda x: x.detach().cpu().numpy())

    #         action = np_action_dict['action'].squeeze(0)

    #         # Generate action sequence
    #         action_sequence = np.tile(action, (self.n_action_steps, 1))  # Shape (n_action_steps, action_dim)
            
    #         # print(f"action_sequence: {action_sequence}")

    #         # Debugging
    #         # import pdb; pdb.set_trace()
    #         # Step environment
    #         obs, reward, done, info = env.step(action_sequence)
    #         reward_sum += reward
    #         actual_step_count += self.n_action_steps

    #         all_returns.append(reward_sum)

    #     returns_mean = np.mean(all_returns)

    #     # Log results
    #     log_data = dict()
    #     log_data['mean_returns'] = returns_mean

    #     cprint(f"Mean returns: {returns_mean:.3f}", 'green')

    #     # Clear out any residual data
    #     _ = env.reset()
    #     # Clear memory
    #     del env

    #     return log_data