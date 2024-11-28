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

                # Device transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().cpu().numpy())

                action = np_action_dict['action'].squeeze(0)

                # Generate action sequence
                action_sequence = np.tile(action, (self.n_action_steps, 1))  # Shape (n_action_steps, action_dim)

                # Step environment
                obs, reward, done, info = env.step(action_sequence)
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
