from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

def rot_6d_to_matrix(r6d):
    x_raw = r6d[0:3]
    y_raw = r6d[3:6]
    x = x_raw / np.linalg.norm(x_raw)
    y = y_raw - x * np.dot(x, y_raw)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    R_mat = np.stack([x, y, z], axis=1)
    return R_mat

def matrix_to_rot_6d(R_mat):
    x_axis = R_mat[:, 0]
    y_axis = R_mat[:, 1]
    r6d = np.concatenate([x_axis, y_axis], axis=0)
    return r6d

def convert_ee_delta_to_robot_frame(action, state):
    """
    Convert the delta pose in end-effector frame to robot frame.

    action: [Δx, Δy, Δz, rot_6d(6), gripper] (10D)
    state: [x, y, z, rot_6d(6), gripper] (10D) absolute EE pose in robot frame
    """

    # Extract position and orientation from state
    p_abs = state[0:3]
    r6d_abs = state[3:9]
    R_abs = rot_6d_to_matrix(r6d_abs)

    # Extract delta pose in EE frame
    delta_pos_ee = action[0:3]
    delta_r6d_ee = action[3:9]
    gripper_val = action[9]

    # Convert EE-frame delta rotation to matrix
    R_delta_ee = rot_6d_to_matrix(delta_r6d_ee)

    # Δp_robot = R_abs * Δp_ee
    delta_pos_robot = R_abs @ delta_pos_ee

    # R_delta_robot = R_abs * R_delta_ee * R_abs^T
    R_delta_robot = R_abs @ R_delta_ee @ R_abs.T

    # Convert back to 6D rotation
    delta_r6d_robot = matrix_to_rot_6d(R_delta_robot)

    # Construct action in robot frame
    action_robot = np.zeros(10, dtype=action.dtype)
    action_robot[0:3] = delta_pos_robot
    action_robot[3:9] = delta_r6d_robot
    action_robot[9] = gripper_val

    return action_robot

class KortexDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'img'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='gaussian', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32) # shape: (T, 10)
        point_cloud = sample['point_cloud'].astype(np.float32) # (T, 1024, 6)
        action = sample['action'].astype(np.float32) # (T, 10)

        # Convert each action from EE frame to robot frame
        for t in range(action.shape[0]):
            state_abs = agent_pos[t]    # absolute EE pose in robot frame
            action[t] = convert_ee_delta_to_robot_frame(action[t], state_abs)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos,     # T, 10
            },
            'action': action              # T, 10 (now in robot frame)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

