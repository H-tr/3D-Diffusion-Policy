import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time
import cv2
from pathlib import Path
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


extrinsics_matrix = np.array([[-9.99975517e-01, -5.96775278e-03, -3.65387159e-03, 3.19190930e-01],
                                [ 6.32579231e-04,  4.42935136e-01, -8.96553437e-01, 4.15892902e-01],
                                [ 6.96883737e-03, -8.96533798e-01, -4.42920516e-01, 3.58429336e-01],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(points, use_cuda=True, step_index=1):
    
    num_points = 1024

    # scale
    point_xyz = points[..., :3]
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix)
    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz
    
    # save the point cloud for debug
    # np.save(f'/data/home/tianrun/3D-Diffusion-Policy/tmp/scaled_pcd_{step_index}.npy', points)
    
    WORK_SPACE = [
        [-0.25, 0.40],
        [-1.00, -0.20],
        [-0.40, 0.05]
    ]
    
     # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
    
    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    return points
   
def preproces_image(image):
    img_size = 84
    
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image



expert_data_path = '/data/home/tianrun/3D-Diffusion-Policy/data/kortex_data/pour_50/'
save_data_path = '/data/home/tianrun/3D-Diffusion-Policy/data/kortex_data/pour_50.zarr'

# Find all episode directories that contain rgb_*.jpg files
demo_dirs = []
for root, _, files in os.walk(expert_data_path):
    if any(f.startswith('rgb_') and f.endswith('.jpg') for f in files):
        demo_dirs.append(root)
demo_dirs = sorted(demo_dirs)

# storage
total_count = 0
img_arrays = []
point_cloud_arrays = []
depth_arrays = []
state_arrays = []
action_arrays = []
episode_ends_arrays = []


if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = 'y'
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)

for demo_dir in demo_dirs:
    cprint('Processing {}'.format(demo_dir), 'green')
    
    # Get RGB and depth files
    # rgb_files = sorted([f for f in os.listdir(demo_dir) if f.startswith('rgb_') and f.endswith('.jpg')])
    # depth_files = sorted([f for f in os.listdir(demo_dir) if f.startswith('depth_') and f.endswith('.npy')])
    # pcd_files = sorted([f for f in os.listdir(demo_dir) if f.startswith('pcd_') and f.endswith('.npy')])
    
    # rgb_len = len(rgb_files)
    # depth_len = len(depth_files)
    # pcd_len = len(pcd_files)
    
    rgb_len = len(list(Path(demo_dir).glob('rgb_*.jpg')))
    depth_len = len(list(Path(demo_dir).glob('depth_*.npy')))
    pcd_len = len(list(Path(demo_dir).glob('pcd_*.npy')))
    
    assert rgb_len == depth_len and depth_len == pcd_len, "RGB, depth, and pointcloud files are not in the same length."
    
    action_values = np.load(os.path.join(demo_dir, 'abs_ee_pose.npy'))
    # TODO: Current robot action: 3 position and 4 orientation. Convert it to 3 position and 6 rotation
    action_values = np.array([
        np.concatenate([
            action[:3],  # Keep position (x,y,z)
            quaternion_to_6d(action[3:])  # Convert quaternion to 6D rotation
        ])
        for action in action_values
    ])
    button_state_values = np.load(os.path.join(demo_dir, 'controller_btn_states.npy'))
    # TODO: The second value in each tensor is the button state. merge it with the action tensor
    action_values = np.array([
        np.concatenate([
            action_values[i],  # Position and rotation (9D)
            [button_state_values[i, 1]]  # Add button state (1D)
        ])
        for i in range(len(action_values))
    ])
    
    demo_length = rgb_len
    for step_idx in tqdm.tqdm(range(demo_length)):
        total_count += 1
        
        # Load RGB image
        obs_image = cv2.imread(os.path.join(demo_dir, f"rgb_{step_idx}.jpg"))
        obs_image = cv2.cvtColor(obs_image, cv2.COLOR_BGR2RGB)
        
        # Load depth
        obs_depth = np.load(os.path.join(demo_dir, f"depth_{step_idx}.npy"))
        
        if np.issubdtype(obs_depth.dtype, np.floating):
            # Repalce nan with 0
            obs_depth[np.isnan(obs_depth)] = 0
            # If depth values are floats, multiply by 1000 and convert to int
            obs_depth = (obs_depth * 1000).astype(np.uint16)
        
        # Load pointcloud
        pcd_path = os.path.join(demo_dir, f"pcd_{step_idx}.npy")
        obs_pointcloud = np.load(pcd_path)
        
        # Process data
        obs_image = preproces_image(obs_image)
        obs_depth = preproces_image(np.expand_dims(obs_depth, axis=-1)).squeeze(-1)
        obs_pointcloud = preprocess_point_cloud(obs_pointcloud, use_cuda=True, step_index=step_idx)
        
        # Save the point cloud for debug
        # np.save(f'/data/home/tianrun/3D-Diffusion-Policy/tmp/pcd_{step_idx}.npy', obs_pointcloud)
        
        # Placeholder for state and action (you'll need to provide these)
        robot_state = action_values[step_idx]
        action = action_values[step_idx]
        
        # Store processed data
        img_arrays.append(obs_image)
        action_arrays.append(action)
        point_cloud_arrays.append(obs_pointcloud)
        depth_arrays.append(obs_depth)
        state_arrays.append(robot_state)
    
    episode_ends_arrays.append(total_count)




# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

img_arrays = np.stack(img_arrays, axis=0)
if img_arrays.shape[1] == 3: # make channel last
    img_arrays = np.transpose(img_arrays, (0,2,3,1))
point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
depth_arrays = np.stack(depth_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    raise NotImplementedError
zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# print shape
cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')
