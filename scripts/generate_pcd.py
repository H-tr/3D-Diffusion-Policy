import os
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import psutil

def create_pointcloud(rgb_path, depth_path, save_dir, frame_idx):
    """
    Create pointcloud from RGB and depth images.
    """
    # Read RGB and depth images
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
        
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth = np.load(depth_path)
    if depth is None:
        raise ValueError(f"Failed to read depth data: {depth_path}")
    
    # Handle depth data type
    if np.issubdtype(depth.dtype, np.floating):
        # If depth values are floats, multiply by 1000 and convert to int
        depth = (depth * 1000).astype(np.uint16)
    else:
        depth = depth.astype(np.uint16)
    
    # Handle distortion
    D = np.array([
        0.11890428513288498,
        -2.5930066108703613,
        0.0006491912645287812,
        -0.0004916685866191983,
        1.8071327209472656,
        0.0036302555818110704,
        -2.3954696655273438,
        1.7094197273254395
    ])
    
    # Create camera intrinsic matrix
    K = np.array([
        [608.0169677734375, 0.0, 641.7816162109375],
        [0.0, 607.9260864257812, 363.21063232421875],
        [0.0,0.0, 1.0]
    ])
    
    # Undistort RGB image
    rgb_undistorted = cv2.undistort(rgb, K, D)
    
    # Create remapping for undistortion
    height, width = depth.shape
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), cv2.CV_32FC1)
    
    # Remap depth image
    depth_undistorted = cv2.remap(
        depth, map1, map2, cv2.INTER_NEAREST
    )
    
    # Create Open3D images
    rgb_image = o3d.geometry.Image(rgb_undistorted)
    depth_image = o3d.geometry.Image(depth_undistorted)
    
    # Define camera intrinsics (using undistorted parameters)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    fx = K[0, 0]  # 972.8271484375
    fy = K[1, 1]  # 972.6817016601562
    cx = K[0, 2]  # 1027.150634765625
    cy = K[1, 2]  # 773.43701171875
    width = rgb.shape[1]
    height = rgb.shape[0]
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, 
        depth_image,
        depth_scale=1000.0,  # Scale for converting depth to meters
        depth_trunc=3.0,     # Maximum depth in meters
        convert_rgb_to_intensity=False
    )
    
    # Generate pointcloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic
    )
    
    # Convert to numpy array
    points = np.asarray(pcd.points)  # Nx3 XYZ coordinates
    colors = np.asarray(pcd.colors)  # Nx3 RGB values in [0,1]
    
    # Check for NaNs or Infs in points and colors
    valid_points_mask = np.all(np.isfinite(points), axis=1)
    valid_colors_mask = np.all(np.isfinite(colors), axis=1)
    valid_mask = valid_points_mask & valid_colors_mask

    # Filter out invalid points and colors
    points = points[valid_mask]
    colors = colors[valid_mask]

    # Ensure colors are within [0,1]
    colors = np.clip(colors, 0.0, 1.0)
    
    # Handle empty point cloud after filtering
    if points.shape[0] == 0:
        print(f"Warning: No valid points remain after filtering for frame {frame_idx}.")
        return  # Skip saving this frame or handle accordingly
    
    # Verify shapes and values
    assert points.shape[1] == 3, f"Expected XYZ points to be Nx3, got shape {points.shape}"
    assert colors.shape[1] == 3, f"Expected RGB colors to be Nx3, got shape {colors.shape}"
    assert points.shape[0] == colors.shape[0], f"Mismatch in number of points and colors"
    
    # Combine points and colors into a single array
    pointcloud_data = np.concatenate([points, colors], axis=1)  # [x, y, z, r, g, b]
    
    # Save pointcloud
    output_path = os.path.join(save_dir + "/pcd", f'pcd_{frame_idx}.npy')
    np.save(output_path, pointcloud_data)

def main():
    # 1. Input is a list of task names that user defined (manually input)
    user_defined_tasks = ['hammer']  # Replace with your task names

    root_dir = Path("/data/home/tianrun/3D-Diffusion-Policy/data/kortex_data")

    frame_pairs = []

    for task_name in user_defined_tasks:
        task_folder = root_dir / task_name
        if not task_folder.is_dir():
            raise FileNotFoundError(f"Task folder not found: {task_folder}")
        print(f"Processing task: {task_name}")

        # 2. In each task, search how many episode_{i} are there
        no_episodes = len(list(task_folder.glob("episode_*")))

        # 3. For i in range(len(num_episodes)), current_folder = root_dir/episode_{i}
        for i in range(no_episodes):
            current_folder = task_folder / f"episode_{i}"
            print(f"Processing episode_{i} in task {task_name}")

            # 4. In current folder, find numbers of rgb, numbers of depth, check if they are the same
            no_frames = len(list((current_folder / "rgb").glob("rgb_*")))
            # If current folder doesn't have pcd, create one
            if not (current_folder / "pcd").is_dir():
                os.makedirs(current_folder / "pcd")
            for frame_idx in range(no_frames):
                rgb_path = current_folder / "rgb" / f"rgb_{frame_idx}.jpg"
                depth_path = current_folder / "depth" / f"depth_{frame_idx}.npy"
                if not rgb_path.exists():
                    raise FileNotFoundError(f"RGB file not found: {rgb_path}")
                if not depth_path.exists():
                    raise FileNotFoundError(f"Depth file not found: {depth_path}")
                
                frame_pairs.append((str(rgb_path), str(depth_path), str(current_folder), frame_idx))
                
    total_frames = len(frame_pairs)
    print(f"Total frames to process: {total_frames}")

    # 6. Multiprocessing call the create_pointcloud
    if total_frames == 0:
        print("No frames to process!")
        return

    success_count = 0
    error_count = 0
    error_frames = []

    # Use number of CPU cores minus 2 for worker processes
    num_workers = max(1, psutil.cpu_count(logical=True) - 2)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for args in frame_pairs:
            futures.append(executor.submit(create_pointcloud, *args))

        # Monitor progress
        for i, future in enumerate(tqdm(futures, desc="Processing frames", leave=False)):
            try:
                future.result()
                success_count += 1
            except Exception as e:
                error_count += 1
                error_frames.append(i)
                print(f"Error processing frame {i}:")
                print(str(e))

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {total_frames}")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {error_count}")

if __name__ == "__main__":
    # Disable OpenCV's internal multithreading to avoid conflicts
    cv2.setNumThreads(1)
    main()
