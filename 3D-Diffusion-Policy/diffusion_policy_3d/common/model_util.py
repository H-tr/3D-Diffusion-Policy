from termcolor import cprint

import numpy as np
from scipy.interpolate import CubicSpline
from numpy.linalg import norm


def print_params(model):
    """
    Print the number of parameters in each part of the model.
    """    
    params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        part_name = name.split('.')[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
        params_dict[part_name] += param.numel()

    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Number of parameters: {all_num_param / 1e6:.4f}M', 'cyan')
    for part_name, num_params in params_dict.items():
        cprint(f'   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})', 'cyan')
    cprint(f'----------------------------------', 'cyan')

def interpolate_trajectory(trajectory, num_inter_points=3, start_ratio=0.3):
    """
    Input:
        trajectory: list of points in x, y, z, qx, qy, qz, qw format
        num_inter_points: number of interpolated points between each pair of points
        start_ratio: the first few points (start_ratio * len(trajectory)) will not be interpolated
    Output:
        interpolated_trajectory: list of interpolated trajectory points in x, y, z, qx, qy, qz, qw format
    """
    trajectory = np.array(trajectory)
    n_points = len(trajectory)
    n_start_points = int(start_ratio * n_points)

    # Split position and orientation
    positions = trajectory[:, :3]
    quaternions = trajectory[:, 3:]

    # Initialize output trajectory with non-interpolated starting points
    interpolated_trajectory = list(trajectory[:n_start_points])

    def slerp(q1, q2, t):
        """Spherical Linear Interpolation for quaternions"""
        # Ensure unit quaternions
        q1 = q1 / norm(q1)
        q2 = q2 / norm(q2)

        # Calculate cosine of angle between quaternions
        dot = np.dot(q1, q2)

        # If quaternions are very close, linear interpolation is fine
        if dot > 0.9995:
            return q1 + t * (q2 - q1)

        # If dot product is negative, negate one quaternion to take shorter path
        if dot < 0:
            q2 = -q2
            dot = -dot

        dot = min(1.0, max(-1.0, dot))  # Clamp to [-1, 1]
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        if sin_theta < 1e-6:  # Prevent division by zero
            return q1

        # Perform SLERP
        ratio1 = np.sin((1 - t) * theta) / sin_theta
        ratio2 = np.sin(t * theta) / sin_theta

        return q1 * ratio1 + q2 * ratio2

    # Generate time parameters for interpolation
    original_times = np.linspace(0, 1, n_points - n_start_points)

    # Create cubic spline for positions
    cs_x = CubicSpline(original_times, positions[n_start_points:, 0])
    cs_y = CubicSpline(original_times, positions[n_start_points:, 1])
    cs_z = CubicSpline(original_times, positions[n_start_points:, 2])

    # Generate interpolation points
    for i in range(n_start_points, n_points - 1):
        # Current and next point
        curr_quat = quaternions[i]
        next_quat = quaternions[i + 1]

        # Generate interpolated points
        for j in range(num_inter_points + 1):
            t = float(j) / (num_inter_points + 1)

            # Interpolate position using cubic spline
            local_time = (i - n_start_points + t) / (n_points - n_start_points - 1)
            pos_x = cs_x(local_time)
            pos_y = cs_y(local_time)
            pos_z = cs_z(local_time)

            # Interpolate orientation using SLERP
            quat = slerp(curr_quat, next_quat, t)

            # Add interpolated point
            interpolated_point = np.concatenate(([pos_x, pos_y, pos_z], quat))
            interpolated_trajectory.append(interpolated_point)

    # Add final point
    interpolated_trajectory.append(trajectory[-1])

    return np.array(interpolated_trajectory)
