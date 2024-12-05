import gym
import numpy as np
import torch
import rospy
import json
import os
import time
import struct

from kortex_driver.srv import SendGripperCommand, SendGripperCommandRequest
from kortex_driver.msg import BaseCyclic_Feedback, GripperMode, GripperCommand, Finger
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, JointState, PointCloud2
from cv_bridge import CvBridge
import cv2

# Import the PD controller and utility functions
from diffusion_policy_3d.env.kortex.pd_controller import KortexPDController
from diffusion_policy_3d.common.model_util import interpolate_trajectory

import torchvision
import pytorch3d.ops as torch3d_ops
import sensor_msgs.point_cloud2 as pc2  # For PointCloud2 processing

from gym import spaces

my_path = os.path.dirname(os.path.abspath(__file__))


class KortexEnv(gym.Env):
    """
    OpenAI Gym environment for controlling the Kortex robot arm.

    This environment allows for interaction with the Kortex robot using the Gym interface.
    It produces observations suitable for a 3D diffusion policy, including images, depth maps,
    point clouds, and robot state with rotation represented in 6D.
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(
        self,
        task_name,
        robot_name="my_gen3",
        traj_representation="abs_ee_pose",
        is_inference=False,
        num_points=1024,
        img_size=84,
    ):
        super(KortexEnv, self).__init__()

        # Initialize ROS node
        rospy.init_node("kortex_env", anonymous=True)

        self.task = task_name
        self.step_count = 0
        self.accumulated_pose = None  # Initialize accumulated pose
        self.f = 10  # Frequency
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.point_cloud_raw = None  # Raw point cloud data
        self.traj_representation = traj_representation
        self.num_points = num_points
        self.img_size = img_size

        # Load task parameters
        self.load_parameters()

        # Initialize PD controller
        self.pd_controller = KortexPDController(robot_name=robot_name, rate=100)

        # Gripper service client
        rospy.wait_for_service(f"/{robot_name}/base/send_gripper_command")
        self.gripper_service = rospy.ServiceProxy(
            f"/{robot_name}/base/send_gripper_command", SendGripperCommand
        )
        print("Kortex Robot gripper service client connected!")

        # Publisher for desired poses
        self.desired_pose_pub = rospy.Publisher(
            "/desired_pose", PoseStamped, queue_size=10
        )

        # Initialize current_pose and other necessary variables
        self.current_pose = None
        self.current_velocity = None
        self.last_position_error = np.zeros(3)
        self.last_angular_error = np.zeros(3)
        self.last_time = rospy.Time.now()

        # Initialize joint data
        self.joint_positions = None
        self.joint_velocities = None
        self.joint_torques = None
        self.observed_btn_states = [[0, 0, 0, 0, 0, 0]]

        # Subscribers
        rospy.Subscriber(
            f"/{robot_name}/base_feedback",
            BaseCyclic_Feedback,
            self.base_feedback_callback,
        )
        rospy.Subscriber(
            f"/{robot_name}/joint_states",
            JointState,
            self.joint_states_callback,
        )
        rospy.Subscriber("/rgb/image_raw", Image, self.kinect_rgb_callback)
        rospy.Subscriber("/depth_to_rgb/image", Image, self.kinect_depth_callback)
        rospy.Subscriber("/points2", PointCloud2, self.pointcloud_callback)

        # Wait for the initial pose to be received
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Define extrinsics matrix (from your reference)
        self.extrinsics_matrix = np.array([
            [-0.9995885060604603, 0.02737040572434423, -0.008583673007599053, 0.3145712335775515], 
            [0.019431110570558213, 0.42596851329953994, -0.9045293016919754, 0.5144627713101816], 
            [-0.02110095954717077, -0.9043238836655059, -0.4263250672178739, 0.3630794805194548], 
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Define workspace boundaries
        self.WORK_SPACE = [
            [0.08, 0.72],
            [-0.49, 0.31],
            [-0.02, 0.43]
        ]

        # Define action and observation spaces

        # Action space: [delta_translation (3), delta_rotation (6), gripper command (1)]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(10,),  # 3 for translation, 6 for rotation (6D), 1 for gripper
            dtype=np.float32,
        )

        # Observation space
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),  # Translation (3) + Rotation (6) + Gripper (1)
                dtype=np.float32,
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 6),  # x, y, z, r, g, b
                dtype=np.float32,
            ),
        })

        # Start the PD controller
        self.pd_controller.start()

        # Go to home pose
        self.go_homepose()

    def load_parameters(self):
        """
        Loads task parameters from prepare_conf.json
        """
        # Load task configuration from prepare_conf.json
        prepare_config_path = os.path.join(my_path, "prepare_conf.json")
        prepare_conf = {}

        if os.path.exists(prepare_config_path):
            with open(prepare_config_path, "r") as f:
                try:
                    prepare_conf = json.load(f)
                except json.JSONDecodeError as e:
                    rospy.logerr(
                        f"JSONDecodeError while parsing prepare_conf.json: {e}"
                    )
                    prepare_conf = {}
        else:
            rospy.logerr(f"Configuration file {prepare_config_path} does not exist.")
            prepare_conf = {}

        self.task_config = prepare_conf.get(self.task, None)
        if self.task_config is None:
            rospy.logerr(
                f"No configuration found for task '{self.task}' in prepare_conf."
            )
            raise ValueError(
                f"No configuration found for task '{self.task}' in prepare_conf."
            )

        self.max_step = self.task_config["max_len"]
        self.speed = self.task_config["speed"]
        self.home_pose = self.task_config["arm_joints"]  # Load home pose
        self.home_ee_pose = self.task_config["ee_pose"]  # Load home end-effector pose

    def base_feedback_callback(self, msg):
        # Extract current end-effector pose from the message
        self.current_pose = {
            "position": np.array(
                [msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z]
            ),
            "orientation": R.from_euler(
                "xyz",
                [
                    np.deg2rad(msg.base.tool_pose_theta_x),
                    np.deg2rad(msg.base.tool_pose_theta_y),
                    np.deg2rad(msg.base.tool_pose_theta_z),
                ],
            ).as_quat(),  # Convert from degrees to radians
        }

    def joint_states_callback(self, msg):
        # Extract joint data
        joint_positions = []
        joint_velocities = []
        joint_torques = []
        for name, position, velocity, torque in zip(
            msg.name, msg.position, msg.velocity, msg.effort
        ):
            joint_positions.append(position)
            joint_velocities.append(velocity)
            joint_torques.append(torque)
        self.joint_positions = np.array(joint_positions)
        self.joint_velocities = np.array(joint_velocities)
        self.joint_torques = np.array(joint_torques)

    def kinect_rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("Failed to receive RGB image: %s", e)

    def kinect_depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            self.depth_image = np.nan_to_num(self.depth_image)
        except Exception as e:
            rospy.logerr("Failed to receive depth image: %s", e)

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 message to numpy array
        self.point_cloud_raw = self.pointcloud2_to_array(msg)

    def pointcloud2_to_array(self, cloud_msg):
        # Extract field names
        field_names = [field.name for field in cloud_msg.fields]
        # print(f"PointCloud2 field names: {field_names}")  # For debugging
        # Check if 'rgb' field exists
        has_rgb = 'rgb' in field_names

        # Use the point_cloud2.read_points function
        point_list = []
        for point in pc2.read_points(cloud_msg, field_names=field_names, skip_nans=True):
            x, y, z = point[0], point[1], point[2]
            if has_rgb:
                # Extract RGB values
                rgb = point[field_names.index('rgb')]
                # The RGB data is packed into a float32 or uint32
                # Use struct to unpack it
                if isinstance(rgb, float):
                    # For float32 packed RGB data
                    s = struct.pack('>f' , rgb)
                    i = struct.unpack('>l', s)[0]
                else:
                    # For uint32 packed RGB data
                    i = int(rgb)
                # Extract the individual RGB components
                r = (i >> 16) & 0x0000ff
                g = (i >> 8)  & 0x0000ff
                b = (i)       & 0x0000ff
                point_list.append([x, y, z, r, g, b])
            else:
                point_list.append([x, y, z])
        point_array = np.array(point_list, dtype=np.float32)
        return point_array

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (np.ndarray): Action to be taken. It's an array:
                - First 3 are delta translation
                - Next 6 are delta rotation (6D representation)
                - Next value is the gripper command (0: open, 1: close)

        Returns:
            obs (Dict): Observation from the environment.
            reward (float): The reward obtained from this step.
            done (bool): Whether the episode has ended.
            info (Dict): Additional information.
        """
        # Extract delta translation, delta rotation (6D), and gripper command from action
        delta_translation = action[:3]
        delta_rotation_6d = action[3:9]
        gripper_command = action[9]

        # Convert 6D rotation representation back to quaternion
        delta_rotation_quat = self.rotation_6d_to_quat(delta_rotation_6d)

        # Combine delta translation and rotation quaternion
        delta_pose = np.concatenate([delta_translation, delta_rotation_quat])

        # Process gripper command
        self.move_gripper([gripper_command])

        # Update accumulated pose
        delta_pose_full = np.zeros(13)
        delta_pose_full[:7] = delta_pose
        delta_pose_full[7] = 0  # touchpad
        delta_pose_full[8] = gripper_command  # trigger (gripper)
        delta_pose_full[9] = 0  # grip
        delta_pose_full[10] = 0  # primary_button
        delta_pose_full[11:] = [0, 0]  # direction

        self.update_accumulated_pose(delta_pose_full)
        self.publish_desired_pose(self.accumulated_pose)

        # Sleep to maintain control frequency and account for speed
        time.sleep(1.0 / self.f / self.speed)

        # Increment step count
        self.step_count += 1

        # Get observation
        obs = self.get_observation()

        # Set reward (you can customize this)
        reward = 0.0

        # Check if the episode is done
        done = False
        if self.step_count >= self.max_step:
            done = True
            self.pd_controller.stop()

        # Additional information can be added to the info dictionary
        info = {}

        return obs, reward, done, info

    def get_observation(self):
        """
        Generate observation including image, depth, point cloud, state, and episode end flag.
        """
        # Process RGB image
        # if self.rgb_image is not None:
        #     obs_image = self.preprocess_image(self.rgb_image)
        # else:
        #     obs_image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Process depth image
        # if self.depth_image is not None:
        #     obs_depth = self.preprocess_depth(self.depth_image)
        # else:
        #     obs_depth = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Save point cloud raw data for visualization
        # np.save(f"point_cloud_raw_{self.step_count}.npy", self.point_cloud_raw)
        # Process point cloud
        if self.point_cloud_raw is not None:
            point_cloud = self.process_point_cloud(self.point_cloud_raw)
        else:
            # If no point cloud is available, create a placeholder
            point_cloud = np.zeros((self.num_points, 6), dtype=np.float32)

        # Get state (translation, rotation in 6D, gripper)
        if self.current_pose is not None:
            translation = self.current_pose["position"]
            rotation_quat = self.current_pose["orientation"]
            rotation_6d = self.quaternion_to_6d(rotation_quat)
            # Assuming gripper state is known or can be inferred
            gripper_state = np.array([self.get_gripper_state()], dtype=np.float32)
            state = np.concatenate([translation, rotation_6d, gripper_state])
        else:
            state = np.zeros(10, dtype=np.float32)

        # Episode ends flag
        # episode_ends = np.array([float(self.step_count >= self.max_step)], dtype=np.float32)  # Changed to array
        agent_pos = state.astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = point_cloud.astype(np.float32) # (T, 1024, 6)
        
        # Save the point cloud for visualization
        # np.save(f"point_cloud_{self.step_count}.npy", point_cloud)

        obs = {
            "agent_pos": agent_pos,
            "point_cloud": point_cloud,
        }

        return obs

    def process_point_cloud(self, points):
        """
        Processes the raw point cloud to generate a downsampled point cloud within the workspace.
        """
        num_points = self.num_points

        # Extract XYZ and RGB
        if points.shape[1] >= 6:
            # Points with RGB
            point_xyz = points[:, :3]
            point_rgb = points[:, 3:6] / 255.0  # Normalize RGB values if needed
        else:
            # Points without RGB, create dummy RGB values
            point_xyz = points[:, :3]
            point_rgb = np.zeros_like(point_xyz)

        # Apply extrinsics transformation
        ones = np.ones((point_xyz.shape[0], 1))
        points_homogeneous = np.hstack((point_xyz, ones))
        points_transformed = points_homogeneous @ self.extrinsics_matrix.T
        point_xyz_transformed = points_transformed[:, :3]

        # Combine transformed XYZ with RGB
        points_transformed = np.hstack((point_xyz_transformed, point_rgb))
        
        # Save point cloud for visualization
        # np.save(f"point_cloud_transformed_{self.step_count}.npy", points_transformed)

        # Crop point cloud to workspace
        mask = (
            (points_transformed[:, 0] > self.WORK_SPACE[0][0]) & (points_transformed[:, 0] < self.WORK_SPACE[0][1]) &
            (points_transformed[:, 1] > self.WORK_SPACE[1][0]) & (points_transformed[:, 1] < self.WORK_SPACE[1][1]) &
            (points_transformed[:, 2] > self.WORK_SPACE[2][0]) & (points_transformed[:, 2] < self.WORK_SPACE[2][1])
        )
        points_cropped = points_transformed[mask]

        # Remove dark points (assumed to be plane and background)
        # brightness_threshold = 0.1  # Adjust threshold as needed (range 0-1)
        # rgb_brightness = np.mean(points_cropped[:, 3:], axis=1)  # Calculate brightness
        # bright_points_mask = rgb_brightness > brightness_threshold
        # points_cropped = points_cropped[bright_points_mask]

        # Detect and remove the dominant plane
        points_xyz_cropped = points_cropped[:, :3]
        plane_model, inliers = self.detect_plane(points_xyz_cropped)  # RANSAC plane detection
        points_cropped = points_cropped[~inliers]  # Remove inliers (plane points)

        # Remove outliers using statistical methods
        points_cropped = self.remove_outliers(points_cropped)
        
        # np.save(f"point_cloud_cropped_{self.step_count}.npy", points_cropped)

        # If not enough points, pad with zeros
        if points_cropped.shape[0] < num_points:
            padding = np.zeros((num_points - points_cropped.shape[0], 6))
            points_cropped = np.vstack((points_cropped, padding))

        # Perform farthest point sampling to downsample to num_points
        points_xyz = points_cropped[:, :3]
        points_rgb = points_cropped[:, 3:]

        if points_xyz.shape[0] > num_points:
            sampled_points_xyz, sampled_indices = self.farthest_point_sampling(points_xyz, num_points)
            sampled_points_rgb = points_rgb[sampled_indices]
            # Combine XYZ and RGB
            sampled_points = np.hstack((sampled_points_xyz, sampled_points_rgb))
        else:
            sampled_points = points_cropped  # Already padded if needed

        point_cloud = sampled_points.astype(np.float32)
        # np.save(f"point_cloud_downsampled_{self.step_count}.npy", point_cloud)
        return point_cloud

    def detect_plane(self, points_xyz):
        """
        Detect the dominant plane using RANSAC and return inlier mask.
        """
        from sklearn.linear_model import RANSACRegressor

        # Fit a plane using RANSAC
        ransac = RANSACRegressor(residual_threshold=0.01, max_trials=100)
        xy = points_xyz[:, :2]
        z = points_xyz[:, 2]
        ransac.fit(xy, z)

        # Get inliers
        inliers = ransac.inlier_mask_
        return ransac.estimator_, inliers

    def remove_outliers(self, points, neighbors=30, std_ratio=1.0):
        """
        Remove statistical outliers from the point cloud.
        """
        from sklearn.neighbors import NearestNeighbors
        import numpy as np

        points_xyz = points[:, :3]
        nbrs = NearestNeighbors(n_neighbors=neighbors).fit(points_xyz)
        distances, _ = nbrs.kneighbors(points_xyz)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)

        # Filter outliers
        mask = mean_distances < threshold
        return points[mask]


    def farthest_point_sampling(self, points_xyz, num_points):
        """
        Perform farthest point sampling on the point cloud.

        Args:
            points_xyz (np.ndarray): The XYZ coordinates of the points.
            num_points (int): The number of points to sample.

        Returns:
            sampled_points (np.ndarray): The sampled XYZ coordinates.
            indices (np.ndarray): The indices of the sampled points.
        """
        points = torch.from_numpy(points_xyz).unsqueeze(0).float()
        if torch.cuda.is_available():
            points = points.cuda()
        # Sample farthest points
        sampled_points, indices = torch3d_ops.sample_farthest_points(points, K=num_points)
        sampled_points = sampled_points.squeeze(0)
        indices = indices.squeeze(0)
        if torch.cuda.is_available():
            sampled_points = sampled_points.cpu()
            indices = indices.cpu()
        return sampled_points.numpy(), indices.numpy()

    def preprocess_image(self, image):
        """
        Preprocesses the RGB image to the desired size and format.
        """
        img_size = self.img_size
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # HxWxC -> CxHxW
        image = torchvision.transforms.functional.resize(image, (img_size, img_size))
        image = image.permute(1, 2, 0)  # CxHxW -> HxWxC
        image = image.numpy().astype(np.uint8)
        return image

    def preprocess_depth(self, depth):
        """
        Preprocesses the depth image to the desired size and format.
        """
        img_size = self.img_size
        depth = depth.astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)  # HxW -> 1xHxW
        depth = torchvision.transforms.functional.resize(depth, (img_size, img_size))
        depth = depth.squeeze(0)
        depth = depth.numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)  # Normalize to [0,1]
        return depth

    def quaternion_to_6d(self, quat):
        """
        Converts a quaternion to 6D rotation representation.
        """
        r = R.from_quat(quat)
        rot_matrix = r.as_matrix()  # 3x3 matrix

        # Take the first two columns
        m1 = rot_matrix[:, 0]
        m2 = rot_matrix[:, 1]

        # Flatten to 6D vector
        rot_6d = np.concatenate([m1, m2], axis=-1)
        return rot_6d

    def rotation_6d_to_quat(self, rot_6d):
        """
        Converts a 6D rotation representation back to a quaternion.
        """
        # Reshape to 3x2 matrix
        rot_matrix = np.zeros((3, 3))
        rot_matrix[:, 0] = rot_6d[:3]
        rot_matrix[:, 1] = rot_6d[3:6]
        # Recover the third column with cross product
        rot_matrix[:, 2] = np.cross(rot_matrix[:, 0], rot_matrix[:, 1])
        # Orthogonalize the rotation matrix using SVD
        u, s, vh = np.linalg.svd(rot_matrix)
        rot_matrix = np.dot(u, vh)
        # Convert to quaternion
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()
        return quat

    def get_gripper_state(self):
        """
        Returns the current gripper state.
        """
        # Placeholder: Return 1.0 if gripper is closed, 0.0 if open.
        # You'll need to implement this based on your robot's gripper feedback.
        return 0.0  # Assuming gripper is open

    def reset(self):
        # Reset the environment
        self.step_count = 0

        # Start the PD controller
        if not self.pd_controller.processing:
            self.pd_controller.start()

        # Go to home pose
        self.go_homepose()

        # Get initial observation
        obs = self.get_observation()

        return obs

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            if self.rgb_image is not None:
                rgb = self.rgb_image.copy()
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                return rgb
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            super(KortexEnv, self).render(mode=mode)  # Just raise an exception

    def close(self):
        # Stop the PD controller
        if self.pd_controller.processing:
            self.pd_controller.stop()
        rospy.signal_shutdown("Environment closed")

    def go_homepose(self):
        """
        Moves the robot to the home pose specified in self.home_pose.
        """
        # Open the gripper
        self.move_gripper([0])

        # Set ee trajectory
        current_ee_pose = np.concatenate(
            (self.current_pose["position"], self.current_pose["orientation"])
        ).tolist()
        ee_trajectory = interpolate_trajectory(
            [current_ee_pose, self.home_ee_pose], 40, 0
        )

        # Set rate
        rate = rospy.Rate(self.f)

        # Move to home pose
        for pose in ee_trajectory:
            self.publish_desired_pose(pose)
            rate.sleep()

        time.sleep(3.0)  # Wait for the robot to stop moving

        print("Home pose reached.")

    def update_accumulated_pose(self, delta_pose):
        self.step_count += 1

        if self.accumulated_pose is None:
            if self.current_pose is None:
                rospy.logwarn("Current pose is not available yet.")
                return
            self.accumulated_pose = np.concatenate(
                (self.current_pose["position"], self.current_pose["orientation"])
            )

        # Extract current accumulated position and orientation
        current_position = self.accumulated_pose[:3]
        current_orientation_quat = self.accumulated_pose[3:7]
        current_rot = R.from_quat(current_orientation_quat).as_matrix()

        # Construct current transformation matrix T_e_t^b (end-effector pose in base frame)
        T_e_t_b = np.eye(4)
        T_e_t_b[:3, :3] = current_rot
        T_e_t_b[:3, 3] = current_position

        # Extract delta translation and rotation in end-effector frame
        delta_translation = delta_pose[:3]
        delta_rot_quat = delta_pose[3:7]
        delta_rot = R.from_quat(delta_rot_quat).as_matrix()

        # Construct delta transformation matrix Delta_T_e^{e_t}
        Delta_T_e = np.eye(4)
        Delta_T_e[:3, :3] = delta_rot
        Delta_T_e[:3, 3] = delta_translation

        # Compute new end-effector pose in base frame
        # T_e_{t+1}^b = T_e_t^b * Delta_T_e^{e_t}
        T_e_new_b = np.dot(T_e_t_b, Delta_T_e)

        # Extract new position and orientation from T_e_new_b
        new_position = T_e_new_b[:3, 3]
        new_rot = T_e_new_b[:3, :3]
        new_orientation_quat = R.from_matrix(new_rot).as_quat()

        # Update accumulated pose
        self.accumulated_pose[:3] = new_position
        self.accumulated_pose[3:7] = new_orientation_quat

    def publish_desired_pose(self, desired_pose):
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "base_link"

        # Set position
        pose_msg.pose.position.x = desired_pose[0]
        pose_msg.pose.position.y = desired_pose[1]
        pose_msg.pose.position.z = desired_pose[2]

        # Set orientation
        pose_msg.pose.orientation.x = desired_pose[3]
        pose_msg.pose.orientation.y = desired_pose[4]
        pose_msg.pose.orientation.z = desired_pose[5]
        pose_msg.pose.orientation.w = desired_pose[6]

        # Publish desired pose
        self.desired_pose_pub.publish(pose_msg)

    def move_gripper(self, gripper):
        """
        Sends a gripper command to open or close the gripper using the Kinova Gen3 gripper service.

        Args:
            gripper (list): Gripper command from the action.
        """
        # Determine the target position (0.0 to 1.0)
        target_position = 1.0 if gripper[-1] > 0.5 else 0.0  # Close if gripper[-1] > 0.5 else open

        # Create the service request
        request = SendGripperCommandRequest()
        request.input.mode = GripperMode.GRIPPER_POSITION

        # Create a Gripper object
        gripper_command = GripperCommand()
        gripper_command.mode = GripperMode.GRIPPER_POSITION

        # Create a Finger object and set its value
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = target_position  # Value between 0.0 (open) and 1.0 (closed)
        gripper_command.gripper.finger.append(finger)

        # Assign the gripper command to the request
        request.input = gripper_command

        try:
            # Call the service
            self.gripper_service(request)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to send gripper command: %s", e)
