import gym
import numpy as np
import torch
import json
import os
import time
import struct

try:
    import rospy
    from kortex_driver.srv import SendGripperCommand, SendGripperCommandRequest
    from kortex_driver.msg import BaseCyclic_Feedback, GripperMode, GripperCommand, Finger
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, JointState, PointCloud2
    from cv_bridge import CvBridge
    import sensor_msgs.point_cloud2 as pc2  # For PointCloud2 processing
    # Import the PD controller and utility functions
    from diffusion_policy_3d.env.kortex.pd_controller import KortexPDController
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None

from scipy.spatial.transform import Rotation as R
import cv2

from diffusion_policy_3d.common.model_util import interpolate_trajectory

import torchvision
import pytorch3d.ops as torch3d_ops

from gym import spaces

my_path = os.path.dirname(os.path.abspath(__file__))


class KortexEnv(gym.Env):
    """
    OpenAI Gym environment for controlling the Kortex robot arm.
    Works in two modes:
        - If ROS_AVAILABLE=True: Uses ROS, kortex drivers, and robot hardware.
        - If ROS_AVAILABLE=False: No ROS calls, robot actions become no-ops (useful for training).
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

        self.task = task_name
        self.step_count = 0
        self.accumulated_pose = None  # Initialize accumulated pose
        self.f = 10  # Frequency
        self.rgb_image = None
        self.depth_image = None
        self.point_cloud_raw = None  # Raw point cloud data
        self.traj_representation = traj_representation
        self.num_points = num_points
        self.img_size = img_size

        self.current_pose = None
        self.current_velocity = None
        self.last_position_error = np.zeros(3)
        self.last_angular_error = np.zeros(3)

        # Initialize joint data
        self.joint_positions = None
        self.joint_velocities = None
        self.joint_torques = None
        self.observed_btn_states = [[0, 0, 0, 0, 0, 0]]

        if ROS_AVAILABLE:
            # Initialize ROS node
            rospy.init_node("kortex_env", anonymous=True)
            self.bridge = CvBridge()
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

            self.last_time = rospy.Time.now()
        else:
            self.bridge = None
            self.gripper_service = None
            self.desired_pose_pub = None

        # Load task parameters
        self.load_parameters()

        # Initialize PD controller only if ROS is available
        if ROS_AVAILABLE:
            self.pd_controller = KortexPDController(robot_name=robot_name, rate=100)
        else:
            self.pd_controller = None

        if ROS_AVAILABLE:
            # Wait for the initial pose to be received
            while self.current_pose is None and not rospy.is_shutdown():
                rospy.sleep(0.1)

        # Define extrinsics matrix
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
        # Action: [delta_translation(3), delta_rotation(6), gripper(1)]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(10,),
            dtype=np.float32,
        )

        # Observation: Dict of agent_pos (10,) and point_cloud (num_points, 6)
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32,
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 6),
                dtype=np.float32,
            ),
        })

        # Start the PD controller if ROS is available
        if ROS_AVAILABLE and self.pd_controller is not None:
            self.pd_controller.start()

        # Go to home pose if ROS is available
        if ROS_AVAILABLE:
            self.go_homepose()
        else:
            print("ROS not available, skipping go_homepose.")

    def load_parameters(self):
        """
        Loads task parameters from prepare_conf.json
        """
        prepare_config_path = os.path.join(my_path, "prepare_conf.json")
        prepare_conf = {}

        if os.path.exists(prepare_config_path):
            with open(prepare_config_path, "r") as f:
                try:
                    prepare_conf = json.load(f)
                except json.JSONDecodeError as e:
                    if ROS_AVAILABLE:
                        rospy.logerr(f"JSONDecodeError while parsing prepare_conf.json: {e}")
                    else:
                        print(f"JSONDecodeError while parsing prepare_conf.json: {e}")
                    prepare_conf = {}
        else:
            msg = f"Configuration file {prepare_config_path} does not exist."
            if ROS_AVAILABLE:
                rospy.logerr(msg)
            else:
                print(msg)
            prepare_conf = {}

        self.task_config = prepare_conf.get(self.task, None)
        if self.task_config is None:
            msg = f"No configuration found for task '{self.task}' in prepare_conf."
            if ROS_AVAILABLE:
                rospy.logerr(msg)
            else:
                print(msg)
            raise ValueError(msg)

        self.max_step = self.task_config["max_len"]
        self.speed = self.task_config["speed"]
        self.home_pose = self.task_config["arm_joints"]  # home pose
        self.home_ee_pose = self.task_config["ee_pose"]  # home end-effector pose

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
            ).as_quat(),
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
        if ROS_AVAILABLE:
            try:
                self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                rospy.logerr("Failed to receive RGB image: %s", e)

    def kinect_depth_callback(self, msg):
        if ROS_AVAILABLE:
            try:
                self.depth_image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="passthrough"
                )
                self.depth_image = np.nan_to_num(self.depth_image)
            except Exception as e:
                rospy.logerr("Failed to receive depth image: %s", e)

    def pointcloud_callback(self, msg):
        if ROS_AVAILABLE:
            self.point_cloud_raw = self.pointcloud2_to_array(msg)

    def pointcloud2_to_array(self, cloud_msg):
        if not ROS_AVAILABLE:
            return np.zeros((0, 6), dtype=np.float32)

        field_names = [field.name for field in cloud_msg.fields]
        has_rgb = 'rgb' in field_names

        point_list = []
        for point in pc2.read_points(cloud_msg, field_names=field_names, skip_nans=True):
            x, y, z = point[0], point[1], point[2]
            if has_rgb:
                rgb = point[field_names.index('rgb')]
                if isinstance(rgb, float):
                    s = struct.pack('>f', rgb)
                    i = struct.unpack('>l', s)[0]
                else:
                    i = int(rgb)

                r = (i >> 16) & 0x0000ff
                g = (i >> 8) & 0x0000ff
                b = (i) & 0x0000ff
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
        # Extract parts of the action
        delta_translation = action[:3]
        delta_rotation_6d = action[3:9]
        gripper_command = action[9]

        # Convert 6D rotation back to quaternion
        delta_rotation_quat = self.rotation_6d_to_quat(delta_rotation_6d)
        delta_pose = np.concatenate([delta_translation, delta_rotation_quat])

        # Move gripper if ROS available
        self.move_gripper([gripper_command])

        # Update accumulated pose
        delta_pose_full = np.zeros(13)
        delta_pose_full[:7] = delta_pose
        delta_pose_full[7] = 0  # touchpad
        delta_pose_full[8] = gripper_command  # trigger (gripper)
        delta_pose_full[9] = 0  # grip
        delta_pose_full[10] = 0  # primary_button
        delta_pose_full[11:] = [0, 0]

        self.update_accumulated_pose(delta_pose_full)

        # Publish desired pose if ROS available
        if ROS_AVAILABLE:
            self.publish_desired_pose(self.accumulated_pose)

        # Sleep to maintain control frequency
        time.sleep(1.0 / self.f / self.speed)

        # Increment step count
        self.step_count += 1

        # Get observation
        obs = self.get_observation()

        # Set reward (customize as needed)
        reward = 0.0

        # Check if episode done
        done = False
        if self.step_count >= self.max_step:
            done = True
            if ROS_AVAILABLE and self.pd_controller is not None:
                self.pd_controller.stop()

        info = {}

        return obs, reward, done, info

    def get_observation(self):
        # Process point cloud
        if self.point_cloud_raw is not None:
            point_cloud = self.process_point_cloud(self.point_cloud_raw)
        else:
            point_cloud = np.zeros((self.num_points, 6), dtype=np.float32)

        # Get state
        if self.current_pose is not None:
            translation = self.current_pose["position"]
            rotation_quat = self.current_pose["orientation"]
            rotation_6d = self.quaternion_to_6d(rotation_quat)
            gripper_state = np.array([self.get_gripper_state()], dtype=np.float32)
            state = np.concatenate([translation, rotation_6d, gripper_state])
        else:
            state = np.zeros(10, dtype=np.float32)

        agent_pos = state.astype(np.float32)
        point_cloud = point_cloud.astype(np.float32)

        obs = {
            "agent_pos": agent_pos,
            "point_cloud": point_cloud,
        }

        return obs

    def process_point_cloud(self, points):
        num_points = self.num_points

        # Extract XYZ and RGB
        if points.shape[1] >= 6:
            # Points with RGB
            point_xyz = points[:, :3]
            point_rgb = points[:, 3:6] / 255.0
        else:
            point_xyz = points[:, :3]
            point_rgb = np.zeros_like(point_xyz)

        # Apply extrinsics
        ones = np.ones((point_xyz.shape[0], 1))
        points_homogeneous = np.hstack((point_xyz, ones))
        points_transformed = points_homogeneous @ self.extrinsics_matrix.T
        point_xyz_transformed = points_transformed[:, :3]

        # Combine XYZ with RGB
        points_transformed = np.hstack((point_xyz_transformed, point_rgb))

        # Crop to workspace
        mask = (
            (points_transformed[:, 0] > self.WORK_SPACE[0][0]) & (points_transformed[:, 0] < self.WORK_SPACE[0][1]) &
            (points_transformed[:, 1] > self.WORK_SPACE[1][0]) & (points_transformed[:, 1] < self.WORK_SPACE[1][1]) &
            (points_transformed[:, 2] > self.WORK_SPACE[2][0]) & (points_transformed[:, 2] < self.WORK_SPACE[2][1])
        )
        points_cropped = points_transformed[mask]

        # Detect and remove plane
        points_xyz_cropped = points_cropped[:, :3]
        plane_model, inliers = self.detect_plane(points_xyz_cropped)
        points_cropped = points_cropped[~inliers]

        # Remove outliers
        points_cropped = self.remove_outliers(points_cropped)

        # Pad if needed
        if points_cropped.shape[0] < num_points:
            padding = np.zeros((num_points - points_cropped.shape[0], 6))
            points_cropped = np.vstack((points_cropped, padding))

        # Farthest point sampling
        points_xyz = points_cropped[:, :3]
        points_rgb = points_cropped[:, 3:]
        if points_xyz.shape[0] > num_points:
            sampled_points_xyz, sampled_indices = self.farthest_point_sampling(points_xyz, num_points)
            sampled_points_rgb = points_rgb[sampled_indices]
            sampled_points = np.hstack((sampled_points_xyz, sampled_points_rgb))
        else:
            sampled_points = points_cropped

        return sampled_points.astype(np.float32)

    def detect_plane(self, points_xyz):
        from sklearn.linear_model import RANSACRegressor

        # Fit a plane using RANSAC
        ransac = RANSACRegressor(residual_threshold=0.01, max_trials=100)
        xy = points_xyz[:, :2]
        z = points_xyz[:, 2]
        ransac.fit(xy, z)

        inliers = ransac.inlier_mask_
        return ransac.estimator_, inliers

    def remove_outliers(self, points, neighbors=30, std_ratio=1.0):
        from sklearn.neighbors import NearestNeighbors

        points_xyz = points[:, :3]
        nbrs = NearestNeighbors(n_neighbors=neighbors).fit(points_xyz)
        distances, _ = nbrs.kneighbors(points_xyz)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)

        mask = mean_distances < threshold
        return points[mask]

    def farthest_point_sampling(self, points_xyz, num_points):
        points_torch = torch.from_numpy(points_xyz).unsqueeze(0).float()
        if torch.cuda.is_available():
            points_torch = points_torch.cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points_torch, K=num_points)
        sampled_points = sampled_points.squeeze(0)
        indices = indices.squeeze(0)
        if torch.cuda.is_available():
            sampled_points = sampled_points.cpu()
            indices = indices.cpu()
        return sampled_points.numpy(), indices.numpy()

    def preprocess_image(self, image):
        img_size = self.img_size
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # HWC -> CHW
        image = torchvision.transforms.functional.resize(image, (img_size, img_size))
        image = image.permute(1, 2, 0)  # CHW -> HWC
        image = image.numpy().astype(np.uint8)
        return image

    def preprocess_depth(self, depth):
        img_size = self.img_size
        depth = depth.astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)  # HW -> 1HW
        depth = torchvision.transforms.functional.resize(depth, (img_size, img_size))
        depth = depth.squeeze(0)
        depth = depth.numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def quaternion_to_6d(self, quat):
        r = R.from_quat(quat)
        rot_matrix = r.as_matrix()
        m1 = rot_matrix[:, 0]
        m2 = rot_matrix[:, 1]
        rot_6d = np.concatenate([m1, m2], axis=-1)
        return rot_6d

    def rotation_6d_to_quat(self, rot_6d):
        rot_matrix = np.zeros((3, 3))
        rot_matrix[:, 0] = rot_6d[:3]
        rot_matrix[:, 1] = rot_6d[3:6]
        rot_matrix[:, 2] = np.cross(rot_matrix[:, 0], rot_matrix[:, 1])
        u, s, vh = np.linalg.svd(rot_matrix)
        rot_matrix = np.dot(u, vh)
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()
        return quat

    def get_gripper_state(self):
        # Placeholder: Return 1.0 if gripper closed, else 0.0.
        return 0.0

    def reset(self):
        self.step_count = 0

        # Start PD controller if available
        if ROS_AVAILABLE and self.pd_controller and not self.pd_controller.processing:
            self.pd_controller.start()

        # Go to home pose if available
        if ROS_AVAILABLE:
            self.go_homepose()
        else:
            print("ROS not available, skipping go_homepose in reset.")

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
            super(KortexEnv, self).render(mode=mode)

    def close(self):
        # Stop PD controller if running
        if ROS_AVAILABLE and self.pd_controller and self.pd_controller.processing:
            self.pd_controller.stop()
        if ROS_AVAILABLE:
            rospy.signal_shutdown("Environment closed")
        else:
            print("Environment closed (no ROS)")

    def go_homepose(self):
        # If no ROS, just return
        if not ROS_AVAILABLE:
            print("ROS not available, cannot go_homepose.")
            return

        # Open gripper
        self.move_gripper([0])

        # Create trajectory
        if self.current_pose is not None:
            current_ee_pose = np.concatenate(
                (self.current_pose["position"], self.current_pose["orientation"])
            ).tolist()
        else:
            if ROS_AVAILABLE:
                rospy.logwarn("Current pose not available, using home_ee_pose directly.")
            else:
                pass
            current_ee_pose = self.home_ee_pose  # fallback

        ee_trajectory = interpolate_trajectory(
            [current_ee_pose, self.home_ee_pose], 40, 0
        )

        rate = rospy.Rate(self.f)
        for pose in ee_trajectory:
            self.publish_desired_pose(pose)
            rate.sleep()

        time.sleep(3.0)  # Wait to settle
        print("Home pose reached.")

    def update_accumulated_pose(self, delta_pose):
        self.step_count += 1

        if self.accumulated_pose is None:
            if self.current_pose is None:
                msg = "Current pose is not available yet."
                if ROS_AVAILABLE:
                    rospy.logwarn(msg)
                else:
                    print(msg)
                return
            self.accumulated_pose = np.concatenate(
                (self.current_pose["position"], self.current_pose["orientation"])
            )

        current_position = self.accumulated_pose[:3]
        current_orientation_quat = self.accumulated_pose[3:7]
        current_rot = R.from_quat(current_orientation_quat).as_matrix()

        T_e_t_b = np.eye(4)
        T_e_t_b[:3, :3] = current_rot
        T_e_t_b[:3, 3] = current_position

        delta_translation = delta_pose[:3]
        delta_rot_quat = delta_pose[3:7]
        delta_rot = R.from_quat(delta_rot_quat).as_matrix()

        Delta_T_e = np.eye(4)
        Delta_T_e[:3, :3] = delta_rot
        Delta_T_e[:3, 3] = delta_translation

        T_e_new_b = np.dot(T_e_t_b, Delta_T_e)

        new_position = T_e_new_b[:3, 3]
        new_rot = T_e_new_b[:3, :3]
        new_orientation_quat = R.from_matrix(new_rot).as_quat()

        self.accumulated_pose[:3] = new_position
        self.accumulated_pose[3:7] = new_orientation_quat

    def publish_desired_pose(self, desired_pose):
        if not ROS_AVAILABLE:
            # No-op or print a message
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "base_link"

        pose_msg.pose.position.x = desired_pose[0]
        pose_msg.pose.position.y = desired_pose[1]
        pose_msg.pose.position.z = desired_pose[2]

        pose_msg.pose.orientation.x = desired_pose[3]
        pose_msg.pose.orientation.y = desired_pose[4]
        pose_msg.pose.orientation.z = desired_pose[5]
        pose_msg.pose.orientation.w = desired_pose[6]

        self.desired_pose_pub.publish(pose_msg)

    def move_gripper(self, gripper):
        if not ROS_AVAILABLE or self.gripper_service is None:
            # No ROS environment, just skip
            return

        target_position = 1.0 if gripper[-1] > 0.5 else 0.0

        request = SendGripperCommandRequest()
        request.input.mode = GripperMode.GRIPPER_POSITION

        gripper_command = GripperCommand()
        gripper_command.mode = GripperMode.GRIPPER_POSITION

        finger = Finger()
        finger.finger_identifier = 0
        finger.value = target_position
        gripper_command.gripper.finger.append(finger)

        request.input = gripper_command

        try:
            self.gripper_service(request)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to send gripper command: %s", e)
