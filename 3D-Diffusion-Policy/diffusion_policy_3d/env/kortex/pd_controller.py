#!/usr/bin/env python
import threading
import rospy
import numpy as np
from kortex_driver.msg import (
    BaseCyclic_Feedback,
    TwistCommand,
    Base_JointSpeeds,
    JointSpeed,
)
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

# Import PyKDL and URDF parser
import PyKDL
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel  # For KDL tree parsing


class KortexPDController:
    def __init__(self, robot_name="my_gen3", rate=100, home_joint_positions=None):
        # Store the robot name for topic names
        self.robot_name = robot_name

        # PD gains
        self.Kp = np.array([1.5, 1.5, 1.5])
        self.Kd = np.array([0.001, 0.001, 0.001])
        self.Kp_rot = np.array([1.5, 1.5, 1.5])
        self.Kd_rot = np.array([0.01, 0.01, 0.01])

        # State variables
        self.current_pose = None
        self.desired_pose = None
        self.current_joint_positions = None
        self.last_position_error = np.zeros(3)
        self.last_angular_error = np.zeros(3)
        self.last_time = rospy.Time.now()

        # Initialize last command time to current time
        self.last_command_time = rospy.Time.now()

        # Control variables
        self.processing = False
        self.control_thread = None

        # Publishers and Subscribers
        self.velocity_pub = rospy.Publisher(
            f"/{self.robot_name}/in/cartesian_velocity", TwistCommand, queue_size=10
        )

        self.joint_velocity_pub = rospy.Publisher(
            f"/{self.robot_name}/in/joint_velocity", Base_JointSpeeds, queue_size=10
        )

        rospy.Subscriber(
            f"/{self.robot_name}/base_feedback",
            BaseCyclic_Feedback,
            self.base_feedback_callback,
        )

        rospy.Subscriber("/desired_pose", PoseStamped, self.desired_pose_callback)

        # Initialize KDL
        self.initialize_kdl()

        # Control loop frequency
        self.rate = rospy.Rate(rate)

        # Store home joint positions
        self.home_joint_positions = home_joint_positions  # Add home joint positions

    def initialize_kdl(self):
        # Load URDF from parameter server
        robot_description_param = f"/{self.robot_name}/robot_description"
        robot = URDF.from_parameter_server(robot_description_param)

        # Build the KDL tree using kdl_parser
        success, self.kdl_tree = treeFromUrdfModel(robot)
        if not success:
            rospy.logerr("Failed to convert URDF to KDL tree")
            return

        # Define the base and end-effector links
        self.base_link = robot.get_root()
        self.end_effector_link = "end_effector_link"  # Adjust if different

        # Create the kinematic chain
        self.kdl_chain = self.kdl_tree.getChain(self.base_link, self.end_effector_link)

        # Set up solvers
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(self.kdl_chain)
        self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)

        # Get the number of joints
        self.num_joints = self.kdl_chain.getNrOfJoints()

    def base_feedback_callback(self, msg):
        # Extract current end-effector pose
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
        # Extract current joint positions
        self.current_joint_positions = []
        for actuator in msg.actuators:
            self.current_joint_positions.append(np.deg2rad(actuator.position))

    def desired_pose_callback(self, msg):
        # Extract desired end-effector pose from PoseStamped message
        self.desired_pose = {
            "position": np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            ),
            "orientation": np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            ),
        }
        # Update the last command time
        self.last_command_time = rospy.Time.now()

    def start(self):
        if not self.processing:
            self.processing = True
            # Start the control loop in a new thread
            self.control_thread = threading.Thread(target=self.control_loop)
            self.control_thread.start()
            rospy.loginfo("Kortex PD Controller started.")

    def stop(self):
        if self.processing:
            self.processing = False
            # Wait for the control loop to exit
            self.control_thread.join()
            self.control_thread = None
            # Stop the robot
            self.stop_robot()
            rospy.loginfo("Kortex PD Controller stopped.")

    def control_loop(self):
        while not rospy.is_shutdown() and self.processing:
            if (
                self.current_pose is not None
                and self.desired_pose is not None
                and self.current_joint_positions is not None
            ):
                # Check for timeout
                current_time = rospy.Time.now()
                time_since_last_command = (
                    current_time - self.last_command_time
                ).to_sec()
                if time_since_last_command > 5.0:
                    # rospy.logwarn(
                    #     "No new desired pose received for 5 seconds. Stopping the robot."
                    # )
                    self.stop_robot()
                else:
                    # Choose which control method to use:
                    # Uncomment one of the following lines to select the control method

                    # self.compute_and_send_velocity_command()  # Original PD control using twist commands
                    self.compute_and_send_joint_velocity_command()  # New control using joint velocities
            else:
                # If initial data is not yet received, ensure the robot is stopped
                self.stop_robot()
            self.rate.sleep()

    def compute_and_send_velocity_command(self):
        # Compute position error
        position_error = self.desired_pose["position"] - self.current_pose["position"]

        # Compute orientation error using rotation matrices
        R_desired = R.from_quat(self.desired_pose["orientation"])
        R_current = R.from_quat(self.current_pose["orientation"])

        # Compute rotation error in the base frame
        R_error = R_desired * R_current.inv()

        # Convert rotation error to rotation vector (axis-angle)
        angular_error = R_error.as_rotvec()

        # Adjust axis order and signs if necessary (depending on your coordinate frame)

        # Ensure the rotation angle is within [-π, π]
        angle = np.linalg.norm(angular_error)
        if angle > np.pi:
            angular_error = angular_error * ((angle - 2 * np.pi) / angle)

        # Time step
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        if dt == 0:
            dt = 1e-6  # Avoid division by zero

        # Derivative of errors
        position_error_dot = (position_error - self.last_position_error) / dt
        angular_error_dot = (angular_error - self.last_angular_error) / dt

        # PD control law
        linear_velocity = self.Kp * position_error + self.Kd * position_error_dot
        angular_velocity = self.Kp_rot * angular_error + self.Kd_rot * angular_error_dot

        # Create and publish TwistCommand
        twist_msg = TwistCommand()
        twist_msg.reference_frame = 0  # Base frame

        twist_msg.twist.linear_x = linear_velocity[0]
        twist_msg.twist.linear_y = linear_velocity[1]
        twist_msg.twist.linear_z = linear_velocity[2]

        twist_msg.twist.angular_x = angular_velocity[0]
        twist_msg.twist.angular_y = angular_velocity[1]
        twist_msg.twist.angular_z = angular_velocity[2]
        twist_msg.duration = 0  # Continuous

        # Publish the twist command
        self.velocity_pub.publish(twist_msg)

        # Update last errors and time
        self.last_position_error = position_error
        self.last_angular_error = angular_error
        self.last_time = current_time

    def compute_and_send_joint_velocity_command(self):
        # Compute position error
        position_error = self.desired_pose["position"] - self.current_pose["position"]

        if np.linalg.norm(position_error) > 0.4:
            rospy.logwarn("Position error is too high. Not moving.")
            self.stop_robot()
            return

        # Compute orientation error using rotation matrices
        R_desired = R.from_quat(self.desired_pose["orientation"])
        R_current = R.from_quat(self.current_pose["orientation"])

        # Compute rotation error in the base frame
        R_error = R_desired * R_current.inv()

        # Convert rotation error to rotation vector (axis-angle)
        angular_error = R_error.as_rotvec()

        # Ensure the rotation angle is within [-π, π]
        angle = np.linalg.norm(angular_error)
        if angle > np.pi:
            angular_error = angular_error * ((angle - 2 * np.pi) / angle)

        # Time step
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        if dt == 0:
            dt = 1e-6  # Avoid division by zero

        # Derivative of errors
        position_error_dot = (position_error - self.last_position_error) / dt
        angular_error_dot = (angular_error - self.last_angular_error) / dt

        # PD control law
        twist_error = np.zeros(6)
        twist_error[0:3] = self.Kp * position_error + self.Kd * position_error_dot
        twist_error[3:6] = self.Kp_rot * angular_error + self.Kd_rot * angular_error_dot

        # Create KDL JntArray for current joint positions
        q_current = PyKDL.JntArray(self.num_joints)
        for i in range(self.num_joints):
            q_current[i] = self.current_joint_positions[i]

        # Compute Jacobian
        jacobian = PyKDL.Jacobian(self.num_joints)
        self.jacobian_solver.JntToJac(q_current, jacobian)
        jacobian_array = np.array(
            [[jacobian[i, j] for j in range(self.num_joints)] for i in range(6)]
        )

        try:
            # Compute the pseudoinverse of the Jacobian
            jacobian_pinv = np.linalg.pinv(jacobian_array)

            # Primary task: End-effector control
            joint_velocities_task = jacobian_pinv.dot(twist_error)

            if self.home_joint_positions is not None:
                # Null space projection to drive towards home configuration
                identity = np.eye(self.num_joints)
                null_space_projector = identity - jacobian_pinv.dot(jacobian_array)

                # Gains for null space control
                K_nullspace = 0.1  # Gain for moving towards home pose
                K_minimize_joint_velocity = 0.05  # Gain to minimize joint movement

                # Current joint positions as numpy array
                q_current_array = np.array(self.current_joint_positions)
                q_home_diff = self.home_joint_positions - q_current_array

                # Null space contribution to drive towards home configuration
                joint_velocities_nullspace_home = K_nullspace * q_home_diff

                # Null space contribution to minimize joint velocities
                joint_velocities_nullspace_minimize = (
                    -K_minimize_joint_velocity * q_current_array
                )

                # Combine null space contributions
                joint_velocities_nullspace = (
                    joint_velocities_nullspace_home
                    + joint_velocities_nullspace_minimize
                )

                # Apply null space projection
                joint_velocities = joint_velocities_task + null_space_projector.dot(
                    joint_velocities_nullspace
                )
            else:
                # No home joint positions provided, use primary task only
                joint_velocities = joint_velocities_task

        except np.linalg.LinAlgError:
            rospy.logwarn("Singular matrix encountered while computing pseudoinverse.")
            return

        # Limit joint velocities to safe values
        joint_velocities = np.clip(joint_velocities, -1.0, 1.0)

        # Create a Base_JointSpeeds message
        joint_velocity_msg = Base_JointSpeeds()

        # Iterate over joint velocities and populate the message
        for i, velocity in enumerate(joint_velocities):
            joint_speed = JointSpeed()  # Create a new JointSpeed message for each joint
            joint_speed.joint_identifier = i  # Set the joint index (0-based)
            joint_speed.value = velocity  # Set the desired velocity in rad/s
            joint_speed.duration = 0  # Duration of 0 for continuous motion
            joint_velocity_msg.joint_speeds.append(joint_speed)  # Append to the list

        # Publish the joint velocity message
        self.joint_velocity_pub.publish(joint_velocity_msg)

        # Update last errors and time
        self.last_position_error = position_error
        self.last_angular_error = angular_error
        self.last_time = current_time

    def stop_robot(self):
        """
        Stop the robot by setting the desired pose to the current pose
        and sending zero velocities to all joints.
        """
        # Set the desired pose to the current pose
        if self.current_pose is not None:
            self.desired_pose = self.current_pose.copy()
        else:
            rospy.logwarn("Current pose is None. Cannot set desired pose.")

        # Send zero velocities to all joints
        joint_velocity_msg = Base_JointSpeeds()
        for i in range(self.num_joints):
            joint_speed = JointSpeed()
            joint_speed.joint_identifier = i
            joint_speed.value = 0.0  # Zero velocity
            joint_speed.duration = 0
            joint_velocity_msg.joint_speeds.append(joint_speed)

        # Publish the stop command
        self.joint_velocity_pub.publish(joint_velocity_msg)


# No main function since this class is intended to be integrated into an existing node
