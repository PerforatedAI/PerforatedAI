#!/usr/bin/env python3
import math
import os
import random
import time
import numpy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty  # <--- CHANGED: Standard Empty service for reset

ROS_DISTRO = os.environ.get('ROS_DISTRO')

class RLEnvironment(Node):
    def __init__(self):
        super().__init__('rl_environment')
        
        # --- CONFIGURATION ---
        self.goal_pose_x = 0.5
        self.goal_pose_y = 0.5
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0
        self.action_size = 5
        self.max_step = 6000
        self.done = False
        self.fail = False
        self.succeed = False
        
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]
        
        # --- PUBS/SUBS ---
        qos = QoSProfile(depth=10)
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_sub_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_sub_callback, qos_profile_sensor_data)

        # --- CHANGED: USE STANDARD RESET SIMULATION ---
        # This service is built-in to Gazebo and resets everything to t=0
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        
    # =========================================================================
    #   RESET FUNCTION
    # =========================================================================
    def reset(self):
        """Resets the environment by resetting the whole simulation."""
        self.done = False
        self.fail = False
        self.succeed = False
        self.local_step = 0
        
        # 1. Stop the robot
        self.stop_robot()
        
        # 2. Call Reset Simulation
        # We assume the service is available since it's standard.
        if self.reset_client.service_is_ready():
            req = Empty.Request()
            # Call async so we don't block main thread forever
            self.reset_client.call_async(req)
        else:
            self.get_logger().warn('/reset_simulation service down! Manually move robot to start.')

        # 3. Generate a new Goal
        self.generate_new_goal()
        
        # 4. Wait for sensors to stabilize (resetting sim takes a split second)
        time.sleep(0.2) 
        for _ in range(15):
            rclpy.spin_once(self, timeout_sec=0.05)
            
        return self.calculate_state()
    # =========================================================================

    def step(self, action):
        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = 0.15
            msg.angular.z = self.angular_vel[action]
        else:
            msg = TwistStamped()
            msg.twist.linear.x = 0.15
            msg.twist.angular.z = self.angular_vel[action]
        self.cmd_vel_pub.publish(msg)

        start_time = time.time()
        while time.time() - start_time < 0.15:
            rclpy.spin_once(self, timeout_sec=0.01)

        state = self.calculate_state()
        reward = self.calculate_reward()
        done = self.done

        if done:
            self.stop_robot()

        return state, reward, done

    def generate_new_goal(self):
        # Goal shouldn't be too close to (0,0)
        self.goal_pose_x = random.uniform(-1.5, 1.5)
        self.goal_pose_y = random.uniform(-1.5, 1.5)
        
        while abs(self.goal_pose_x) < 0.3 and abs(self.goal_pose_y) < 0.3:
             self.goal_pose_x = random.uniform(-1.5, 1.5)
             self.goal_pose_y = random.uniform(-1.5, 1.5)
             
        self.get_logger().info(f'New Goal: [{self.goal_pose_x:.2f}, {self.goal_pose_y:.2f}]')

    def stop_robot(self):
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())

    def scan_sub_callback(self, scan):
        self.scan_ranges = []
        self.front_ranges = []
        
        if not scan.ranges:
            return

        num_of_lidar_rays = len(scan.ranges)

        for i in range(num_of_lidar_rays):
            angle = scan.angle_min + i * scan.angle_increment
            distance = scan.ranges[i]

            if distance == float('Inf') or numpy.isinf(distance):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)

            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                self.front_ranges.append(distance)

        if len(self.scan_ranges) > 0:
            self.min_obstacle_distance = min(self.scan_ranges)
        else:
            self.min_obstacle_distance = 10.0

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        self.goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_angle = goal_angle

    def calculate_state(self):
        state = []
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        
        if not self.front_ranges:
             current_ranges = [3.5] * 24
        else:
             current_ranges = self.front_ranges
             if len(current_ranges) > 24:
                 current_ranges = current_ranges[:24]
             while len(current_ranges) < 24:
                 current_ranges.append(3.5)

        for var in current_ranges:
            state.append(float(var))
        
        self.local_step += 1

        if self.goal_distance < 0.20:
            self.succeed = True
            self.done = True
            self.call_task_succeed()

        if self.min_obstacle_distance < 0.20: 
            self.fail = True
            self.done = True
            self.call_task_failed()

        if self.local_step >= self.max_step:
            self.fail = True
            self.done = True
            self.call_task_failed()

        return state

    def call_task_succeed(self):
        self.get_logger().info('Goal Reached! +100')

    def call_task_failed(self):
        self.get_logger().info('Collision or Timeout. -50')

    def calculate_reward(self):
        yaw_reward = 1 - (2 * abs(self.goal_angle) / math.pi)
        
        obstacle_penalty = 0.0
        if self.min_obstacle_distance < 0.5:
             obstacle_penalty = -2.0

        reward = yaw_reward + obstacle_penalty

        if self.succeed:
            reward = 100.0
        elif self.fail:
            reward = -50.0

        return reward

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw