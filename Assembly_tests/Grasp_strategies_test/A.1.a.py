#!/usr/bin/env python3
"""
Main controller for automated peg-in-hole task using UR robot and ROS2.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import numpy as np
import pandas as pd
from motion_node import MotionNode
import time

ROBOT_IP = "192.168.100.0"


class GripperClient:
    """ROS2 service client for gripper control."""

    def __init__(self, node):
        self.node = node
        self.cli = node.create_client(SetBool, 'gripper_control')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Waiting for gripper service ...")

    def open(self):
        """Send request to open gripper."""
        req = SetBool.Request(); req.data = False
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        print("Gripper opened:", future.result().message)
        return future.result().success

    def close(self):
        """Send request to close gripper."""
        req = SetBool.Request(); req.data = True
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        print("Gripper closed:", future.result().message)
        return future.result().success


class MainController(Node):
    """Main sequence: pick object, insert peg, release."""

    def __init__(self):
        super().__init__('main_controller')
        self.motion = MotionNode(robot_ip=ROBOT_IP)
        self.gripper = GripperClient(self)
        self.run_flow()

    def peginhole(self):
        """Move peg into predefined hole and release."""
        df_holes = pd.read_excel(
            "/Taskboard_locations/taskboard_holes.ods", engine="odf"
        )
        hole_row = df_holes[df_holes["name"] == "hole_1"].iloc[0]
        base_target = np.array([0.3659, 0.3494, 0.111])
        target = base_target + np.array([hole_row["center_x"], hole_row["center_y"], 0.0])

        above = target.copy(); above[2] += 0.008
        self.motion.go_to_point(target)

        down = target.copy(); down[2] -= 0.005
        self.motion.go_to_point(down)

        self.gripper.open()
        print("Peg successfully inserted.")
        return time.time()

    def run_flow(self):
        """Full task execution: pick, lift, insert, return."""
        self.gripper.open()

        # Define pick position and approach waypoints
        pick = np.array([0.4065, 0.3772, 0.123])
        start = pick + np.array([-0.03, 0.0, 0.03])
        approach = np.array([0.3441, 0.38078, 0.15761])

        self.motion.move_joint_path_servoj_interpolated([approach, start, pick], dt=0.03)

        # Pick object
        self.gripper.close()
        self.motion.rtde_c.servoStop()
        start_time = time.time()

        # Lift
        pose = self.motion.rtde_r.getActualTCPPose()
        pose[2] += 0.015
        self.motion.moveL(pose, 3, 3)

        # Insert peg
        end_time = self.peginhole()
        print(f"Cycle time: {end_time - start_time:.3f} s")

        # Retreat and return to final pose
        pose = self.motion.rtde_r.getActualTCPPose()
        pose[2] += 0.05
        self.motion.moveL(pose, 3, 3)
        self.motion.go_to_point(np.array([0.3441, 0.38078, 0.15761]))

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    MainController()


if __name__ == '__main__':
    main()
