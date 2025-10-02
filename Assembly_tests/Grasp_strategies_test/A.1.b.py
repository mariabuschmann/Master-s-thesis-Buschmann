#!/usr/bin/env python3
"""
Main controller for automated peg-in-hole task using UR robot and ROS2 with blind search.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import numpy as np
import pandas as pd
from motion_node import MotionNode
from peginsertion_without_sensor import PegInsertion
import time

ROBOT_IP = "192.168.100.0"


class GripperClient:
    """ROS2 service client for gripper control."""

    def __init__(self, node):
        self.node = node
        self.cli = node.create_client(SetBool, 'gripper_control')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            pass

    def open(self):
        req = SetBool.Request(); req.data = False
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        return future.result().success

    def close(self):
        req = SetBool.Request(); req.data = True
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        return future.result().success


class MainController(Node):
    """Main sequence: pick, insert peg with spiral, release."""

    def __init__(self):
        super().__init__('main_controller')
        self.motion = MotionNode(robot_ip=ROBOT_IP)
        self.gripper = GripperClient(self)
        self.peginsertion = PegInsertion(
            self.motion, self.motion.rtde_c, self.motion.rtde_r
        )
        self.run_flow()

    def peginhole(self):
        df = pd.read_excel("/Taskboard_locations/taskboard_holes.ods", engine="odf")
        hole = df[df["name"] == "hole_1"].iloc[0]
        target = np.array([0.3659, 0.3504, 0.110]) + np.array([hole["center_x"], hole["center_y"], 0])
        down_pose = target.copy(); down_pose[2] += 0.0025
        self.peginsertion.leadin_spiral_without_sensor(down_pose)
        self.gripper.open()
        return time.time()

    def run_flow(self):
        self.gripper.open()
        pick = np.array([0.4065, 0.3772, 0.124])
        start = pick + np.array([-0.03, 0.0, 0.03])
        approach = np.array([0.3441, 0.38078, 0.15761])
        self.motion.move_joint_path_servoj_interpolated([approach, start, pick], dt=0.03)

        self.gripper.close()
        self.motion.rtde_c.servoStop()
        start_time = time.time()

        pose = self.motion.rtde_r.getActualTCPPose()
        pose[2] += 0.015
        self.motion.moveL(pose, 3, 3)

        end_time = self.peginhole()
        print(f"Cycle time: {end_time - start_time:.3f} s")

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
