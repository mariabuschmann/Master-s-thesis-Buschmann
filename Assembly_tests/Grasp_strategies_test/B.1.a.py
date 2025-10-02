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
from peginsertion_without_sensor import PegInsertion
import time

ROBOT_IP = "192.168.100.0"
HOLES_FILE = "/Taskboard_locations/taskboard_holes.ods"


class GripperClient:
    """Client for gripper control."""

    def __init__(self, node):
        self.cli = node.create_client(SetBool, "gripper_control")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Waiting for gripper service...")

    def _cmd(self, close=False):
        req = SetBool.Request(); req.data = close
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        print("Gripper closed." if close else "Gripper opened.")
        return future.result().success

    def open(self): return self._cmd(False)
    def close(self): return self._cmd(True)


class MainController(Node):
    """Main pick-and-place with spiral peg insertion."""

    def __init__(self):
        super().__init__("main_controller")
        self.motion = MotionNode(robot_ip=ROBOT_IP)
        self.gripper = GripperClient(self)
        self.peg_insertion = PegInsertion(self.motion, self.motion.rtde_c, self.motion.rtde_r)
        self.hole = pd.read_excel(HOLES_FILE, engine="odf").query("name == 'hole_1'").iloc[0]
        self.run_flow()

    def peginhole(self):
        """Perform spiral peg insertion."""
        base = np.array([0.3659, 0.3504, 0.110])
        target = base + np.array([self.hole["center_x"], self.hole["center_y"], 0.0])

        print("\n=== Starting insertion ===")
        down = target.copy(); down[2] += 0.0025
        self.peg_insertion.leadin_spiral_without_sensor(down)

        self.gripper.open()
        print("✅ Peg inserted.")
        return time.time()

    def run_flow(self):
        """Pick object, insert into hole, retreat."""
        self.gripper.open()
        print("\n--- STARTING TASK ---\n")

        # Pick trajectory
        pick = np.array([0.4065, 0.3772, 0.124])
        start = pick + [-0.03, 0, 0.03]
        approach = np.array([0.3441, 0.38078, 0.15761])
        self.motion.move_joint_path_servoj_interpolated([approach, start, pick], dt=0.03)

        self.gripper.close()
        self.motion.rtde_c.servoStop()
        start_time = time.time()

        # Lift
        pose = self.motion.rtde_r.getActualTCPPose(); pose[2] += 0.015
        self.motion.moveL(pose, 3, 3)

        # Peg-in-hole
        end_time = self.peginhole()
        print(f"⏱️ Cycle time: {end_time - start_time:.3f} s")

        # Retreat
        pose = self.motion.rtde_r.getActualTCPPose(); pose[2] += 0.05
        self.motion.moveL(pose, 3, 3)
        self.motion.go_to_point(np.array([0.3441, 0.38078, 0.15761]))

        print("\n--- DONE ---\n")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    MainController()


if __name__ == "__main__":
    main()
