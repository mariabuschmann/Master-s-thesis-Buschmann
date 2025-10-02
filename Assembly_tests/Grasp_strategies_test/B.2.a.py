#!/usr/bin/env python3
"""
Main controller for peg-in-hole with sensor-based grasp evaluation with spiral insertion with force-control
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
import numpy as np
import pandas as pd
from motion_node import MotionNode
from peginsertion_with_sensor import PegInsertion
import time

ROBOT_IP = "192.168.100.0"
HOLES_FILE = "/Taskboard_locations/taskboard_holes.ods"
COP_FILE = "~/Metrics/KET12/Excel_results/average_KET12_analysis.ods"


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


class SensorClient:
    """Client for tactile sensor service."""

    def __init__(self, node):
        self.cli = node.create_client(Trigger, "get_sensor_data")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Waiting for sensor service...")

    def get_sensor_data(self):
        req = Trigger.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        if future.result().success:
            import json; return json.loads(future.result().message)
        print("⚠️ No sensor data received.")
        return None


class MainController(Node):
    """Main pick-place-insert cycle with grasp evaluation and correction."""

    def __init__(self):
        super().__init__("main_controller")
        self.motion = MotionNode(robot_ip=ROBOT_IP)
        self.gripper = GripperClient(self)
        self.sensor = SensorClient(self)
        self.peg_insertion = PegInsertion(self.motion, self.motion.rtde_c, self.motion.rtde_r, self.sensor)
        self.hole = pd.read_excel(HOLES_FILE, engine="odf").query("name == 'hole_1'").iloc[0]
        self.run_flow()

    def peginhole(self, target):
        """Perform peg-in-hole insertion at given target."""
        above = target.copy(); above[2] += 0.008
        self.motion.go_to_point(above)
        down = target.copy(); down[2] += 0.0025

        print("\n=== Insertion ===")
        self.peg_insertion.leadin_spiral(down)
        self.gripper.open()
        print("✅ Peg inserted.")
        return time.time()

    def run_flow(self):
        self.gripper.open()
        print("\n--- STARTING TASK ---\n")

        # Pick trajectory
        pick = np.array([0.4065, 0.375, 0.124])
        start = pick + [-0.03, 0, 0.03]
        approach = np.array([0.3441, 0.38078, 0.15761])
        self.motion.move_joint_path_servoj_interpolated([approach, start, pick], dt=0.03)

        self.gripper.close()
        self.motion.rtde_c.servoStop()
        start_time = time.time()

        # Lift
        pose = self.motion.rtde_r.getActualTCPPose(); pose[2] += 0.015
        self.motion.moveL(pose, 2, 2)

        # Sensor evaluation
        cop_info = self.sensor.get_sensor_data()
        if not cop_info:
            print("Aborting: no sensor data.")
            return

        dist_cop_x = cop_info["dist_cop_x"]
        bad_grasp = dist_cop_x > 0.00002 or dist_cop_x < -0.00014
        print("Grasp quality:", "❌ Bad" if bad_grasp else "✅ Good")

        # Target hole
        base = np.array([0.3689, 0.348, 0.106])
        target = base + np.array([self.hole["center_x"], self.hole["center_y"], 0.0])

        if bad_grasp:
            cop_df = pd.read_excel(COP_FILE, engine="odf")
            cop_df["abs_diff"] = (cop_df["Dist_COP_x"] - dist_cop_x).abs()
            correction = cop_df.loc[cop_df["abs_diff"].idxmin()]["position_offset_x"]
            target[0] += correction
            print(f"Applying correction offset: {correction}")

        # Insert
        end_time = self.peginhole(target)

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
