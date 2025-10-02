#!/usr/bin/env python3
"""
Main controller for peg-in-hole with sensor-based grasp evaluation (direct correction).
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
import numpy as np
import pandas as pd
from motion_node import MotionNode
import time

ROBOT_IP = "192.168.100.0"
HOLES_FILE = "/Taskboard_locations/taskboard_holes.ods"
CORR_FILE = "~/Metrics/KET12/Excel_results/average_KET12_analysis.ods"


class GripperClient:
    def __init__(self, node):
        self.cli = node.create_client(SetBool, "gripper_control")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Waiting for gripper service ...")

    def command(self, close=False):
        req = SetBool.Request(); req.data = close
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        print("Gripper closed." if close else "Gripper opened.")
        return future.result().success

    def open(self): return self.command(close=False)
    def close(self): return self.command(close=True)


class SensorClient:
    def __init__(self, node):
        self.cli = node.create_client(Trigger, "get_sensor_data")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Waiting for sensor service ...")

    def get(self):
        req = Trigger.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        import json
        return json.loads(future.result().message) if future.result().success else None


class MainController(Node):
    def __init__(self):
        super().__init__("main_controller")
        self.motion = MotionNode(robot_ip=ROBOT_IP)
        self.gripper = GripperClient(self)
        self.sensor = SensorClient(self)
        self.hole = pd.read_excel(HOLES_FILE, engine="odf").query("name == 'hole_1'").iloc[0]
        self.run_flow()

    def target_hole(self, correction=0.0):
        base = np.array([0.3659, 0.3499, 0.111])
        return base + np.array([self.hole["center_x"] + correction, self.hole["center_y"], 0.0])

    def peginhole(self, correction=0.0):
        t = self.target_hole(correction)
        self.motion.go_to_point(t)
        self.motion.go_to_point(t + [0, 0, -0.005])
        self.gripper.open()
        return time.time()

    def run_flow(self):
        self.gripper.open()
        print("\n--- STARTING ---\n")

        pick = np.array([0.4065, 0.3772, 0.123])
        start = pick + [-0.03, 0.0, 0.03]
        approach = np.array([0.3441, 0.38078, 0.15761])
        self.motion.move_joint_path_servoj_interpolated([approach, start, pick], dt=0.03)

        self.gripper.close()
        self.motion.rtde_c.servoStop()
        start_time = time.time()

        pose = self.motion.rtde_r.getActualTCPPose(); pose[2] += 0.015
        self.motion.moveL(pose, 3, 3)

        data = self.sensor.get()
        if not data: return print("No sensor data. Aborting.")
        bad_grasp = data["dist_cop_x"] > 0.00002 or data["dist_cop_x"] < -0.00014
        print("Grasp:", "Bad" if bad_grasp else "Good")

        correction = 0.0
        if bad_grasp:
            df = pd.read_excel(CORR_FILE, engine="odf")
            df["abs_diff"] = (df["Dist_COP_x"] - data["dist_cop_x"]).abs()
            correction = df.loc[df["abs_diff"].idxmin()]["position_offset_x"]

        end_time = self.peginhole(correction)
        print(f"⏱️ Cycle time: {end_time - start_time:.3f} s")

        self.motion.go_to_point(np.array([0.3441, 0.38078, 0.15761]))
        print("\n--- TASK COMPLETE ---\n")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    MainController()


if __name__ == "__main__":
    main()
