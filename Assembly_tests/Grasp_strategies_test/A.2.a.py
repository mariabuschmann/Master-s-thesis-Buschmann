#!/usr/bin/env python3
"""
Main controller for peg-in-hole with sensor-based grasp evaluation (regrasp).
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
import numpy as np
import pandas as pd
from motion_node import MotionNode
from peginsertion_without_sensor import PegInsertion
import time

ROBOT_IP = "192.168.100.0"


class GripperClient:
    """Client for gripper open/close service."""

    def __init__(self, node):
        self.node = node
        self.cli = node.create_client(SetBool, 'gripper_control')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Waiting for gripper service ...")

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


class SensorClient:
    """Client for tactile sensor service."""

    def __init__(self, node):
        self.node = node
        self.cli = node.create_client(Trigger, 'get_sensor_data')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Waiting for sensor service ...")

    def get_sensor_data(self):
        req = Trigger.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result().success:
            import json
            return json.loads(future.result().message)
        return None


class MainController(Node):
    """Sequence: pick, evaluate grasp, insert peg."""

    def __init__(self):
        super().__init__('main_controller')
        self.motion = MotionNode(robot_ip=ROBOT_IP)
        self.gripper = GripperClient(self)
        self.sensor = SensorClient(self)
        self.peg_insertion = PegInsertion(
            self.motion, self.motion.rtde_c, self.motion.rtde_r, self.sensor
        )
        self.run_flow()

    def peginhole(self):
        df = pd.read_excel("/Taskboard_locations/taskboard_holes.ods", engine="odf")
        hole = df[df["name"] == "hole_1"].iloc[0]
        target = np.array([0.3659, 0.3499, 0.111]) + np.array([hole["center_x"], hole["center_y"], 0])
        self.motion.go_to_point(target + [0, 0, 0.008])
        self.motion.go_to_point(target)
        self.motion.go_to_point(target + [0, 0, -0.005])
        self.gripper.open()
        return time.time()

    def run_flow(self):
        self.gripper.open()
        pick = np.array([0.4085, 0.3778, 0.123])
        start = pick + np.array([-0.03, 0.0, 0.03])
        approach = np.array([0.3441, 0.38078, 0.15761])
        self.motion.move_joint_path_servoj_interpolated([approach, start, pick], dt=0.03)

        self.gripper.close()
        self.motion.rtde_c.servoStop()
        start_time = time.time()

        pose = self.motion.rtde_r.getActualTCPPose(); pose[2] += 0.008
        self.motion.moveL(pose, 3, 3)

        cop_info = self.sensor.get_sensor_data()
        if not cop_info:
            print("No sensor data. Aborting.")
            return

        dist_cop_x = cop_info["dist_cop_x"]
        bad_grasp = dist_cop_x > 0.00002 or dist_cop_x < -0.00014
        magazin = np.array([0.4065, 0.4306, 0.123])

        if bad_grasp:
            df = pd.read_excel(
                "~/Metrics/KET12/Excel_results/average_KET12_analysis.ods",
                engine="odf"
            )
            df["abs_diff"] = (df["Dist_COP_x"] - dist_cop_x).abs()
            correction = df.loc[df["abs_diff"].idxmin()]["position_offset_x"]

            target = magazin.copy(); target[0] += correction
            self.motion.go_to_point(target + [0, 0, 0.015], in_world=True)
            self.motion.go_to_point(target, in_world=True)
            self.gripper.open()

            self.motion.go_to_point(magazin, in_world=True)
            self.gripper.close()

            pose = self.motion.rtde_r.getActualTCPPose(); pose[2] += 0.015
            self.motion.moveL(pose, 3, 3)

        end_time = self.peginhole()
        print(f"Cycle time: {end_time - start_time:.3f} s")

        pose = self.motion.rtde_r.getActualTCPPose(); pose[2] += 0.05
        self.motion.go_to_point(np.array([0.3441, 0.38078, 0.15761]))
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    MainController()


if __name__ == '__main__':
    main()
