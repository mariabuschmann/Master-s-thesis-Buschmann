#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
import numpy as np
import pandas as pd
import time

from motion_node import MotionNode
from peginsertion_oS import PegInsertion


ROBOT_IP = "192.168.100.0"


# ----------------- Gripper Client -----------------
class GripperClient:
    def __init__(self, node):
        self.node = node
        self.cli = node.create_client(SetBool, 'gripper_control')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("⏳ Waiting for gripper service...")

    def open(self):
        req = SetBool.Request()
        req.data = False
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        print("🤖 Gripper opened:", future.result().message)
        return future.result().success

    def close(self):
        req = SetBool.Request()
        req.data = True
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        print("🤖 Gripper closed:", future.result().message)
        return future.result().success


# ----------------- Sensor Client -----------------
class SensorClient:
    def __init__(self, node):
        self.node = node
        self.cli = node.create_client(Trigger, 'get_sensor_data')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("⏳ Waiting for sensor service...")

    def get_sensor_data(self):
        req = Trigger.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result().success:
            import json
            return json.loads(future.result().message)
        else:
            print("⚠️ No sensor data:", future.result().message)
            return None


# ----------------- Main Controller -----------------
class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')
        self.motion = MotionNode(robot_ip=ROBOT_IP)
        self.gripper = GripperClient(self)
        self.sensor = SensorClient(self)
        self.peginsertion = PegInsertion(self.motion, self.motion.rtde_c, self.motion.rtde_r)

        self.start_time = None
        self.run_flow()

    # --- Simple Peg-in-Hole routine ---
    def peginhole(self):
        df_holes = pd.read_excel(
            "/Taskboard_locations/taskboard_holes.ods",
            engine="odf"
        )
        hole_row = df_holes[df_holes["name"] == "hole_1"].iloc[0]
        center_x, center_y = hole_row["center_x"], hole_row["center_y"]

        target_hole = np.array([0.4051, 0.484, 0.140])

        print("\n=== Peg-in-Hole ===")
        above_pose = target_hole.copy()
        above_pose[2] += 0.01

        # Lead-in spiral approach
        down_pose = target_hole.copy()
        down_pose[2] += 0.0025
        self.peginsertion.leadin_spiral_oS(down_pose)

        self.gripper.open()
        print("✅ Peg inserted!\n")
        return time.time()

    # --- Main pipeline ---
    def run_flow(self):
        self.gripper.open()
        print("\n*** STARTING AUTOMATED GRASP FLOW ***\n")

        pick_pos_world = np.array([0.4051, 0.484, 0.140])
        orient = [np.pi, 0, np.pi/2]

        # Approach and grasp
        self.motion.go_to_point(pick_pos_world, orient)
        print("⏳ Motion complete, closing gripper...")
        self.gripper.close()
        self.start_time = time.time()

        # Lift
        current_pose = self.motion.rtde_r.getActualTCPPose()
        lifted_pose = current_pose.copy()
        lifted_pose[2] += 0.05
        self.motion.moveL(lifted_pose, 3, 3)

        # Sensor check
        cop_info = self.sensor.get_sensor_data()
        if not cop_info:
            print("⚠️ No CoP data, aborting.")
            return
        dist_cop_x = cop_info["dist_cop_x"]
        is_bad_grasp = dist_cop_x > 9999.0 or dist_cop_x < -9999999.0
        print("📊 Grasp quality:", "❌ Bad" if is_bad_grasp else "✅ Good")

        # Bad grasp → correction
        if is_bad_grasp:
            cop_df = pd.read_excel(
                "~/Assembly_tests/Metrics/Ethernet/Excel_results/average_ethernet_analysis.ods",
                engine="odf"
            )
            cop_df["abs_diff"] = (cop_df["Dist_COP_x"] - dist_cop_x).abs()
            closest_row = cop_df.loc[cop_df["abs_diff"].idxmin()]
            correction_x = closest_row["position_offset_x"]

            corrected_target = np.array([0.4051 + correction_x, 0.484, 0.140])
            above_pose = corrected_target.copy()
            above_pose[2] += 0.01

            down_pose = corrected_target.copy()
            down_pose[2] += 0.01

            print("➡️ Corrected placement, performing spiral insertion...")
            self.peginsertion.leadin_spiral_oS(down_pose)

            self.gripper.open()
            end_time = time.time()
        else:
            end_time = self.peginhole()

        # Measure time
        duration = end_time - self.start_time
        print(f"⏱️ Time between close/open: {duration:.3f} s")

        # Final lift
        current_pose = self.motion.rtde_r.getActualTCPPose()
        lifted_pose = current_pose.copy()
        lifted_pose[2] += 0.05
        self.motion.moveL(lifted_pose, 3, 3)

        # Final world pose
        end_pose = np.array([0.4051, 0.484, 0.170])
        self.motion.go_to_point(end_pose)

        tcp_base = np.array(self.motion.rtde_r.getActualTCPPose())
        print("Endpose (Base):", np.round(tcp_base, 5))

        T_base_to_world = np.array([
            [1, 0, 0, 0.8332],
            [0, 1, 0, 0.6735005],
            [0, 0, 1, 0.04],
            [0, 0, 0, 1]
        ])
        tcp_world = T_base_to_world @ np.append(tcp_base[:3], 1)
        print("Endpose (World):", np.round(tcp_world[:3], 5))

        print("\n*** DONE ***\n")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    MainController()


if __name__ == '__main__':
    main()
