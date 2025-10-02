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
        req = SetBool.Request(); req.data = False
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        print("🤖 Gripper opened:", future.result().message)
        return future.result().success

    def close(self):
        req = SetBool.Request(); req.data = True
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

    # --- Standard Peg-in-Hole ---
    def peginhole(self):
        hole_data = pd.read_excel(
            "/home/maria/xela/quality/test/Peg&Hole_Algorithmus/taskboard_holes2.ods",
            engine="odf"
        )
        target_hole = np.array([0.1931, 0.543, 0.135])  # predefined hole

        print("\n=== Executing Peg-in-Hole ===")
        above_pose = target_hole.copy(); above_pose[2] += 0.01
        down_pose = target_hole.copy();  down_pose[2] += 0.0025

        self.peginsertion.leadin_spiral_oS(down_pose)
        self.gripper.open()

        print("✅ Peg successfully inserted!\n")
        return time.time()

    # --- Main process flow ---
    def run_flow(self):
        self.gripper.open()
        print("\n*** STARTING AUTO GRASP FLOW ***\n")

        pick_pos = np.array([0.4065, 0.375, 0.128])
        orient = [np.pi, 0, np.pi/2]

        # Approach path
        waypoints = [
            np.array([0.3441, 0.38078, 0.15761]),
            pick_pos + np.array([-0.03, 0, 0.03]),
            pick_pos
        ]
        self.motion.move_joint_path_servoj_interpolated(waypoints, dt=0.03)
        print("⏳ Path executed, closing gripper...")
        self.gripper.close()
        self.motion.rtde_c.servoStop()

        self.start_time = time.time()

        # Lift
        lifted = self.motion.rtde_r.getActualTCPPose()
        lifted[2] += 0.015
        self.motion.moveL(lifted, 3, 3)

        # Sensor check
        cop_info = self.sensor.get_sensor_data()
        if not cop_info:
            print("⚠️ No sensor data, aborting.")
            return
        dist_cop_x = cop_info["dist_cop_x"]
        bad_grasp = dist_cop_x > 0.00002 or dist_cop_x < -0.00014
        print("📊 Grasp:", "❌ Bad" if bad_grasp else "✅ Good")

        # Bad grasp correction
        if bad_grasp:
            cop_df = pd.read_excel(
                "~/Assembly_tests/Metrics/Gear/Excel_results/average_gear_analysis.ods",
                engine="odf"
            )
            cop_df["abs_diff"] = (cop_df["Dist_COP_x"] - dist_cop_x).abs()
            correction = cop_df.loc[cop_df["abs_diff"].idxmin()]["position_offset_x"]

            target = np.array([0.1931 + correction, 0.543, 0.135])
            down_pose = target.copy(); down_pose[2] += 0.0025

            print("➡️ Correcting placement...")
            self.peginsertion.leadin_spiral_oS(down_pose)

            orient[2] += np.radians(30)
            final_pose = down_pose.copy(); final_pose[2] -= 0.012
            self.motion.go_to_point(final_pose, orient)

            self.gripper.open()
            end_time = time.time()
        else:
            end_time = self.peginhole()

        # Duration
        duration = end_time - self.start_time
        print(f"⏱️ Time between close and open: {duration:.3f}s")

        # Lift again
        lifted = self.motion.rtde_r.getActualTCPPose()
        lifted[2] += 0.05
        self.motion.moveL(lifted, 3, 3)

        # Final pose
        end_pose = np.array([0.3441, 0.38078, 0.15761])
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
