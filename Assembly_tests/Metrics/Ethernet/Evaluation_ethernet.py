#!/bin/env python3
import os, time, json, subprocess
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rtde_control, rtde_receive

ROBOT_IP = "192.168.100.0"
VAR_FILE = os.path.expanduser("~/Assembly_tests/Metrics/Ethernet/grasp_variations.json")
SCRIPT_PATH = os.path.expanduser("~/Assembly_tests/Metrics/Ethernet/evaluation_metrics_ethernet.py")

# --- Docker Gripper ---
def run_in_container(cmd, container="e380247f700b"):
    command = (
        f'docker exec -i {container} bash -c '
        f'"source /root/catkin_ws/devel/setup.bash && {cmd}"'
    )
    return subprocess.run(command, shell=True, capture_output=True, text=True)

def open_gripper():
    run_in_container("rosservice call /gripper_srv '{position: 60, force: 25, relative: false}'")

def close_gripper():
    run_in_container("rosservice call /gripper_srv '{position: 25, force: 100, relative: false}'")

# --- Kinematics ---
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,       sa,    ca,    d],
        [0,        0,     0,    1]
    ])

def forward_kinematics(joints):
    dh_params = [
        (0.1625, 0,       np.pi/2),
        (0,     -0.425,   0),
        (0,     -0.3922,  0),
        (0.1333, 0,       np.pi/2),
        (0.0997, 0,      -np.pi/2),
        (0.0996, 0,       0)
    ]
    T = np.eye(4)
    for (d, a, alpha), theta in zip(dh_params, joints):
        T = T @ dh_transform(theta, d, a, alpha)
    return T @ dh_transform(0, 0.13, 0, 0)  # TCP Offset

def world_to_base(pos_world):
    T_base_to_world = np.array([
        [1, 0, 0, 0.8332],
        [0, 1, 0, 0.6735],
        [0, 0, 1, 0.04],
        [0, 0, 0, 1]
    ])
    return np.linalg.inv(T_base_to_world) @ np.append(pos_world, 1)

# --- Inverse Kinematic ---
def ik_objective(joints, T_target):
    T = forward_kinematics(joints)
    pos_error = np.linalg.norm(T[:3, 3] - T_target[:3, 3])
    rot_error = np.linalg.norm(T[:3, :3] - T_target[:3, :3])
    return pos_error + 0.44 * rot_error

def make_target_pose(position, rpy):
    T = np.eye(4)
    T[:3, 3] = position
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    return T

def solve_ik(T_target, guess, bounds):
    return minimize(
        ik_objective, guess, args=(T_target,),
        method="SLSQP", bounds=bounds,
        options={"ftol": 1e-6, "maxiter": 2000}
    )

# --- ROS2 Commander ---
class UR5Commander(Node):
    def __init__(self):
        super().__init__("ur5_commander")
        self.pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            10
        )

    def send(self, angles):
        msg = JointTrajectory()
        msg.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        pt = JointTrajectoryPoint()
        pt.positions = angles.tolist()
        pt.time_from_start.sec = 5
        msg.points = [pt]
        self.pub.publish(msg)
        self.get_logger().info(f"🚚 Zielpose gesendet: {np.round(angles, 3)}")

# --- Main Flow ---
def main():
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    rclpy.init()
    node = UR5Commander()

    ziel_world = np.array([0.4051, 0.484, 0.140])
    pos_loch = world_to_base(ziel_world)[:3]
    orientation = [np.pi, 0, np.pi/2]
    bounds = [(-np.pi, np.pi)] * 6

    with open(VAR_FILE) as f:
        variations = json.load(f)

    for var in variations:
        print(f"\n🔄 {var['name']}")
        open_gripper()
        time.sleep(1)

        pos = pos_loch + np.array(var["position_offset"])
        rpy = np.array(orientation) + np.radians(var["orientation_offset_deg"])
        T_target = make_target_pose(pos, rpy)

        guesses = [
            [0, -0.5, 0.1, -1, 0, 0],
            [0.1, -0.4, 0.2, -1.1, 0.1, 0]
        ]

        result = None
        for guess in guesses:
            result = solve_ik(T_target, guess, bounds)
            if result.success:
                break
        if not result or not result.success:
            print("❌ Kein gültiger IK gefunden")
            continue

        joints = result.x
        rtde_c.moveJ(joints, 0.1, 0.2)
        close_gripper()
        time.sleep(1)

        # Object lift
        lifted = rtde_r.getActualTCPPose()
        lifted[2] += 0.05
        rtde_c.moveL(lifted, 0.1, 0.1)
        time.sleep(1)

        # Metrics-Script
        args = [
            f"--joints={','.join(map(str, joints))}",
            f"--var_name={var['name']}",
            f"--pos_offset={','.join(map(str, var['position_offset']))}",
            f"--rot_offset={','.join(map(str, var['orientation_offset_deg']))}"
        ]
        out = subprocess.run(
            ["python3", SCRIPT_PATH] + args,
            capture_output=True, text=True
        )
        print(out.stdout if out.returncode == 0 else out.stderr)

    print("\n*** FERTIG ***")

if __name__ == "__main__":
    main()
