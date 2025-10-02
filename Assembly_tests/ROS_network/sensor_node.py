#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from xela_server_ros2.msg import SensStream
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import json
import rtde_receive

ROBOT_IP = "192.168.100.0"

rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

class XelaSensorNode(Node):
    """
    ROS2 Node that:
      - Subscribes to Xela tactile sensor data
      - Computes CoP (Center of Pressure) and forces
      - Provides this via a Trigger service (`get_sensor_data`)
    """

    def __init__(self):
        super().__init__('xela_sensor_service')

        self.ROBOT_IP = ROBOT_IP
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
        self.joint_angles = self.rtde_r.getActualQ()

        # Subscribe to tactile data
        self.subscription = self.create_subscription(
            SensStream, '/xServTopic_force', self.listener_callback, 10
        )

        self.latest_df = None

        # Service
        self.srv = self.create_service(Trigger, 'get_sensor_data', self.handle_service)
        self.get_logger().info("Sensor service started (get_sensor_data)")

        # Transformation: robot base → world
        self.T_robot_to_world = np.array([
            [1, 0, 0, 0.8332],
            [0, 1, 0, 0.676005],
            [0, 0, 1, 0.04],
            [0, 0, 0, 1]
        ])

    # --- DH transform helper ---
    def dh_transform(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,       sa,      ca,      d],
            [0,        0,       0,      1]
        ])

    # --- World → Base coordinates ---
    def world_to_base(self, pos_world):
        T_base_to_world = self.T_robot_to_world
        T_world_to_base = np.linalg.inv(T_base_to_world)
        pos_world_hom = np.append(pos_world, 1)
        pos_base = T_world_to_base @ pos_world_hom
        return pos_base[:3]

    # --- Compute CoP and forces ---
    def get_cop_and_force_local(self, df_sensor):
        df_active = df_sensor[df_sensor["Force_Z"] > 0]
        if df_active.empty:
            return None

        forces = df_active["Force_Z"].values
        positions = df_active[["X_world", "Y_world", "Z_world"]].values
        total_force = np.sum(forces)
        cop = np.average(positions, axis=0, weights=forces)

        # Expected CoP position (centerline)
        x_opt = 0.0
        y_opt = -0.004
        z_vals = [0.0161 - i * 0.0062 for i in range(6)]
        local_points = np.array([[y_opt, x_opt, z] for z in z_vals])

        # Forward kinematics
        T_tcp_robot = np.identity(4)
        dh_params = [
            [self.joint_angles[0], 0.1625, 0, np.pi/2],
            [self.joint_angles[1], 0, -0.425, 0],
            [self.joint_angles[2], 0, -0.3922, 0],
            [self.joint_angles[3], 0.1333, 0, np.pi/2],
            [self.joint_angles[4], 0.0997, 0, -np.pi/2],
            [self.joint_angles[5], 0.0996, 0, 0],
            [0, 0.130, 0, 0]
        ]
        for theta, d, a, alpha in dh_params:
            T_tcp_robot = T_tcp_robot @ self.dh_transform(theta, d, a, alpha)

        T_tcp_world = self.T_robot_to_world @ T_tcp_robot
        world_points = (T_tcp_world[:3, :3] @ local_points.T).T + T_tcp_world[:3, 3]

        cop_x_opt = np.mean(world_points[:, 0])
        dist_cop_x = cop[0] - cop_x_opt

        # Total Force_X
        df_activex = df_sensor[df_sensor["Force_X"] > 0.0]
        total_forcex = np.sum(df_activex["Force_X"].values) if not df_activex.empty else 0.0

        return {
            "CoP_plus_world": tuple(cop),
            "cop_x_opt": cop_x_opt,
            "cop_x": cop[0],
            "dist_cop_x": dist_cop_x,
            "total_force": total_force,
            "total_forcex": total_forcex
        }

    # --- ROS2 subscriber callback ---
    def listener_callback(self, msg):
        taxel_list = []
        start_z = 0.0161
        start_x = 0.0062 / 2
        x_offsets = [start_x, -start_x]

        # Forward kinematics
        dh_params = [
            [self.joint_angles[0], 0.1625, 0, np.pi/2],
            [self.joint_angles[1], 0, -0.425, 0],
            [self.joint_angles[2], 0, -0.3922, 0],
            [self.joint_angles[3], 0.1333, 0, np.pi/2],
            [self.joint_angles[4], 0.0997, 0, -np.pi/2],
            [self.joint_angles[5], 0.0996, 0, 0],
            [0, 0.130, 0, 0]
        ]
        T_tcp_robot = np.identity(4)
        for theta, d, a, alpha in dh_params:
            T_tcp_robot = T_tcp_robot @ self.dh_transform(theta, d, a, alpha)
        T_tcp_world = self.T_robot_to_world @ T_tcp_robot

        offset = -0.004
        T_sensor_pos_robot = T_tcp_robot.copy()
        T_sensor_pos_robot[:3, 3] += offset * T_tcp_robot[:3, 1]
        T_sensor_pos_world = self.T_robot_to_world @ T_sensor_pos_robot

        for sensor_index, sensor in enumerate(msg.sensors):
            for row in range(2):
                for col in range(6):
                    idx = row * 6 + col
                    force = sensor.forces[idx]
                    offset_x = x_offsets[row]
                    offset_y = offset
                    offset_local = np.array([offset_y, offset_x, start_z - col * 0.0062])
                    point_robot = T_tcp_robot[:3, 3] + T_tcp_robot[:3, :3] @ offset_local
                    point_world = self.T_robot_to_world[:3, :3] @ point_robot + self.T_robot_to_world[:3, 3]
                    taxel_list.append({
                        'Sensor': "Sensor_y+",
                        'Taxel_Label': idx+1,
                        'Taxel': idx+1,
                        'X_world': float(point_world[0]),
                        'Y_world': float(point_world[1]),
                        'Z_world': float(point_world[2]),
                        "X_local": float(offset_local[0]),
                        "Y_local": float(offset_local[1]),
                        "Z_local": float(offset_local[2]),
                        'Force_X': float(force.x),
                        'Force_Y': float(force.y),
                        'Force_Z': float(force.z)
                    })
        self.latest_df = pd.DataFrame(taxel_list)

    # --- Service handler ---
    def handle_service(self, request, response):
        if self.latest_df is not None:
            cop_info = self.get_cop_and_force_local(self.latest_df)
            result = {
                "CoP_plus_world": cop_info["CoP_plus_world"] if cop_info else None,
                "cop_x_opt": cop_info["cop_x_opt"] if cop_info else None,
                "cop_x": cop_info["cop_x"] if cop_info else None,
                "dist_cop_x": cop_info["dist_cop_x"] if cop_info else None,
                "total_force": cop_info["total_force"] if cop_info else None,
                "total_forcex": cop_info["total_forcex"] if cop_info else None,
                "num_taxels": len(self.latest_df),
            }
            response.success = True
            response.message = json.dumps(result)
        else:
            response.success = False
            response.message = "No sensor data received yet!"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = XelaSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
