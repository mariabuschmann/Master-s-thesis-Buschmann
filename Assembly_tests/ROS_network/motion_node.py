#!/usr/bin/env python3
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
import rtde_control, rtde_receive
import socket, time
from scipy.interpolate import CubicSpline

class MotionNode(Node):
    def __init__(self, robot_ip="192.168.100.0"):
        super().__init__('motion_node')
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.orientation_base = [np.pi, 0, np.pi/2]
        self.bounds = [(-np.pi, np.pi)] * 6
        self.guesses = [
            [0.1, -0.5, 0.2, -1.0, 0.1, 0],
            [0, -0.5, 0.1, -1.1, 0, 0],
            [0.1, -0.4, 0.2, -1.2, 0.1, 0.1],
            [0.2, -0.5, 0.1, -1.0, 0.2, 0.2],
            [0.1, -0.4, 0.0, -1.1, 0.0, 0.0]
        ]

    # ---------- Kinematics ----------
    def dh_transform(self, theta, d, a, alpha):
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, q):
        dh_params = [
            (0.1625, 0, np.pi/2), (0, -0.425, 0), (0, -0.3922, 0),
            (0.1333, 0, np.pi/2), (0.0997, 0, -np.pi/2), (0.0996, 0, 0)
        ]
        T = np.eye(4)
        for i, (d, a, alpha) in enumerate(dh_params):
            T = T @ self.dh_transform(q[i], d, a, alpha)
        return T

    def forward_kinematics_with_tcp(self, q):
        return self.forward_kinematics(q) @ self.dh_transform(0, 0.13, 0, 0)

    def pose_from_matrix(self, T):
        pos, rot = T[:3, 3], R.from_matrix(T[:3, :3]).as_euler('xyz')
        return np.round(np.concatenate((pos, rot)), 5)

    def world_to_base(self, pos_world):
        T = np.array([[1,0,0,0.8332],[0,1,0,0.6735],[0,0,1,0.04],[0,0,0,1]])
        return (np.linalg.inv(T) @ np.append(pos_world, 1))[:3]

    def make_target_pose(self, pos, rpy):
        T = np.eye(4); T[:3, 3] = pos; T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
        return T

    # ---------- Inverse Kinematics ----------
    def ik_objective(self, q, T_target):
        T = self.forward_kinematics_with_tcp(q)
        pos_err = np.linalg.norm(T[:3, 3] - T_target[:3, 3])
        rot_err = np.linalg.norm(T[:3, :3] - T_target[:3, :3])
        return pos_err + 0.44 * rot_err

    def solve_ik(self, T_target, guess):
        return minimize(self.ik_objective, guess, args=(T_target,),
                        method='SLSQP', bounds=self.bounds,
                        options={'ftol':1e-6,'maxiter':2000,'eps':5e-6})

    def validate_solution(self, result, target_pos):
        if not result.success: return False
        q, T = result.x, self.forward_kinematics_with_tcp(result.x)
        pos_err = np.linalg.norm(target_pos - T[:3, 3])
        near_limit = any(np.isclose(q[i], b[0], atol=0.01) or np.isclose(q[i], b[1], atol=0.01)
                         for i, b in enumerate(self.bounds))
        return pos_err <= 0.01 and not near_limit

    # ---------- Motion Commands ----------
    def moveJ(self, q, v=3.14, a=3.14):
        print("moveJ:", np.round(q, 3))
        self.rtde_c.moveJ(q, v, a)

    def moveL(self, pose, v=3, a=3):
        print("moveL:", np.round(pose, 4))
        self.rtde_c.moveL(pose, v, a)

    def go_to_point(self, pos, orientation=None, in_world=True):
        base_pos = self.world_to_base(pos) if in_world else pos
        rpy = orientation if orientation is not None else self.orientation_base
        T_target = self.make_target_pose(base_pos, rpy)
        for guess in self.guesses:
            result = self.solve_ik(T_target, guess)
            if self.validate_solution(result, base_pos):
                self.moveJ(result.x)
                print("✅ Target reached")
                return
        print("❌ No valid IK solution found")

    # ---------- URScript Sender ----------
    def send_urscript(self, script, ip="192.168.100.0", port=30002):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip, port))
            s.send(script.encode('utf-8'))

    def move_through_points_movep(self, poses, a=0.05, v=0.02, r=0.03):
        def to_pose(p): return f"p[{','.join(f'{x:.5f}' for x in p)}]"
        script = "def traj():\n"
        for i, p in enumerate(poses):
            rad = r if i < len(poses)-1 else 0
            script += f"  movep({to_pose(p)}, a={a}, v={v}, r={rad})\n"
        script += "end\ntraj()\n"
        self.send_urscript(script)

    def move_joint_path_servoj(self, poses_world, orientation=None,
                               steps_per_seg=5, dt=0.5, v=0.05, a=0.05,
                               lookahead=0.1, gain=300):
        joint_path = []
        for pos in poses_world:
            base_pos = self.world_to_base(pos)
            rpy = orientation if orientation else self.orientation_base
            T_target = self.make_target_pose(base_pos, rpy)
            for guess in self.guesses:
                result = self.solve_ik(T_target, guess)
                if result.success and self.validate_solution(result, base_pos):
                    joint_path.append(result.x); break
            else:
                print("❌ IK failed for", pos); return

        spline = CubicSpline(np.linspace(0,1,len(joint_path)), joint_path, axis=0)
        for q in spline(np.linspace(0,1,steps_per_seg*(len(joint_path)-1))):
            self.rtde_c.servoJ(q.tolist(), v, a, dt, lookahead, gain)
            time.sleep(dt)

def main():
    rclpy.init()
    MotionNode("192.168.100.0")  # just init node

if __name__ == "__main__":
    main()
