#!/usr/bin/env python3
"""
UR5e + Weiss Gripper Control (ROS2 + Docker):
- Opens/closes gripper inside container
- Solves IK for grasp variations
- Executes grasp, lift, and runs evaluation script
"""

import os, time, json, subprocess, datetime
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rtde_control, rtde_receive

ROBOT_IP = "192.168.100.0"

# ---------------- Gripper in Docker ----------------
def run_gripper_cmd(cmd, container="65d8a6d90187"):
    bash = f"""
    docker exec -i {container} bash -c "
        source /root/catkin_ws/devel/setup.bash && \
        rosservice call /gripper_srv '{cmd}'
    "
    """
    res = subprocess.run(bash, shell=True, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0: print("⚠️ Gripper error:", res.stderr)

def open_gripper():  run_gripper_cmd("{position: 60, force: 25, relative: false}")
def close_gripper(): run_gripper_cmd("{position: 28, force: 100, relative: false}")

# ---------------- Kinematics ----------------
def dh_transform(theta,d,a,alpha):
    ct,st,ca,sa=np.cos(theta),np.sin(theta),np.cos(alpha),np.sin(alpha)
    return np.array([[ct,-st*ca,st*sa,a*ct],
                     [st, ct*ca,-ct*sa,a*st],
                     [0,     sa,   ca,   d],
                     [0,      0,    0,   1]])

def forward_kinematics(joints):
    dh=[(0.1625,0,np.pi/2),(0,-0.425,0),(0,-0.3922,0),
        (0.1333,0,np.pi/2),(0.0997,0,-np.pi/2),(0.0996,0,0)]
    T=np.eye(4)
    for i,(d,a,al) in enumerate(dh):
        T=T@dh_transform(joints[i],d,a,al)
    return T@dh_transform(0,0.13,0,0)   # TCP offset

def make_target(pos,rpy):
    T=np.eye(4); T[:3,3]=pos; T[:3,:3]=R.from_euler("xyz",rpy).as_matrix(); return T

def ik_objective(joints,target):
    T=forward_kinematics(joints)
    return np.linalg.norm(T[:3,3]-target[:3,3])+0.4*np.linalg.norm(T[:3,:3]-target[:3,:3])

def solve_ik(target,guess,bounds):
    return minimize(ik_objective,guess,args=(target,),method="SLSQP",bounds=bounds,
                    options={"ftol":1e-6,"maxiter":2000})

def validate(result,target_pos,bounds):
    if not result.success: return False
    joints=result.x; T=forward_kinematics(joints)
    pos_err=np.linalg.norm(T[:3,3]-target_pos)
    near_limit=any(np.isclose(joints[i],b[0],atol=0.01) or np.isclose(joints[i],b[1],atol=0.01)
                   for i,b in enumerate(bounds))
    if pos_err>0.01 or near_limit: return False
    print("✅ IK solution:", np.round(joints,4)); return True

# ---------------- ROS2 Node ----------------
class UR5Commander(Node):
    def __init__(self): 
        super().__init__("ur5_commander")
        self.pub=self.create_publisher(JointTrajectory,"/joint_trajectory_controller/joint_trajectory",10)
    def moveJ(self,joints):
        msg=JointTrajectory(); msg.joint_names=[f"{n}_joint" for n in 
            ["shoulder_pan","shoulder_lift","elbow","wrist_1","wrist_2","wrist_3"]]
        p=JointTrajectoryPoint(); p.positions=joints.tolist(); p.time_from_start.sec=5
        msg.points=[p]; self.pub.publish(msg)
        self.get_logger().info(f"Sent joints: {np.round(joints,3)}")

# ---------------- Main ----------------
def main():
    rtde_c=rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r=rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rclpy.init(); node=UR5Commander()

    world_target=np.array([0.6921,0.2487,0.04])
    T_base_to_world=np.array([[1,0,0,0.8332],[0,1,0,0.6735],[0,0,1,0.04],[0,0,0,1]])
    target_base=(np.linalg.inv(T_base_to_world)@np.append(world_target,1))[:3]
    print("Target (base):",np.round(target_base,4))

    with open(os.path.expanduser("~/Assembly_tests/Metrics/Rotation/grasp_variations.json")) as f:
        variations=json.load(f)

    orientation=[np.pi,0,np.pi/2]
    bounds=[(-np.pi,np.pi)]*6

    for var in variations:
        print(f"\n🔄 Variation: {var['name']}")
        open_gripper(); time.sleep(2)
        pos=target_base+np.array(var["position_offset"])
        rpy=orientation+np.radians(var["orientation_offset_deg"])
        T_target=make_target(pos,rpy)

        for guess in [[0.1,-0.5,0.2,-1.0,0.1,0],[0,-0.5,0.1,-1.1,0,0]]:
            res=solve_ik(T_target,guess,bounds)
            if validate(res,pos,bounds):
                jt=res.x; rtde_c.moveJ(jt,0.1,0.2)
                close_gripper(); time.sleep(1)
                lifted=rtde_r.getActualTCPPose(); lifted[2]+=0.05
                rtde_c.moveL(lifted,0.1,0.1); time.sleep(1)

                # Run evaluation
                script="~/Assembly_tests/Metrics/Rotation/evaluation_rotation.py"
                subprocess.run(["python3",os.path.expanduser(script),
                    "--joints",",".join(map(str,jt)),
                    "--var_name",var["name"],
                    f"--pos_offset={','.join(map(str,var['position_offset']))}",
                    f"--rot_offset={','.join(map(str,var['orientation_offset_deg']))}"])
                break
        else: print("❌ No valid IK found.")

    print("\n*** DONE ***")
    rclpy.shutdown()

if __name__=="__main__": main()
