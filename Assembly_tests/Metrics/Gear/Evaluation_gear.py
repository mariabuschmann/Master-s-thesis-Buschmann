#!/bin/env python3
import os, time, json, subprocess, datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rtde_control, rtde_receive
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROBOT_IP = "192.168.100.0"
CONTAINER_ID = "7f869ad2a0bc"

# ---------------- Gripper ----------------
def run_in_container(cmd):
    full = f'docker exec -i {CONTAINER_ID} bash -c "source /root/catkin_ws/devel/setup.bash && {cmd}"'
    res = subprocess.run(full, shell=True, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0: print("❌ Fehler:", res.stderr)

def open_gripper():  run_in_container("rosservice call /gripper_srv '{position: 80, force: 25, relative: false}'")
def close_gripper(): run_in_container("rosservice call /gripper_srv '{position: 48, force: 100, relative: false}'")

# ---------------- Kinematics ----------------
def dh_transform(theta,d,a,alpha):
    ct,st,ca,sa = np.cos(theta),np.sin(theta),np.cos(alpha),np.sin(alpha)
    return np.array([[ct,-st*ca, st*sa,a*ct],
                     [st, ct*ca,-ct*sa,a*st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1]])

def forward_kinematics_with_tcp(joints):
    dh = [(0.1625,0,np.pi/2), (0,-0.425,0), (0,-0.3922,0),
          (0.1333,0,np.pi/2), (0.0997,0,-np.pi/2), (0.0996,0,0)]
    T=np.eye(4)
    for th,(d,a,al) in zip(joints,dh): T=T@dh_transform(th,d,a,al)
    return T @ dh_transform(0,0.13,0,0)

def world_to_base(pos):
    T = np.array([[1,0,0,0.8332],[0,1,0,0.6735005],[0,0,1,0.04],[0,0,0,1]])
    return (np.linalg.inv(T) @ np.append(pos,1))[:3]

def make_target_pose(p, rpy):
    T=np.eye(4); T[:3,3]=p; T[:3,:3]=R.from_euler("xyz",rpy).as_matrix(); return T

def ik_objective(joints,Tt):
    T=forward_kinematics_with_tcp(joints)
    return np.linalg.norm(T[:3,3]-Tt[:3,3])+0.44*np.linalg.norm(T[:3,:3]-Tt[:3,:3])

def solve_ik(Tt,guess,bounds):
    return minimize(ik_objective,guess,args=(Tt,),method="SLSQP",bounds=bounds,
                    options={'ftol':1e-6,'maxiter':2000,'eps':5e-6})

def validate(res,target,bounds):
    if not res.success: return False
    joints,respos=res.x,forward_kinematics_with_tcp(res.x)[:3,3]
    err=np.linalg.norm(respos-target); near=any(np.isclose(joints[i],b[j],atol=0.01)
             for i,b in enumerate(bounds))
    print("🔧 Lösung:",np.round(joints,3),"Fehler:",round(err*1000,2),"mm")
    return err<0.01 and not near

# ---------------- ROS Commander ----------------
class UR5JointCommander(Node):
    def __init__(self): super().__init__('ur5_cmd')
    def send(self,angles):
        msg=JointTrajectory()
        msg.joint_names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                         'wrist_1_joint','wrist_2_joint','wrist_3_joint']
        p=JointTrajectoryPoint(); p.positions=angles.tolist(); p.time_from_start.sec=5
        msg.points=[p]
        self.create_publisher(JointTrajectory,'/joint_trajectory_controller/joint_trajectory',10).publish(msg)

# ---------------- Main ----------------
def main():
    rtde_c, rtde_r = rtde_control.RTDEControlInterface(ROBOT_IP), rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rclpy.init(); node=UR5JointCommander()

    world_target=np.array([0.19281,0.543,0.1155])
    pos_hole=world_to_base(world_target)
    print("🎯 Ziel Welt:",world_target,"→ Base:",pos_hole)

    with open(os.path.expanduser("~/Assembly_tests/Metrics/Gear/grasp_variations.json")) as f: variations=json.load(f)
    orientation=[np.pi,0,np.pi/2]; bounds=[(-np.pi,np.pi)]*6

    for var in variations:
        print("\n🔄 Greife:",var['name']); open_gripper(); time.sleep(2)
        pos,rot=pos_hole+np.array(var["position_offset"]),orientation+np.radians(var["orientation_offset_deg"])
        Tt=make_target_pose(pos,rot)

        for guess in [[0.1,-0.5,0.2,-1.0,0.1,0],[0,-0.5,0.1,-1.1,0,0],[0.1,-0.4,0.2,-1.2,0.1,0.1]]:
            res=solve_ik(Tt,guess,bounds)
            if validate(res,pos,bounds): break
        else: print("❌ Kein IK gefunden"); continue

        rtde_c.moveJ(res.x,0.1,0.2); close_gripper(); time.sleep(1)
        cur=rtde_r.getActualTCPPose(); cur[2]+=0.05; rtde_c.moveL(cur,0.1,0.1); time.sleep(1)

        args=[f"--joints={','.join(f'{x:.6f}' for x in res.x)}",
              f"--var_name={var['name']}",
              f"--pos_offset={','.join(map(str,var['position_offset']))}",
              f"--rot_offset={','.join(map(str,var['orientation_offset_deg']))}"]
        script=os.path.expanduser("~/Assembly_tests/Metrics/Gear/evaluation_metrics_gear.py")
        out=subprocess.run(["python3",script,*args],capture_output=True,text=True)
        print("▶️ Output:",out.stdout or out.stderr)

    print("\n✅ FERTIG!")
    rclpy.shutdown()

if __name__=="__main__": main()
