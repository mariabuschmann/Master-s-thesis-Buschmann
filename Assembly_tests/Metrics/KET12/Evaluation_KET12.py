#!/usr/bin/env python3
"""
UR5e grasp attempt with IK + Gripper control in Docker
- Opens/closes Weiss gripper via rosservice inside container
- Solves IK (SLSQP) for target pose with variations
- Executes MoveJ + MoveL via RTDE
- Runs evaluation script
"""

import os, time, json, subprocess
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rtde_control, rtde_receive

ROBOT_IP = "192.168.100.0"
CONTAINER_ID = "d6d9f1bc9659"   # update if container changes


# ---------------- Gripper ----------------
def run_in_container(cmd):
    full_cmd = f'docker exec -i {CONTAINER_ID} bash -c "source /root/catkin_ws/devel/setup.bash && {cmd}"'
    res = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0: print("❌ Error:", res.stderr)
    return res.stdout.strip()

def open_gripper():  print("🟢 Open gripper"); print(run_in_container("rosservice call /gripper_srv '{position:60, force:25, relative:false}'"))
def close_gripper(): print("🔴 Close gripper"); print(run_in_container("rosservice call /gripper_srv '{position:28, force:100, relative:false}'"))


# ---------------- Kinematics ----------------
def dh_transform(th, d, a, al):
    ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
    return np.array([[ct,-st*ca, st*sa,a*ct],
                     [st, ct*ca,-ct*sa,a*st],
                     [0,     sa,    ca,    d],
                     [0,      0,     0,    1]])

def forward_kinematics(j):
    dh=[(0.1625,0,np.pi/2),(0,-0.425,0),(0,-0.3922,0),
        (0.1333,0,np.pi/2),(0.0997,0,-np.pi/2),(0.0996,0,0)]
    T=np.eye(4)
    for (d,a,al),th in zip(dh,j): T=T@dh_transform(th,d,a,al)
    return T @ dh_transform(0,0.13,0,0)

def make_target_pose(pos,rpy):
    T=np.eye(4); T[:3,3]=pos; T[:3,:3]=R.from_euler('xyz',rpy).as_matrix(); return T

def ik_objective(j,Tt):
    T=forward_kinematics(j)
    return np.linalg.norm(T[:3,3]-Tt[:3,3])+0.44*np.linalg.norm(T[:3,:3]-Tt[:3,:3])

def solve_ik(Tt,guess,bounds):
    return minimize(ik_objective,guess,args=(Tt,),method="SLSQP",bounds=bounds,
                    options={"ftol":1e-6,"maxiter":2000,"eps":5e-6})

def world_to_base(p):
    T=np.array([[1,0,0,0.8332],[0,1,0,0.6735],[0,0,1,0.04],[0,0,0,1]])
    return (np.linalg.inv(T)@np.append(p,1))[:3]

def validate(result,target,bounds):
    if not result.success: return False
    pos=forward_kinematics(result.x)[:3,3]
    err=np.linalg.norm(target-pos)
    if err>0.01: return False
    near=any(np.isclose(result.x[i],[b[0],b[1]],atol=0.01).any() for i,b in enumerate(bounds))
    return not near


# ---------------- ROS2 Joint Commander ----------------
class UR5JointCommander(Node):
    def __init__(self):
        super().__init__('ur5_commander')
        self.pub=self.create_publisher(JointTrajectory,'/joint_trajectory_controller/joint_trajectory',10)
    def send(self,angles):
        msg=JointTrajectory()
        msg.joint_names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                         'wrist_1_joint','wrist_2_joint','wrist_3_joint']
        pt=JointTrajectoryPoint(); pt.positions=angles.tolist(); pt.time_from_start.sec=5
        msg.points=[pt]; self.pub.publish(msg)
        self.get_logger().info(f"➡️ Sent joint trajectory: {np.round(angles,3)}")


# ---------------- Main ----------------
def main():
    rtde_c, rtde_r = rtde_control.RTDEControlInterface(ROBOT_IP), rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rclpy.init(); node = UR5JointCommander()

    # Target
    world_target=np.array([0.6921,0.2487,0.038])
    base_target=world_to_base(world_target)
    print("🎯 Target (World):",world_target,"→ Base:",base_target)

    bounds=[(-np.pi,np.pi)]*6
    orientation=[np.pi,0,np.pi/2]

    with open(os.path.expanduser("~/Assebly_tests/Metrics/KET12/grasp_variations.json")) as f:
        variations=json.load(f)

    for var in variations:
        print(f"\n🔄 Grasp variation: {var['name']}")
        open_gripper(); time.sleep(2)

        pos=base_target+np.array(var["position_offset"])
        rot=np.array(orientation)+np.radians(var["orientation_offset_deg"])
        Tt=make_target_pose(pos,rot)

        guesses=[[0.1,-0.5,0.2,-1.0,0.1,0],[0,-0.5,0.1,-1.1,0,0],[0.1,-0.4,0.2,-1.2,0.1,0.1]]
        result=next((r for g in guesses if (r:=solve_ik(Tt,g,bounds)).success and validate(r,pos,bounds)),None)
        if not result: print("❌ No valid IK"); continue

        joint_sol=result.x; rtde_c.moveJ(joint_sol,0.1,0.2)
        close_gripper(); time.sleep(1)

        # Lift
        pose=rtde_r.getActualTCPPose(); pose[2]+=0.05
        print("⬆️ Lifting object by 5cm"); rtde_c.moveL(pose,0.1,0.1)

        # Run evaluation
        script=os.path.expanduser("~/Assebly_tests/Metrics/KET12/evaluation_metrics_KET12.py")
        args=["python3", script,
              "--joints", ",".join(map(str,joint_sol)),
              "--var_name", var["name"],
              f"--pos_offset={','.join(map(str,var['position_offset']))}",
              f"--rot_offset={','.join(map(str,var['orientation_offset_deg']))}"]
        res=subprocess.run(args,capture_output=True,text=True)
        print("📊 Eval Output:\n",res.stdout)
        if res.returncode: print("❌ Eval Error:",res.stderr)

    print("\n✅ All variations processed")
    node.destroy_node(); rclpy.shutdown()


if __name__=="__main__": main()
