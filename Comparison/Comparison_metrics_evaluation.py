#!/usr/bin/env python3
import os, time, subprocess, argparse, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from xela_server_ros2.msg import SensStream
import rtde_control, rtde_receive

# --- CONFIG ---
ROBOT_IP      = "192.168.100.0"
CONTAINER_ID  = "c8d8320f9529"  # adjust
LOOKUP_ODS    = os.path.expanduser("~/Assembly_tests/Metrics/KET12/Excel_results/average_KET12_analysis.ods")
LOG_XLSX      = os.path.expanduser("~/Assembly_tests/Metrics/KET12/Excel_results/experiment_log.xlsx")
WORLD_TARGET  = np.array([0.4907, 0.3073, 0.038])
ORIENTATION   = [np.pi, 0, np.pi/2]
BOUNDS        = [(-np.pi, np.pi)]*6

# --- Gripper (Docker) ---
def _dock(cmd): return subprocess.run(
    f'docker exec -i {CONTAINER_ID} bash -c "source /root/catkin_ws/devel/setup.bash && {cmd}"',
    shell=True, capture_output=True, text=True)
def open_gripper():  _dock("rosservice call /gripper_srv '{position: 60, force: 25, relative: false}'")
def close_gripper(): _dock("rosservice call /gripper_srv '{position: 28, force: 100, relative: false}'")

# --- Kinematics ---
def dh(th,d,a,al):
    ct,st,ca,sa=np.cos(th),np.sin(th),np.cos(al),np.sin(al)
    return np.array([[ct,-st*ca, st*sa,a*ct],[st, ct*ca,-ct*sa,a*st],[0,sa,ca,d],[0,0,0,1]])
def fk(j):
    DH=[(0.1625,0,np.pi/2),(0,-0.425,0),(0,-0.3922,0),(0.1333,0,np.pi/2),(0.0997,0,-np.pi/2),(0.0996,0,0)]
    T=np.eye(4)
    for (d,a,al),th in zip(DH,j): T=T@dh(th,d,a,al)
    return T
def fk_tcp(j): return fk(j) @ dh(0,0.13,0,0)
def make_T(p,rpy): T=np.eye(4); T[:3,3]=p; T[:3,:3]=R.from_euler("xyz",rpy).as_matrix(); return T
def ik_obj(j,Tt): T=fk_tcp(j); return np.linalg.norm(T[:3,3]-Tt[:3,3])+0.44*np.linalg.norm(T[:3,:3]-Tt[:3,:3])
def solve_ik(Tt, guesses):
    for g in guesses:
        r=minimize(ik_obj,g,args=(Tt,),method="SLSQP",bounds=BOUNDS,options={"maxiter":2000,"ftol":1e-6})
        if r.success:
            T=fk_tcp(r.x); err=np.linalg.norm(T[:3,3]-Tt[:3,3])
            near=any(np.isclose(r.x[i],b,atol=0.01) for i in range(6) for b in BOUNDS[i])
            if err<=0.01 and not near: return r.x
    return None
def world_to_base(pw):
    T=np.array([[1,0,0,0.8332],[0,1,0,0.6735005],[0,0,1,0.04],[0,0,0,1]])
    return (np.linalg.inv(T)@np.append(pw,1))[:3]

# --- ROS2 Taxel Reader ---
class TaxelReader(Node):
    def __init__(self,T_tcp_robot,T_robot_to_world):
        super().__init__("taxel_reader")
        self.Tt=T_tcp_robot; self.Tw=T_robot_to_world; self.df=None
        self.sub=self.create_subscription(SensStream,"/xServTopic_force",self.cb,10)
    def cb(self,msg):
        off_y=-0.004; z0=0.0161; step=0.0062; xoffs=[step/2,-step/2]; rows=[]
        for s in msg.sensors:
            for r in range(2):
                for c in range(6):
                    idx=r*6+c; f=s.forces[idx]
                    pl=np.array([off_y,xoffs[r],z0-c*step])
                    pr=self.Tt[:3,3]+self.Tt[:3,:3]@pl
                    pw=self.Tw[:3,:3]@pr+self.Tw[:3,3]
                    rows.append({"Taxel_Label":idx+1,"X_world":pw[0],"Y_world":pw[1],"Z_world":pw[2],"Force_Z":f.z})
        self.df=pd.DataFrame(rows)

def cop(df):
    a=df[df.Force_Z>0.1]
    if a.empty: return None,0.0
    F=a.Force_Z.values; P=a[["X_world","Y_world","Z_world"]].values
    return tuple(np.average(P,axis=0,weights=F)), float(np.sum(F))
def optimal_cop_x(T_tcp_world):
    x0,y0=0.0,-0.004; zs=[0.0161-i*0.0062 for i in range(6)]
    local=np.array([[y0,x0,z] for z in zs]); world=(T_tcp_world[:3,:3]@local.T).T+T_tcp_world[:3,3]
    return float(np.mean(world[:,0]))

# --- MAIN ---
def main():
    _ = argparse.ArgumentParser().parse_args()

    rtde_c=rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r=rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    try: deviation_mm=float(input("Deviation (mm): ").strip())
    except: print("Invalid input."); return

    pos_base=world_to_base(WORLD_TARGET); Tt=make_T(pos_base,ORIENTATION)
    joints=solve_ik(Tt, [[0.1,-0.5,0.2,-1.0,0.1,0],[0,-0.5,0.1,-1.1,0,0],[0.2,-0.4,0.1,-1.0,0.1,0.1]])
    if joints is None: print("IK failed."); return

    open_gripper(); rtde_c.moveJ(joints,0.1,0.2); close_gripper()
    pose=rtde_r.getActualTCPPose(); pose[2]+=0.05; rtde_c.moveL(pose,0.2,0.1)

    T_tcp_robot=fk_tcp(joints)
    T_robot_to_world=np.array([[1,0,0,0.8332],[0,1,0,0.676005],[0,0,1,0.04],[0,0,0,1]])
    T_tcp_world=T_robot_to_world@T_tcp_robot

    rclpy.init(); node=TaxelReader(T_tcp_robot,T_robot_to_world)
    t0=time.time()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.4)
        if node.df is not None and len(node.df)>=12: break
        if time.time()-t0>5: break
    rclpy.shutdown()
    if node.df is None or node.df.empty: print("No taxel data."); return
    df=node.df

    cop_w,_=cop(df)
    if cop_w is None: print("No CoP."); return
    dist_cop_x = float(cop_w[0]) - optimal_cop_x(T_tcp_world)

    right=list(range(1,6)); left=list(range(7,12))
    mean_right=float(df[df.Taxel_Label.isin(right)].Force_Z.mean())
    mean_left =float(df[df.Taxel_Label.isin(left)].Force_Z.mean())
    f_diff=abs(mean_right-mean_left)

    # Lookup ODS
    cop_df=pd.read_excel(LOOKUP_ODS, engine="odf")
    cop_df["abs_diff"]=(cop_df["Dist_COP_x"]-dist_cop_x).abs()
    cr=cop_df.loc[cop_df["abs_diff"].idxmin()]
    correction_x=float(cr["position_offset_x"])

    dfm=cop_df.copy()
    dfm["d_diff"]=(dfm["Diff_Means"]-f_diff).abs()
    dfm["d_r"]=(dfm["Mean_1-5"]-mean_right).abs()
    dfm["d_l"]=(dfm["Mean_7-11"]-mean_left).abs()
    filt=dfm[(dfm["d_r"]<0.15)&(dfm["d_l"]<0.15)]
    if not filt.empty:
        crm=filt.sort_values("d_diff").iloc[0]
        base_off=float(crm["position_offset_x"])
        correction_mean = -abs(base_off) if (mean_right-mean_left)>0 else abs(base_off)
        em15, em711, edm = crm["Mean_1-5"], crm["Mean_7-11"], crm["Diff_Means"]
    else:
        correction_mean=0.0; em15=em711=edm=np.nan

    correct_cop  = abs(deviation_mm/1000.0 - correction_x  ) == 0.00
    correct_mean = abs(deviation_mm/1000.0 - correction_mean) == 0.00

    data = {
        "UserDeviation_mm": deviation_mm,
        "dist_cop_x": dist_cop_x,
        "Excel_COP_Dist_COP_x": cr["Dist_COP_x"],
        "Correction_COP": correction_x,
        "COP_Correct": correct_cop,
        "F_mean_right": mean_right,
        "F_mean_left": mean_left,
        "Diff_Means": f_diff,
        "Excel_Mean_1-5": em15,
        "Excel_Mean_7-11": em711,
        "Excel_Diff_Means": edm,
        "Correction_Diff": correction_mean,
        "Diff_Correct": correct_mean,
    }

    os.makedirs(os.path.dirname(LOG_XLSX), exist_ok=True)
    out=pd.concat([pd.read_excel(LOG_XLSX), pd.DataFrame([data])], ignore_index=True) if os.path.exists(LOG_XLSX) else pd.DataFrame([data])
    out.to_excel(LOG_XLSX, index=False)
    print(f"Saved → {LOG_XLSX}")

if __name__=="__main__":
    main()
