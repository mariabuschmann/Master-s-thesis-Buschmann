#!/usr/bin/env python3
import os, time, argparse, datetime
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from xela_server_ros2.msg import SensStream
from odf.opendocument import OpenDocumentSpreadsheet, load
from odf.table import Table, TableRow, TableCell
from odf.text import P

# ---------------- Arguments ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--joints', required=True, help='Joint angles (comma-separated)')
parser.add_argument('--var_name', required=True)
parser.add_argument('--pos_offset', required=True)
parser.add_argument('--rot_offset', required=True)
args = parser.parse_args()

joint_angles = list(map(float, args.joints.split(',')))
var_name = args.var_name
pos_offset = list(map(float, args.pos_offset.split(',')))
rot_offset = list(map(float, args.rot_offset.split(',')))

# ---------------- Helper ----------------
def dh_transform(theta,d,a,alpha):
    ct,st,ca,sa=np.cos(theta),np.sin(theta),np.cos(alpha),np.sin(alpha)
    return np.array([[ct,-st*ca, st*sa,a*ct],
                     [st, ct*ca,-ct*sa,a*st],
                     [0,     sa,    ca,    d],
                     [0,      0,     0,    1]])

def pose_from_matrix(T):
    pos=T[:3,3]; rot=R.from_matrix(T[:3,:3]).as_euler('xyz')
    return np.round(np.concatenate((pos,rot)),5)

def world_to_base(pos):
    T=np.array([[1,0,0,0.8332],[0,1,0,0.6735],[0,0,1,0.04],[0,0,0,1]])
    return (np.linalg.inv(T)@np.append(pos,1))[:3]

# ---------------- Robot Config ----------------
dh_params = [
    [joint_angles[0],0.1625,0,np.pi/2],
    [joint_angles[1],0,-0.425,0],
    [joint_angles[2],0,-0.3922,0],
    [joint_angles[3],0.1333,0,np.pi/2],
    [joint_angles[4],0.0997,0,-np.pi/2],
    [joint_angles[5],0.0996,0,0],
    [0,0.130,0,0]
]
T_robot_to_world=np.array([[1,0,0,0.8332],[0,1,0,0.6760],[0,0,1,0.04],[0,0,0,1]])
T_tcp=np.eye(4)
for th,d,a,al in dh_params: T_tcp=T_tcp@dh_transform(th,d,a,al)
pose_tcp_world=pose_from_matrix(T_robot_to_world@T_tcp)

offset=-0.004
T_sensor=T_tcp.copy(); T_sensor[:3,3]+=offset*T_tcp[:3,1]

# ---------------- ROS2 Node ----------------
class TaxelReader(Node):
    def __init__(self):
        super().__init__('taxel_reader')
        self.taxel_data=None
        self.create_subscription(SensStream,'/xServTopic_force',self.cb,10)
    def cb(self,msg):
        data=[]
        start_z=0.0161; start_x=0.0062/2; x_offsets=[start_x,-start_x]
        for sensor in msg.sensors:
            for row in range(2):
                for col in range(6):
                    idx=row*6+col
                    f=sensor.forces[idx]
                    offset_local=np.array([offset, x_offsets[row], start_z-col*0.0062])
                    pt_robot=T_tcp[:3,3]+T_tcp[:3,:3]@offset_local
                    pt_world=T_robot_to_world[:3,:3]@pt_robot+T_robot_to_world[:3,3]
                    data.append({
                        "Taxel_Label":idx+1,
                        "X_world":pt_world[0],"Y_world":pt_world[1],"Z_world":pt_world[2],
                        "Force_Z":f.z
                    })
        self.taxel_data=pd.DataFrame(data)

# ---------------- Analysis ----------------
def get_cop_and_force(df):
    df_act=df[df["Force_Z"]>0.1]
    if df_act.empty: return dict(CoP=(np.nan,)*3,cop_x_opt=np.nan,cop_x=np.nan,dist=np.nan,total=0)
    forces,pts=df_act["Force_Z"].values,df_act[["X_world","Y_world","Z_world"]].values
    cop=np.average(pts,axis=0,weights=forces)
    t5,t7=df[df.Taxel_Label==5]["X_world"].values[0],df[df.Taxel_Label==7]["X_world"].values[0]
    cop_x_opt=(t5+t7)/2; return dict(CoP=tuple(cop),cop_x_opt=cop_x_opt,
                                    cop_x=cop[0],dist=cop[0]-cop_x_opt,total=np.sum(forces))

def compute_metrics(df):
    df_act=df[df["Force_Z"]>0.1]
    return dict(num=len(df_act),f_max=df["Force_Z"].max(),f_min=df["Force_Z"].min(),
                dist=np.ptp(df["Force_Z"]),mean=df["Force_Z"].mean(),std=df["Force_Z"].std())

# ---------------- Main ----------------
def main():
    rclpy.init(); node=TaxelReader()
    while rclpy.ok() and node.taxel_data is None: rclpy.spin_once(node,timeout_sec=1)
    df=node.taxel_data; node.destroy_node(); rclpy.shutdown()

    df_plus=df.copy(); df_plus[["X_robot","Y_robot","Z_robot"]]=df_plus.apply(
        lambda r: pd.Series(world_to_base([r.X_world,r.Y_world,r.Z_world])),axis=1)

    cop,metrics=get_cop_and_force(df_plus),compute_metrics(df_plus)

    print("\n🎯 Ergebnisse")
    print("CoP:",cop["CoP"],"Δx:",cop["dist"],"F_total:",cop["total"])
    print("Metriken:",metrics)

    # --- save ---
    out_dir=os.path.expanduser("~/Assembly_tests/Metrics/Gear/Excel_results")
    os.makedirs(out_dir,exist_ok=True)
    ods_file=os.path.join(out_dir,"01_gear_analysis.ods")

    headers=["variation","pos_x","pos_y","pos_z","rot_x","rot_y","rot_z",
             "timestamp","dist_COP_x","opt_COPx","CoP X","CoP Y","CoP Z",
             "num_taxels","F_max","F_min","Dist","F_mean","Std"]
    new_row=[var_name,*pos_offset,*rot_offset,datetime.datetime.now().isoformat(),
             cop["dist"],cop["cop_x_opt"],*cop["CoP"],metrics["num"],metrics["f_max"],
             metrics["f_min"],metrics["dist"],metrics["mean"],metrics["std"]]

    if os.path.exists(ods_file): doc=load(ods_file); table=doc.spreadsheet.childNodes[0]
    else:
        doc=OpenDocumentSpreadsheet(); table=Table(name="Grasp"); doc.spreadsheet.addElement(table)
        tr=TableRow(); [tr.addElement(TableCell().addElement(P(text=h)) or tr.addElement(cell)) for h in headers for cell in [TableCell()]]; table.addElement(tr)

    tr=TableRow()
    for v in new_row:
        c=TableCell(valuetype="float",value=str(v)) if isinstance(v,(int,float)) else TableCell(); c.addElement(P(text=str(v))); tr.addElement(c)
    table.addElement(tr); doc.save(ods_file)
    print("💾 Gespeichert in:",ods_file)

if __name__=="__main__": main()
