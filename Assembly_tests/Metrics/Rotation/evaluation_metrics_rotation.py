#!/usr/bin/env python3
"""
ROS2 Taxel Evaluation:
- Reads Xela sensor data
- Computes CoP and grasp metrics
- Exports results to ODS
"""

import os, time, datetime, argparse
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from xela_server_ros2.msg import SensStream
from odf.opendocument import OpenDocumentSpreadsheet, load
from odf.table import Table, TableRow, TableCell
from odf.text import P

# ---------------- Arguments ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--joints", required=True, help="Joint angles (CSV)")
parser.add_argument("--var_name", required=True)
parser.add_argument("--pos_offset", required=True)
parser.add_argument("--rot_offset", required=True)
args = parser.parse_args()

joint_angles = list(map(float, args.joints.split(",")))
var_name = args.var_name
pos_offset = list(map(float, args.pos_offset.split(",")))
rot_offset = list(map(float, args.rot_offset.split(",")))

# ---------------- Transformations ----------------
def dh_transform(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca, st*sa, a*ct],
                     [st,  ct*ca,-ct*sa, a*st],
                     [0,      sa,    ca,    d],
                     [0,       0,     0,    1]])

def world_to_base(p):
    T = np.array([[1,0,0,0.8332],[0,1,0,0.6735],[0,0,1,0.04],[0,0,0,1]])
    return (np.linalg.inv(T) @ np.append(p,1))[:3]

# ---------------- ROS2 Subscriber ----------------
class TaxelReader(Node):
    def __init__(self):
        super().__init__("taxel_reader")
        self.subscription = self.create_subscription(SensStream,"/xServTopic_force",self.cb,10)
        self.taxel_data = None

    def cb(self, msg):
        taxels=[]; start_z=0.0161; x_offsets=[0.0062/2,-0.0062/2]
        for sensor in msg.sensors:
            for row in range(2):
                for col in range(6):
                    idx=row*6+col; force=sensor.forces[idx]
                    offset=np.array([-0.004,x_offsets[row],start_z-col*0.0062])
                    point_robot=T_tcp_robot[:3,3]+T_tcp_robot[:3,:3]@offset
                    point_world=T_robot_to_world[:3,:3]@point_robot+T_robot_to_world[:3,3]
                    taxels.append({"Taxel_Label":idx+1,"Sensor":"Sensor_y+",
                                   "X_world":point_world[0],"Y_world":point_world[1],
                                   "Z_world":point_world[2],"Force_Z":force.z})
        self.taxel_data=pd.DataFrame(taxels)

# ---------------- Metrics ----------------
RIGHT=list(range(1,6)); LEFT=list(range(7,12)); ALL=RIGHT+LEFT

def get_cop(df):
    active=df[df.Force_Z>0.1]
    if active.empty: return None
    forces=active.Force_Z.values; pos=active[["X_world","Y_world","Z_world"]].values
    cop=np.average(pos,axis=0,weights=forces)
    t5=df[df.Taxel_Label==5].X_world.values[0]; t7=df[df.Taxel_Label==7].X_world.values[0]
    return {"CoP":tuple(cop),"cop_x":cop[0],"cop_x_opt":(t5+t7)/2,
            "dist_cop_x":cop[0]-(t5+t7)/2,"total_force":np.sum(forces)}

def compute_metrics(df):
    df=df[df.Taxel_Label.isin(ALL)]
    return {
        "num":len(df[df.Force_Z>0.1]),
        "f_max":df.Force_Z.max(),"f_min":df.Force_Z.min(),
        "dist":abs(df.Force_Z.max()-df.Force_Z.min()),
        "mean":df.Force_Z.mean(),
        "mean_right":df[df.Taxel_Label.isin(RIGHT)].Force_Z.mean(),
        "mean_left":df[df.Taxel_Label.isin(LEFT)].Force_Z.mean(),
        "std":df.Force_Z.std()
    }

# ---------------- Robot Setup ----------------
dh_params=[[joint_angles[0],0.1625,0,np.pi/2],[joint_angles[1],0,-0.425,0],
           [joint_angles[2],0,-0.3922,0],[joint_angles[3],0.1333,0,np.pi/2],
           [joint_angles[4],0.0997,0,-np.pi/2],[joint_angles[5],0.0996,0,0],[0,0.13,0,0]]
T_tcp_robot=np.eye(4)
for th,d,a,al in dh_params: T_tcp_robot=T_tcp_robot@dh_transform(th,d,a,al)
T_robot_to_world=np.array([[1,0,0,0.8332],[0,1,0,0.676],[0,0,1,0.04],[0,0,0,1]])

# ---------------- Main ----------------
def main():
    rclpy.init(); node=TaxelReader()
    while rclpy.ok():
        rclpy.spin_once(node,timeout_sec=1.0)
        if node.taxel_data is not None: break
        time.sleep(0.5)
    df=node.taxel_data.copy(); rclpy.shutdown()

    df_plus=df[df.Sensor=="Sensor_y+"]
    df_plus[["X_robot","Y_robot","Z_robot"]]=df_plus.apply(
        lambda r: pd.Series(world_to_base([r.X_world,r.Y_world,r.Z_world])),axis=1)

    cop=get_cop(df_plus); metrics=compute_metrics(df_plus)

    # ---- Export ODS ----
    outdir=os.path.expanduser("~/Assembly_tests/Metrics/Rotation/Excel_results")
    os.makedirs(outdir,exist_ok=True)
    ods_file=os.path.join(outdir,"01_Rotation_analysis.ods")

    headers=["var_name","pos_x","pos_y","pos_z","rot_x","rot_y","rot_z",
             "Timestamp","Dist_COP_x","Opt_COPx","CoP_X","CoP_Y","CoP_Z",
             "Active","F_max","F_min","ΔF","Mean","Std","Mean_right","Mean_left"]
    new_row=[var_name,*pos_offset,*rot_offset,
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             round(cop["dist_cop_x"],6),round(cop["cop_x_opt"],6),
             *map(lambda x:round(x,6),cop["CoP"]),
             metrics["num"],metrics["f_max"],metrics["f_min"],metrics["dist"],
             metrics["mean"],metrics["std"],metrics["mean_right"],metrics["mean_left"]]

    if os.path.exists(ods_file):
        doc=load(ods_file); table=doc.spreadsheet.getElementsByType(Table)[0]
    else:
        doc=OpenDocumentSpreadsheet(); table=Table(name="Grasp-metrics")
        doc.spreadsheet.addElement(table)
        row=TableRow(); [row.addElement(TableCell().addElement(P(text=h))) for h in headers]; table.addElement(row)

    row=TableRow()
    for v in new_row:
        cell=TableCell(valuetype="float",value=str(v)) if isinstance(v,(int,float)) else TableCell()
        if not isinstance(v,(int,float)): cell.addElement(P(text=str(v)))
        row.addElement(cell)
    table.addElement(row)

    doc.save(ods_file); print(f"✅ Results saved in {ods_file}")

if __name__=="__main__": main()
