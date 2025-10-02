#!/usr/bin/env python3
import os, time, datetime, argparse
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from xela_server_ros2.msg import SensStream
from odf.opendocument import OpenDocumentSpreadsheet, load
from odf.table import Table, TableRow, TableCell
from odf.text import P

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--joints', required=True)
parser.add_argument('--var_name', required=True)
parser.add_argument('--pos_offset', required=True)
parser.add_argument('--rot_offset', required=True)
args = parser.parse_args()

joint_angles = list(map(float, args.joints.split(',')))
var_name = args.var_name
pos_offset = list(map(float, args.pos_offset.split(',')))
rot_offset = list(map(float, args.rot_offset.split(',')))

# --- Transformations ---
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def pose_from_matrix(T):
    pos = T[:3, 3]
    rot = R.from_matrix(T[:3, :3]).as_euler('xyz')
    return np.round(np.concatenate((pos, rot)), 5)

def world_to_base(pos_world):
    T = np.array([
        [1,0,0,0.8332],
        [0,1,0,0.6735005],
        [0,0,1,0.04],
        [0,0,0,1]
    ])
    return (np.linalg.inv(T) @ np.append(pos_world,1))[:3]

# --- ROS2 Taxel-Reader ---
class TaxelReader(Node):
    def __init__(self):
        super().__init__('taxel_reader')
        self.subscription = self.create_subscription(SensStream, '/xServTopic_force', self.cb, 10)
        self.data = None

    def cb(self, msg):
        taxels, offset = [], -0.004
        start_z, dx = 0.0161, 0.0062
        x_offsets = [dx/2, -dx/2]

        for sensor in msg.sensors:
            for row in range(2):
                for col in range(6):
                    idx = row*6+col
                    force = sensor.forces[idx]
                    offset_local = np.array([offset, x_offsets[row], start_z-col*dx])
                    point_robot = T_tcp_robot[:3,3] + T_tcp_robot[:3,:3] @ offset_local
                    point_world = T_robot_to_world[:3,:3] @ point_robot + T_robot_to_world[:3,3]
                    taxels.append({
                        'Taxel_Label': idx+1,
                        'X_world': point_world[0],
                        'Y_world': point_world[1],
                        'Z_world': point_world[2],
                        'Force_Z': force.z
                    })
        self.data = pd.DataFrame(taxels)

# --- Grasp-Metrics ---
RIGHT, LEFT = list(range(1,6)), list(range(7,12))
def compute_metrics(df):
    df_a = df[df.Force_Z>0.1]
    return {
        "num": len(df_a),
        "f_max": df.Force_Z.max(),
        "f_min": df.Force_Z.min(),
        "mean": df.Force_Z.mean(),
        "mean_r": df[df.Taxel_Label.isin(RIGHT)].Force_Z.mean(),
        "mean_l": df[df.Taxel_Label.isin(LEFT)].Force_Z.mean(),
        "std": df.Force_Z.std()
    }

def get_cop(df):
    df_a = df[df.Force_Z>0.1]
    if df_a.empty: return {"cop": (np.nan,np.nan,np.nan), "total": 0}
    forces, pos = df_a.Force_Z.values, df_a[["X_world","Y_world","Z_world"]].values
    return {"cop": tuple(np.average(pos, axis=0, weights=forces)), "total": np.sum(forces)}

# --- Roboter Pose ---
dh_params = [
    [joint_angles[0],0.1625,0,np.pi/2],
    [joint_angles[1],0,-0.425,0],
    [joint_angles[2],0,-0.3922,0],
    [joint_angles[3],0.1333,0,np.pi/2],
    [joint_angles[4],0.0997,0,-np.pi/2],
    [joint_angles[5],0.0996,0,0],
    [0,0.130,0,0]
]
T_tcp_robot = np.eye(4)
for th,d,a,al in dh_params:
    T_tcp_robot = T_tcp_robot @ dh_transform(th,d,a,al)
T_robot_to_world = np.array([
    [1,0,0,0.8332],[0,1,0,0.676005],
    [0,0,1,0.04],[0,0,0,1]
])

# --- Hauptlogik ---
def main():
    rclpy.init()
    node = TaxelReader()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1.0)
        if node.data is not None: break
        time.sleep(0.5)
    df = node.data.copy()
    node.destroy_node(); rclpy.shutdown()

    metrics, cop = compute_metrics(df), get_cop(df)
    print("\n💡 Griffmetriken:", metrics)
    print("CoP:", cop)

    # --- Export ODS ---
    out_dir = os.path.expanduser("~/Assembly_tests/Metrics/Ethernet/Excel_results")
    os.makedirs(out_dir, exist_ok=True)
    ods_file = os.path.join(out_dir, "01_ethernet_analysis.ods")

    new_row = [
        var_name, *pos_offset, *rot_offset,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        cop["cop"][0], cop["cop"][1], cop["cop"][2],
        metrics["num"], metrics["f_max"], metrics["f_min"],
        metrics["mean"], metrics["std"], metrics["mean_r"], metrics["mean_l"]
    ] + df.Force_Z.round(7).tolist()

    headers = [
        "variation_name","pos_x","pos_y","pos_z","rot_x","rot_y","rot_z","Timestamp",
        "CoP_X","CoP_Y","CoP_Z","num","F_max","F_min","F_mean","F_std","Mean_R","Mean_L"
    ] + [f"Taxel{i}" for i in df.Taxel_Label]

    if os.path.exists(ods_file):
        doc = load(ods_file)
        tables = [t for t in doc.spreadsheet.childNodes if t.tagName=="table:table"]
        table = tables[0] if tables else Table(name="Grasp")
    else:
        doc, table = OpenDocumentSpreadsheet(), Table(name="Grasp")
        doc.spreadsheet.addElement(table)
        head = TableRow()
        for h in headers:
            c=TableCell(); c.addElement(P(text=h)); head.addElement(c)
        table.addElement(head)

    row = TableRow()
    for v in new_row:
        c=TableCell(valuetype="float",value=str(v)) if isinstance(v,(int,float)) else TableCell()
        if not isinstance(v,(int,float)): c.addElement(P(text=str(v)))
        row.addElement(c)
    table.addElement(row)
    doc.save(ods_file)
    print("→ Results saved in:", ods_file)

if __name__=="__main__":
    main()
