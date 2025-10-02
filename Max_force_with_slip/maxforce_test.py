#!/usr/bin/env python3
"""
UR5e: Controlled downward motion with force/slip monitoring
- Move to fixed start pose
- Move down until force/safety thresholds
- Lift back up
- Plot force vs time with max annotation
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import rtde_control, rtde_receive

ROBOT_IP = "192.168.100.0"

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# --- Parameters ---
start_pose = [-0.2316, -0.5484, 0.0552, -3.1221, -0.0167, 0.0046]
force_limit = 17.0         # max allowed force (N)
slip_drop = -2.0           # sudden slip threshold (N diff)
speed, acc = 0.01, 0.01
max_lift = 0.02
dt = 0.1                   # step time

# --- Data logging ---
forces, times, positions = [], [], []
baseline_force = None
t0 = time.time()
current_pose = start_pose.copy()


def move_to_start():
    """Go to start pose and measure baseline force."""
    global baseline_force, current_pose
    print("Moving to start pose...")
    if rtde_c.moveL(start_pose, speed, acc):
        current_pose = start_pose.copy()
        base_samples = [rtde_r.getActualTCPForce()[2] for _ in range(10)]
        baseline_force = np.mean(base_samples)
        print(f"Start pose reached, baseline: {baseline_force:.2f} N")
    else:
        raise RuntimeError("❌ Failed to reach start pose")


def move_down_until_stop():
    """Move down until force or slip threshold is reached."""
    global current_pose
    while True:
        fz = rtde_r.getActualTCPForce()[2] - baseline_force
        t = time.time() - t0
        z = rtde_r.getActualTCPPose()[2]

        forces.append(fz); times.append(t); positions.append(z)
        print(f"Force: {fz:.2f} N, z={z:.4f} m")

        if fz > force_limit: 
            print("🛑 Force limit reached!"); break
        if len(forces) > 1 and (forces[-1] - forces[-2]) < slip_drop:
            print("⚠️ Sudden slip detected!"); break

        new_pose = current_pose.copy(); new_pose[2] -= speed * dt
        if rtde_c.moveL(new_pose, speed, acc): current_pose = new_pose
        else: break
        time.sleep(dt)


def move_up():
    """Lift up by max_lift."""
    new_pose = current_pose.copy(); new_pose[2] += max_lift
    rtde_c.moveL(new_pose, speed, acc)
    print("⬆️ Lifted back up")


def plot_data():
    """Plot force vs time with max highlight."""
    plt.plot(times, forces, label="Force (N)", color="blue")
    plt.xlabel("Time (s)"); plt.ylabel("Force (N)")
    plt.title("Force vs Time"); plt.grid(); plt.legend()

    max_f = max(forces); idx = forces.index(max_f)
    plt.annotate(f"Max: {max_f:.2f} N\nZ: {positions[idx]:.4f} m",
                 xy=(times[idx], max_f),
                 xytext=(times[idx]+0.5, max_f+1),
                 arrowprops=dict(facecolor='red', width=2, headwidth=8),
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="red"))
    plt.show()


# --- Execution ---
time.sleep(1)
move_to_start()
move_down_until_stop()
move_up()
plot_data()

rtde_c.stopScript()
