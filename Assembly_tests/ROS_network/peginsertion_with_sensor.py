import numpy as np
from scipy.spatial.transform import Rotation as R

class PegInsertion:
    def __init__(self, motion, rtde_c, rtde_r, sensor_client, orientation_upright=[np.pi, 0, np.pi/2]):
        """
        motion: MotionNode instance
        rtde_c: RTDEControlInterface
        rtde_r: RTDEReceiveInterface
        sensor_client: XelaSensorClient (must implement get_sensor_data())
        """
        self.motion = motion
        self.rtde_c = rtde_c
        self.rtde_r = rtde_r
        self.sensor_client = sensor_client
        self.orientation_upright = orientation_upright

    def set_pose_orientation(self, pose, rpy):
        pose = pose.copy()
        pose[3:6] = R.from_euler('xyz', rpy).as_rotvec()
        return pose

    def get_cop_and_force_local(self):
        """Fetch CoP and force info from tactile sensor client."""
        data = self.sensor_client.get_sensor_data()
        if not data:
            return None
        return {
            "total_force": data["total_force"],
            "total_forcex": data["total_forcex"],
            "cop": data["CoP_plus_world"]
        }

    def leadin_spiral(
        self,
        start_pose,
        tilt_deg=4.0,
        spiral_radius=0.009,
        spiral_steps=35,
        num_loops=2.1,
        dz_per_spiral=-0.0001,
        force_drop_abs=0.55,
        first_outer_extra_dz=-0.001,
        outer_hold_loops=0.6,
        outer_dz_scale=1.0,
        free_force_abs=0.25,
        free_force_consecutive=3
    ):
        """Sensor-based peg-in-hole with spiral search and force monitoring."""
        print("\n=== SENSOR-BASED PEG-IN-HOLE (Force Drop Criterion) ===")

        # --- Initial push test in Z: check if Fx rises ---
        success = False
        max_push_steps, step_size = 10, 0.0005
        last_force_x = (self.get_cop_and_force_local() or {}).get("total_forcex", 999.0)
        down_pose = self.rtde_r.getActualTCPPose()

        for i in range(max_push_steps):
            force_x = (self.get_cop_and_force_local() or {}).get("total_forcex", 999.0)
            delta_fx = force_x - last_force_x
            print(f"Push {i+1}: Fx={force_x:.3f}N (Δ={delta_fx:.3f})")
            if delta_fx > 0.2:
                print("✅ Sudden Fx increase detected (contact confirmed)")
                success = True
                break
            down_pose[2] -= step_size
            self.rtde_c.moveL(down_pose, 0.5, 0.5)
            last_force_x = force_x

        if not success:
            print("⚠️ No significant Fx increase detected — aborting.")
            return True

        # --- Tilt TCP slightly ---
        tilt_rad = np.radians(tilt_deg)
        base_point = self.motion.rtde_r.getActualTCPPose()
        pose_tilted = base_point.copy()
        pose_tilted[4] -= tilt_rad
        pose_tilted[2] += 0.0016
        self.rtde_c.moveL(pose_tilted, 3, 3)

        # Slight lift before spiral
        start_pose = np.array(start_pose, float).copy()
        start_pose[2] += 0.0025

        # Init force ref
        last_force = (self.get_cop_and_force_local() or {}).get("total_forcex", 999.0)

        found_lead_in, delta_forces, z_curr, z_prev = False, [], pose_tilted[2], pose_tilted[2]
        outer_hold_steps = max(1, int(spiral_steps * outer_hold_loops / max(num_loops, 1e-6)))
        outer_hold_steps = min(outer_hold_steps, max(spiral_steps-1, 1))
        free_count = 0

        for i in range(spiral_steps):
            frac = i / max(spiral_steps - 1, 1)
            start_angle = np.pi / 1.4
            angle = 2 * np.pi * num_loops * frac + start_angle

            if i < outer_hold_steps:
                radius, dz_step = spiral_radius, dz_per_spiral * outer_dz_scale
            else:
                inner_frac = (i - outer_hold_steps) / max(spiral_steps - outer_hold_steps - 1, 1)
                radius, dz_step = spiral_radius * (1 - inner_frac), dz_per_spiral

            z_curr = (z_curr + first_outer_extra_dz if i == 0 else z_curr + dz_step)
            if z_curr > z_prev:  # never move up
                z_curr = z_prev + min(dz_step, 0.0)

            wiggle_pose = pose_tilted.copy()
            wiggle_pose[0] += radius * np.cos(angle)
            wiggle_pose[1] += radius * np.sin(angle)
            wiggle_pose[2] = z_curr
            self.rtde_c.moveL(wiggle_pose, 0.5, 0.5)

            if i == 0:
                last_force = (self.get_cop_and_force_local() or {}).get("total_forcex", 999.0)

            force_now = (self.get_cop_and_force_local() or {}).get("total_forcex", 999.0)
            delta_force = last_force - force_now
            delta_forces.append(abs(delta_force))
            print(f"Spiral {i+1}/{spiral_steps}: r={radius*1000:.1f}mm, Fx={force_now:.3f}N (Δ={delta_force:.3f})")

            if delta_force > force_drop_abs:
                print(f"✅ Lead-in/Chamfer detected (ΔFx>{force_drop_abs}N).")
                # Micro-spiral refinement
                x0, y0, z_m, angle0 = wiggle_pose[0], wiggle_pose[1], wiggle_pose[2], angle
                for j in range(15):
                    frac_m = j / 14
                    r_m = 0.005 * frac_m
                    theta_m = 2*np.pi*1.6*frac_m + angle0
                    z_m += dz_per_spiral
                    micro_pose = wiggle_pose.copy()
                    micro_pose[0] = x0 + r_m*np.cos(theta_m)
                    micro_pose[1] = y0 + r_m*np.sin(theta_m)
                    micro_pose[2] = z_m
                    self.rtde_c.moveL(micro_pose, 0.5, 0.5)
                found_lead_in = True
                break

            last_force, z_prev = force_now, z_curr

        if not found_lead_in:
            print("⚠️ No lead-in detected after spiral search — aborting.")
            return False

        # --- Upright and insert ---
        upright_pose = self.rtde_r.getActualTCPPose()
        upright_pose = self.set_pose_orientation(upright_pose, self.orientation_upright)
        self.rtde_c.moveL(upright_pose, 3, 3)

        print("⬇️ Inserting peg downwards...")
        down_pose = upright_pose.copy()
        down_pose[2] -= 0.002
        self.rtde_c.moveL(down_pose, 3, 3)
        print("✅ Peg successfully inserted!")
        print("=== END SENSOR-BASED PEG-IN-HOLE ===\n")
        return True
