import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

class PegInsertion:
    def __init__(self, motion, rtde_c, rtde_r,
                 orientation_upright=[np.pi, 0, np.pi/2 + np.radians(-30)]):
        """
        motion: MotionNode instance
        rtde_c: RTDEControlInterface
        rtde_r: RTDEReceiveInterface
        orientation_upright: default upright tool orientation
        """
        self.motion = motion
        self.rtde_c = rtde_c
        self.rtde_r = rtde_r
        self.orientation_upright = orientation_upright

    def set_pose_orientation(self, pose, rpy):
        """Apply orientation (Euler angles) to pose in rotvec format."""
        pose = pose.copy()
        pose[3:6] = R.from_euler('xyz', rpy).as_rotvec()
        return pose

    def build_spiral_points(self, start_pose_world, *,
                            spiral_steps=20, num_loops=1.8,
                            spiral_radius=0.009, dz_per_spiral=-0.0001, k=2):
        """Generate spiral trajectory points in world frame."""
        pts = []
        for i in range(spiral_steps):
            frac = i / float(spiral_steps)
            angle = 2*np.pi*num_loops*frac
            radius = spiral_radius * (1 - (1 - frac)**k)
            p = np.array(start_pose_world, float).copy()
            p[0] += radius * np.cos(angle)
            p[1] += radius * np.sin(angle)
            p[2] += i * dz_per_spiral
            pts.append(p)
        return pts

    def precompute_spiral_traj(self, start_pose_world, orientation_rpy, *,
                               spiral_steps=20, num_loops=1.8,
                               spiral_radius=0.009, dz_per_spiral=-0.0001, k=2,
                               steps_per_segment=8):
        """Synchronously precompute joint-space spiral trajectory (servoJ path)."""
        pts = self.build_spiral_points(
            start_pose_world,
            spiral_steps=spiral_steps,
            num_loops=num_loops,
            spiral_radius=spiral_radius,
            dz_per_spiral=dz_per_spiral, k=k
        )
        return self.motion.precompute_servoj_joint_path(
            pts, orientation=orientation_rpy, steps_per_segment=steps_per_segment
        )

    def precompute_spiral_traj_async(self, start_pose_world, orientation_rpy, **kwargs):
        """Asynchronously precompute trajectory in a background thread."""
        holder, done = {"traj": None, "err": None}, threading.Event()

        def _worker():
            try:
                holder["traj"] = self.precompute_spiral_traj(start_pose_world, orientation_rpy, **kwargs)
            except Exception as e:
                holder["err"] = str(e)
            finally:
                done.set()

        threading.Thread(target=_worker, daemon=True).start()

        class Handle:
            def ready(self): return done.is_set()
            def result(self): return holder["traj"]
            def error(self):  return holder["err"]
            def wait(self, timeout=None): return done.wait(timeout=timeout)

        return Handle()

    def leadin_spiral_without_sensor(
        self,
        start_pose,
        tilt_deg=3.0,
        spiral_radius=0.004,
        spiral_steps=15,
        num_loops=1.8,
        dz_per_spiral=-0.0001,
    ):
        """Perform peg-in-hole using a predefined spiral, no sensor feedback."""
        print("\n=== PEG-IN-HOLE (No Sensor) ===")

        # Tilt pose
        tilt_rad = np.radians(tilt_deg)
        base_point = self.motion.rtde_r.getActualTCPPose()
        pose_tilted = base_point.copy()
        pose_tilted[4] -= tilt_rad

        new_orient = [np.pi, 0, np.pi/2]
        new_orient[1] += tilt_rad
        new_orient[2] += np.radians(-30)

        start_pose0 = start_pose.copy()
        start_pose0[2] += 0.009
        start_pose0[1] += 0.003

        # Move above insertion point
        pre_pose = start_pose.copy()
        pre_pose[2] += 0.05
        self.motion.go_to_point(pre_pose)
        self.motion.go_to_point(start_pose0, new_orient)

        # Generate spiral points
        spiral_points = []
        for i in range(spiral_steps):
            frac = i / spiral_steps
            angle = 2*np.pi*num_loops*frac
            radius = spiral_radius * (1 - (1 - frac)**2)
            wiggle_pose = start_pose0.copy()
            wiggle_pose[0] += radius * np.cos(angle)
            wiggle_pose[1] += radius * np.sin(angle)
            wiggle_pose[2] += i * dz_per_spiral
            spiral_points.append(wiggle_pose)
        print(f"Generated {len(spiral_points)} spiral points")

        # Execute spiral
        self.motion.move_joint_path_servoj_interpolated(spiral_points, new_orient, dt=0.03)
        self.motion.rtde_c.servoStop()

        # Align upright and insert
        pose = start_pose0.copy()
        pose[2] -= 0.002
        self.motion.go_to_point(pose, self.orientation_upright)

        print("⬇️ Inserting peg downwards...")
        down_pose = pose.copy()
        down_pose[2] -= 0.01
        self.motion.go_to_point(down_pose, self.orientation_upright)

        print("✅ Peg successfully inserted!")
        print("=== END PEG-IN-HOLE ===\n")
        return True
