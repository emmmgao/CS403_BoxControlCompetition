import mujoco
import numpy as np
import mujoco
from BoxControlHandler import BoxControlHandle

class YourCtrl:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m = m
        self.d = d
        self.ee_body = "EE_box"
        self.init_qpos = d.qpos.copy()
        self.boxCtrlhdl = BoxControlHandle(self.m, self.d)
        self.boxCtrlhdl.set_difficulty(0.1)  # Set difficulty level

        self.ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_site")
        self.qpos_dim = 6  # Control the first six joints

    def get_box_target_pose(self):
        s1 = self.d.sensordata[0:3]
        s2 = self.d.sensordata[3:6]
        s3 = self.d.sensordata[6:9]
        s4 = self.d.sensordata[9:12]

        # Safety check
        if np.linalg.norm(s1) < 1e-4 or np.linalg.norm(s2 - s1) < 1e-6 or np.linalg.norm(s3 - s1) < 1e-6:
            print("[WARN] sensor data not ready")
            return self.d.xpos[self.ee_site_id]  # Return current position as target

        center = (s1 + s2 + s3 + s4) / 4.0
        v1 = s2 - s1
        v2 = s3 - s1
        z = np.cross(v1, v2)
        z /= np.linalg.norm(z)
        x = s4 - s1
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        R = np.column_stack([x, y, z])
        target_pos = center + R @ np.array([0.02, 0.0, 0.0])

        return target_pos

    def get_jacobian(self):
        J_pos_full = np.zeros((3, self.m.nv))
        J_rot_full = np.zeros((3, self.m.nv))  # Required but unused
        mujoco.mj_jacSite(self.m, self.d, J_pos_full, J_rot_full, self.ee_site_id)
        return J_pos_full[:, :self.qpos_dim]

    def update(self):
        if self.d.time < 0.05:
            return  # Wait for sensor data to initialize

        # Get target and current position
        target_pos = self.get_box_target_pose()
        ee_pos = self.d.xpos[self.ee_site_id]

        # Compute position error
        pos_err = target_pos - ee_pos

        if np.any(np.isnan(pos_err)) or np.linalg.norm(pos_err) < 1e-6:
            print("[WARN] Invalid target or already at target")
            return

        # Get Jacobian
        J_pos = self.get_jacobian()

        # Use damped pseudo-inverse to compute joint velocity
        damping = 1e-4
        J_pinv = np.linalg.pinv(J_pos, rcond=damping)
        dq = J_pinv @ (2.0 * pos_err)  # Gain of 2.0, tunable

        # Velocity limit protection
        dq = np.clip(dq, -1.0, 1.0)

        # PD control
        for i in range(self.qpos_dim):
            self.d.ctrl[i] = 50.0 * dq[i] - 2.0 * self.d.qvel[i]
