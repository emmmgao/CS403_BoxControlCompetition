import mujoco
import numpy as np
import mujoco
from BoxControlHandler import BoxControlHandle

class YourCtrl:
  
  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.init_qpos = d.qpos.copy()

    self.boxCtrlhdl = BoxControlHandle(self.m,self.d)
    self.boxCtrlhdl.set_difficulty(0.9) #set difficulty level

    self.ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

  def update(self):

  #  for i in range(6):
  #      self.d.ctrl[i] = 150.0*(self.init_qpos[i] - self.d.qpos[i])  - 5.2 *self.d.qvel[i]
   
    pos_err = self.boxCtrlhdl.get_EE_pos_err()


    jacp = np.zeros((3, self.m.nu))
    mujoco.mj_jacSite(self.m, self.d, jacp, None, self.ee_site_id)


    jacp_pinv = np.linalg.pinv(jacp)
    print(f"Jacobian: {jacp_pinv}")
    dq = jacp_pinv @ pos_err
    

    Kp = 300.0
    Kd = 15.0
    for i in range(6):
        self.d.ctrl[i] = Kp * dq[i] - Kd * self.d.qvel[i]
    # self.d.ctrl[5] = 500.0 * (self.init_qpos[5] - self.d.qpos[5]) - 10.0 * self.d.qvel[5]
