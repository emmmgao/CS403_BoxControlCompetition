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
    self.boxCtrlhdl.set_difficulty(0.25) #set difficulty level
    
    self.start_time = None
    self.q_target = None
    
    
  def newton_raphson(self, target_pos, target_quat, num_steps, lr):
    original_qpos = self.d.qpos[:6].copy()
    q_hist = [] #history of joint positions
    error_hist = []
    for steps in range(num_steps):
      
      #get current ee position
      ee_pos = self.boxCtrlhdl._get_ee_position()
      ee_quat = self.boxCtrlhdl._get_ee_orientation()
      
      #compute errors
      #position error
      pos_error = target_pos - ee_pos
      #orientation error
      quat_err = self.boxCtrlhdl.quat_multiply(target_quat, self.boxCtrlhdl.quat_inv(ee_quat))
      rot_err = self.boxCtrlhdl.quat2so3(quat_err) 
      #full 
      error = np.hstack([pos_error, rot_err]) #6 vector
      error_hist.append(error)
      #compute jacobian
      jacp = np.zeros((3, self.m.nv))
      jacr = np.zeros((3, self.m.nv))
      mujoco.mj_jac(self.m, self.d, jacp, jacr, ee_pos, self.boxCtrlhdl.ee_id)
      J = np.vstack([jacp[:, :6], jacr[:, :6]])
      
      try:
          dq = np.linalg.solve(J, error)
      except np.linalg.LinAlgError:
          dq = np.linalg.pinv(J) @ error

      q_hist.append(self.d.qpos[:6].copy())
      
      self.d.qpos[:6] += lr * dq
      
      # for i in range(6):
      #   self.d.qpos[i] = np.clip(self.d.qpos[i], self.m.jnt_range[i][0], self.m.jnt_range[i][1]) 
    
      mujoco.mj_forward(self.m, self.d) #update the jacobian at each change
      #put the new positions in q_hist, or just get the last one here. 
      
    
    self.d.qpos[:6] = original_qpos
    mujoco.mj_forward(self.m, self.d)
      
    return q_hist, error_hist

    #Find joint limits
  #gause newton: need step size
  def update(self):
    if self.start_time is None:
      self.start_time = self.d.time

    box_sensor_names = ["mould_pos_sensor1", "mould_pos_sensor2", "mould_pos_sensor3", "mould_pos_sensor4"]
    box_sensors = []
    
    for name in box_sensor_names:
      idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
      pos = self.d.sensordata[idx*3 : idx*3+3]
      box_sensors.append(pos)

    target_quat, _ = self.boxCtrlhdl.box_orientation(*box_sensors)

    target_quat = self.boxCtrlhdl.rotate_quat_90_y(target_quat)

    pos_err = self.boxCtrlhdl.get_EE_pos_err()
    target_pos = self.boxCtrlhdl._get_ee_position() + pos_err

    q_hist, err_hist = self.newton_raphson(target_pos, target_quat, num_steps=10, lr=0.8)
    if q_hist:
        q_target = q_hist[-1]
    else:
        print("[update] IK failed — reverting to home")
        q_target = self.q_home.copy()
        
  
    for i in range(6):
      self.d.ctrl[i] = 150.0*(q_target[i] - self.d.qpos[i])  - 5.2 *self.d.qvel[i]
    
       


  
   

