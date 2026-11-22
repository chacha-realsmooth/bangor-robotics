"""
Quadruped robot dog: MuJoCo + Gymnasium environment, built for sim-to-real transfer.

Robot  : rigid trunk + 4 legs x 3 joints (hip abduction, hip pitch, knee) = 12 actuated DoF
Terrain: flat plane
Sensing: IMU (gyroscope + accelerometer) + joint encoders  (set imu_only=True to drop encoders)
Task   : walk at a commanded forward speed (and stand still when commanded 0).

Sim-to-real features
  * Everything hardware-specific lives in RobotConfig -> edit it to match your robot.
  * Motors are modelled the way real drivers work: a PD loop running at physics rate, torque
    clipped by a torque-speed curve (not an ideal position actuator).
  * Domain randomisation (per episode): masses, centre of mass, joint damping/friction, foot
    friction, motor kp/kd/torque/speed, encoder calibration offsets, IMU mounting pose, sensor
    noise/bias/scale, action latency (0-40 ms), IMU latency, random pushes.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

LEGS = ["FL", "FR", "RL", "RR"]  # front-left, front-right, rear-left, rear-right
JOINT_TYPES = ["abd", "hip", "knee"]
JOINT_NAMES = [f"{leg}_{j}" for leg in LEGS for j in JOINT_TYPES]  # policy/hardware joint order


@dataclass
class RobotConfig:
    """Edit these to match your robot. Units: metres, kg, radians, N*m."""
    # --- geometry
    trunk_size: tuple = (0.40, 0.20, 0.10)   # full length (x), width (y), height (z)
    hip_xy: tuple = (0.20, 0.11)              # |x|, |y| of the hip abduction axis from trunk centre
    abd_offset: float = 0.05                  # lateral offset: abduction axis -> thigh plane
    thigh_len: float = 0.20
    calf_len: float = 0.20
    foot_radius: float = 0.022
    # --- masses
    trunk_mass: float = 5.0                   # include battery + electronics
    abd_mass: float = 0.30
    thigh_mass: float = 0.30
    calf_mass: float = 0.15
    foot_mass: float = 0.03
    # --- joints (range in rad; sign convention: +hip pitch swings foot backwards, knee is negative)
    abd_range: tuple = (-0.5, 0.5)
    hip_range: tuple = (-0.6, 1.6)
    knee_range: tuple = (-2.6, -0.3)
    default_knee: float = -1.6                # standing pose is solved from this + leg lengths
    # --- motors (same for all joints; per-joint randomisation is applied on top)
    kp: float = 30.0                          # N*m/rad   <- set the SAME gains in your motor drivers
    kd: float = 0.8                           # N*m*s/rad
    torque_limit: float = 20.0                # peak torque at joint output (after gearbox)
    max_joint_speed: float = 20.0             # no-load speed at joint output (rad/s)
    joint_damping: float = 0.1
    joint_friction: float = 0.02
    armature: float = 0.005                   # reflected rotor inertia (matters with geared motors!)
    # --- IMU
    imu_pos: tuple = (0.0, 0.0, 0.0)          # IMU position in trunk frame
    gyro_noise: float = 0.05                  # rad/s std per sample
    accel_noise: float = 0.2                  # m/s^2 std per sample
    # --- timing
    control_hz: int = 50
    sim_dt: float = 0.002

    def joint_ranges(self):
        return np.array([self.abd_range, self.hip_range, self.knee_range] * 4)


def solve_stand_pose(cfg: RobotConfig):
    """Hip pitch that puts the foot directly under the hip for the chosen knee angle."""
    k = cfg.default_knee
    best = None
    for th in np.linspace(cfg.hip_range[0], cfg.hip_range[1], 4001):
        phi = th + k
        x = -cfg.thigh_len * np.sin(th) - cfg.calf_len * np.sin(phi)
        z = -cfg.thigh_len * np.cos(th) - cfg.calf_len * np.cos(phi)
        if best is None or abs(x) < best[0]:
            best = (abs(x), th, z)
    err, th, z = best
    if err > 0.01:
        raise ValueError("Cannot stand with this default_knee/leg length/hip range - adjust RobotConfig.")
    return np.array([0.0, th, k]), -z + cfg.foot_radius


def build_xml(cfg: RobotConfig, stand_height: float) -> str:
    hx, hy = cfg.hip_xy
    tx, ty, tz = (s / 2 for s in cfg.trunk_size)
    legs, actuators = [], []
    for leg in LEGS:
        x = hx if leg[0] == "F" else -hx
        y = hy if leg[1] == "L" else -hy
        side = 1 if y > 0 else -1
        legs.append(f"""
      <body name="{leg}_hip" pos="{x} {y} 0">
        <joint name="{leg}_abd" type="hinge" axis="1 0 0" range="{cfg.abd_range[0]} {cfg.abd_range[1]}"/>
        <geom type="sphere" size="0.035" mass="{cfg.abd_mass}" contype="0" conaffinity="0" rgba=".3 .3 .3 1"/>
        <body name="{leg}_thigh" pos="0 {side * cfg.abd_offset} 0">
          <joint name="{leg}_hip" type="hinge" axis="0 1 0" range="{cfg.hip_range[0]} {cfg.hip_range[1]}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-cfg.thigh_len}" size="0.02" mass="{cfg.thigh_mass}"
                contype="0" conaffinity="0" rgba=".85 .5 .1 1"/>
          <body name="{leg}_calf" pos="0 0 {-cfg.thigh_len}">
            <joint name="{leg}_knee" type="hinge" axis="0 1 0" range="{cfg.knee_range[0]} {cfg.knee_range[1]}"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-cfg.calf_len}" size="0.015" mass="{cfg.calf_mass}"
                  contype="0" conaffinity="0" rgba=".7 .4 .1 1"/>
            <geom name="{leg}_foot" type="sphere" pos="0 0 {-cfg.calf_len}" size="{cfg.foot_radius}"
                  mass="{cfg.foot_mass}" contype="1" conaffinity="0" friction="0.8 0.02 0.01"
                  rgba=".1 .1 .1 1"/>
          </body>
        </body>
      </body>""")
        for j in JOINT_TYPES:
            T = cfg.torque_limit * 1.5  # headroom; real limiting is done in the motor model
            actuators.append(f'<motor name="{leg}_{j}" joint="{leg}_{j}" ctrlrange="-{T} {T}"/>')

    # Floor friction is deliberately tiny: MuJoCo uses max(geom1, geom2) friction, so the
    # (randomised) foot friction is what actually determines grip.
    return f"""
<mujoco model="quadruped">
  <compiler angle="radian"/>
  <option timestep="{cfg.sim_dt}" integrator="implicitfast"/>
  <default>
    <joint damping="{cfg.joint_damping}" armature="{cfg.armature}" frictionloss="{cfg.joint_friction}"/>
  </default>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".6 .75 .9" rgb2="0 0 0" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".25 .3 .35" rgb2=".18 .22 .27" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="20 20" reflectance=".1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="grid"
          friction="0.05 0.005 0.0001" contype="1" conaffinity="1"/>
    <body name="trunk" pos="0 0 {stand_height + 0.01}">
      <freejoint name="root"/>
      <geom name="trunk" type="box" size="{tx} {ty} {tz}" mass="{cfg.trunk_mass}" contype="1" conaffinity="0"
            rgba=".2 .4 .8 1"/>
      <site name="imu" pos="{cfg.imu_pos[0]} {cfg.imu_pos[1]} {cfg.imu_pos[2]}" size="0.01"/>
      {''.join(legs)}
    </body>
  </worldbody>
  <actuator>
    {''.join(actuators)}
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
  </sensor>
</mujoco>
"""


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: RobotConfig | None = None,
        imu_only: bool = False,          # True = drop joint encoders (hard! an IMU can't see joint angles)
        history: int = 3,                # stacked observation frames (more helps with latency)
        max_steps: int = 1000,           # 20 s episodes
        action_scale: float = 0.5,       # rad offset from default pose at action = +-1
        cmd_range: tuple[float, float] = (0.2, 0.8),  # forward speed command range (m/s)
        stand_prob: float = 0.1,         # fraction of episodes commanded to stand still
        randomize: bool = True,          # domain randomisation on/off
        push_interval: int = 200,        # steps between random pushes (0 disables)
    ):
        super().__init__()
        self.cfg = cfg or RobotConfig()
        self.render_mode = render_mode
        self.imu_only, self.history = imu_only, history
        self.max_steps, self.action_scale = max_steps, action_scale
        self.cmd_range, self.stand_prob = cmd_range, stand_prob
        self.randomize, self.push_interval = randomize, push_interval

        self.frame_skip = int(round(1.0 / (self.cfg.control_hz * self.cfg.sim_dt)))
        self.default_leg, self.stand_height = solve_stand_pose(self.cfg)
        self.default_q = np.tile(self.default_leg, 4)
        rng_ = self.cfg.joint_ranges()
        self.q_lo, self.q_hi = rng_[:, 0], rng_[:, 1]
        self.q_mid, self.q_half = (self.q_lo + self.q_hi) / 2, (self.q_hi - self.q_lo) / 2

        self.model = mujoco.MjModel.from_xml_string(build_xml(self.cfg, self.stand_height))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * self.frame_skip

        m = self.model
        n2i = mujoco.mj_name2id
        jids = [n2i(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.qadr, self.dadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
        self.trunk_id = n2i(m, mujoco.mjtObj.mjOBJ_GEOM, "trunk")
        self.trunk_body = n2i(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.floor_id = n2i(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_ids = [n2i(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot") for leg in LEGS]
        self.imu_site = n2i(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.gyro_adr = m.sensor_adr[n2i(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")]
        self.accel_adr = m.sensor_adr[n2i(m, mujoco.mjtObj.mjOBJ_SENSOR, "accel")]

        # nominal values to randomise around
        self._b_mass, self._b_inertia, self._b_ipos = m.body_mass.copy(), m.body_inertia.copy(), m.body_ipos.copy()
        self._b_damp, self._b_fric = m.dof_damping[self.dadr].copy(), m.dof_frictionloss[self.dadr].copy()
        self._b_site_pos = m.site_pos[self.imu_site].copy()

        frame_dim = 6 + (0 if imu_only else 24) + 12 + 3  # imu + (q,qd) + last action + command
        self.frame_dim = frame_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (frame_dim * history,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)

        self._frames: collections.deque = collections.deque(maxlen=history)
        self._viewer = self._renderer = self._cam = None

    # ------------------------------------------------------------------ randomisation
    def _randomize_model(self):
        m, c, rng = self.model, self.cfg, self.np_random
        if self.randomize:
            u = rng.uniform
            s = u(0.9, 1.1, m.nbody)
            s[self.trunk_body] = u(0.8, 1.25)                      # payload / battery variation
            s[0] = 1.0
            m.body_mass[:] = self._b_mass * s
            m.body_inertia[:] = self._b_inertia * s[:, None]
            m.body_ipos[:] = self._b_ipos
            m.body_ipos[self.trunk_body] += u(-0.02, 0.02, 3)      # centre-of-mass offset
            m.dof_damping[self.dadr] = self._b_damp * u(0.5, 2.0, 12)
            m.dof_frictionloss[self.dadr] = self._b_fric * u(0.5, 2.0, 12)
            m.geom_friction[self.foot_ids, 0] = u(0.4, 1.25) * u(0.9, 1.1, 4)
            # IMU mounting error: +-2 cm position, ~1 deg orientation
            m.site_pos[self.imu_site] = self._b_site_pos + u(-0.02, 0.02, 3)
            a = rng.normal(0, 0.02, 3)
            qn = np.concatenate([[1.0], a / 2]); m.site_quat[self.imu_site] = qn / np.linalg.norm(qn)
            # motors
            self.kp = c.kp * u(0.8, 1.2, 12)
            self.kd = c.kd * u(0.8, 1.2, 12)
            self.tau_max = c.torque_limit * u(0.8, 1.05, 12)
            self.w_max = c.max_joint_speed * u(0.8, 1.1, 12)
            self.q_offset = rng.normal(0, 0.02, 12)                # encoder calibration error (rad)
            self.action_delay = int(rng.integers(0, 3))            # 0-2 policy steps = 0-40 ms
            self.imu_delay = int(rng.integers(0, 2))
            self.gyro_bias = rng.normal(0, 0.01, 3)
            self.accel_bias = rng.normal(0, 0.1, 3)
            self.accel_scale = 1.0 + rng.normal(0, 0.02, 3)
        else:
            m.body_mass[:], m.body_inertia[:], m.body_ipos[:] = self._b_mass, self._b_inertia, self._b_ipos
            m.dof_damping[self.dadr], m.dof_frictionloss[self.dadr] = self._b_damp, self._b_fric
            m.geom_friction[self.foot_ids, 0] = 0.8
            m.site_pos[self.imu_site] = self._b_site_pos
            m.site_quat[self.imu_site] = [1, 0, 0, 0]
            self.kp, self.kd = np.full(12, c.kp), np.full(12, c.kd)
            self.tau_max, self.w_max = np.full(12, c.torque_limit), np.full(12, c.max_joint_speed)
            self.q_offset = np.zeros(12)
            self.action_delay = self.imu_delay = 0
            self.gyro_bias = self.accel_bias = np.zeros(3)
            self.accel_scale = np.ones(3)
        mujoco.mj_setConst(m, self.data)

    # ------------------------------------------------------------------ observations
    def _read_imu(self):
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        accel = self.data.sensordata[self.accel_adr:self.accel_adr + 3].copy()
        if self.randomize:
            gyro += self.gyro_bias + self.np_random.normal(0, self.cfg.gyro_noise, 3)
            accel = accel * self.accel_scale + self.accel_bias + self.np_random.normal(0, self.cfg.accel_noise, 3)
        return gyro, accel

    def _frame(self):
        self._imu_buf.append(self._read_imu())
        gyro, accel = self._imu_buf[-1 - self.imu_delay]
        parts = [gyro, accel]
        if not self.imu_only:
            q = self.data.qpos[self.qadr] + self.q_offset
            qd = self.data.qvel[self.dadr]
            if self.randomize:
                q = q + self.np_random.normal(0, 0.005, 12)
                qd = qd + self.np_random.normal(0, 0.05, 12)
            parts += [q - self.default_q, qd * 0.1]
        parts += [self.last_action, self.cmd]
        return np.concatenate(parts).astype(np.float32)

    def _obs(self):
        self._frames.append(self._frame())
        return np.concatenate(self._frames)

    def _contacts(self):
        feet = np.zeros(4, dtype=bool)
        trunk_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.floor_id:
                g1, g2 = g2, g1
            if g2 != self.floor_id:
                continue
            if g1 == self.trunk_id:
                trunk_hit = True
            elif g1 in self.foot_ids:
                feet[self.foot_ids.index(g1)] = True
        return feet, trunk_hit

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_model()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, self.stand_height + 0.01]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.qadr] = self.default_q + self.np_random.uniform(-0.05, 0.05, 12)
        mujoco.mj_forward(self.model, self.data)

        vx = 0.0 if self.np_random.random() < self.stand_prob else self.np_random.uniform(*self.cmd_range)
        self.cmd = np.array([vx, 0.0, 0.0], dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self._act_buf = collections.deque([np.zeros(12, np.float32)] * 3, maxlen=3)
        self._imu_buf = collections.deque(maxlen=2)
        self._imu_buf.append(self._read_imu())
        self.air_time = np.zeros(4)
        self.steps = 0
        self._frames.clear()
        f = self._frame()
        for _ in range(self.history):
            self._frames.append(f)
        return np.concatenate(self._frames), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._act_buf.append(action)
        applied = self._act_buf[-1 - self.action_delay]                    # communication latency
        target = np.clip(self.default_q + self.action_scale * applied, self.q_lo, self.q_hi)

        if self.randomize and self.push_interval and self.steps > 0 and self.steps % self.push_interval == 0:
            self.data.qvel[0:2] += self.np_random.uniform(-0.4, 0.4, 2)

        for _ in range(self.frame_skip):                                   # motor driver loop @ physics rate
            q = self.data.qpos[self.qadr] + self.q_offset
            qd = self.data.qvel[self.dadr]
            tau = self.kp * (target - q) - self.kd * qd
            lim = self.tau_max * np.clip(1.0 - np.abs(qd) / self.w_max, 0.0, 1.0)  # torque-speed curve
            self.data.ctrl[:] = np.clip(tau, -lim, lim)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        # ---- state in the body frame
        R = self.data.xmat[self.trunk_body].reshape(3, 3)
        v_b = R.T @ self.data.qvel[0:3]
        w_b = self.data.qvel[3:6]                    # free-joint angular velocity is already body-frame
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        q = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.dadr]

        feet, trunk_hit = self._contacts()
        self.air_time += self.dt
        first_contact = (self.air_time > 0) & feet
        moving = np.linalg.norm(self.cmd[:2]) > 0.1
        r_air = np.sum((np.clip(self.air_time, 0, 0.5) - 0.25) * first_contact) * moving
        self.air_time *= ~feet

        slip, vel6 = 0.0, np.zeros(6)
        for i, g in enumerate(self.foot_ids):
            if feet[i]:
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, g, vel6, 0)
                slip += np.sum(vel6[3:5] ** 2)
        limit_viol = np.maximum(np.abs(q - self.q_mid) - 0.9 * self.q_half, 0.0).sum()

        # ---- reward
        r_lin = np.exp(-np.sum((self.cmd[:2] - v_b[:2]) ** 2) / 0.25)
        r_yaw = np.exp(-((self.cmd[2] - w_b[2]) ** 2) / 0.25)
        reward = (
            1.0 * r_lin
            + 0.5 * r_yaw
            + 0.3 * r_air
            - 2.0 * v_b[2] ** 2                                    # no bouncing
            - 0.05 * np.sum(w_b[:2] ** 2)                          # no roll/pitch wobble
            - 1.0 * np.sum(g_b[:2] ** 2)                           # stay level
            - 2e-4 * np.sum(self.data.actuator_force ** 2)         # torque / heat
            - 1e-4 * np.sum(qd ** 2)                               # joint speed (hardware limit)
            - 0.01 * np.sum((action - self.last_action) ** 2)      # smooth actions
            - 0.1 * np.sum((q - self.default_q) ** 2)
            - 5.0 * (self.data.qpos[2] - self.stand_height) ** 2
            - 0.1 * slip                                           # no foot skating
            - 5.0 * limit_viol                                     # stay off joint stops
        )

        terminated = bool(trunk_hit or g_b[2] > -0.5)  # fell or tilted > 60 deg
        if terminated:
            reward -= 2.0
        truncated = self.steps >= self.max_steps

        self.last_action = action
        obs = self._obs()
        if self.render_mode == "human":
            self.render()
            time.sleep(self.dt)
        return obs, float(reward), terminated, truncated, {"vx": float(v_b[0]), "cmd_vx": float(self.cmd[0])}

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
                self._cam = mujoco.MjvCamera()
                self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._cam.trackbodyid = self.trunk_body
                self._cam.distance, self._cam.elevation, self._cam.azimuth = 1.8, -20, 135
            self._renderer.update_scene(self.data, self._cam)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
        if self._renderer is not None:
            self._renderer.close()
