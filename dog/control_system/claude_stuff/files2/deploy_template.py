"""
On-robot control loop TEMPLATE. Runs policy.onnx at 50 Hz.

You must implement RobotInterface for your hardware (4 small methods). Everything else
(observation building, history stacking, safety limits, start-up ramp) is done for you.

Frame conventions that MUST match the sim (verify each on the bench before powering the legs):
  * Trunk frame: x forward, y left, z up.
  * Gyro: rad/s (NOT deg/s). Accelerometer: m/s^2 specific force -> reads about +9.8 on z when level
    and at rest (NOT in g units, and not negated).
  * Joints: order = JOINT_NAMES in policy_meta.json (FL, FR, RL, RR x [abd, hip, knee]), radians.
    +hip pitch swings the foot BACKWARD, knee angles are negative, +abd moves a left foot outward.
    If your motor signs/zero offsets differ, convert inside RobotInterface.
"""
import json
import time
from collections import deque

import numpy as np
import onnxruntime as ort


class RobotInterface:
    def read_imu(self):
        """-> (gyro[3] rad/s, accel[3] m/s^2) in the trunk frame."""
        raise NotImplementedError

    def read_joints(self):
        """-> (q[12] rad, qd[12] rad/s) in sim joint order and sign convention."""
        raise NotImplementedError

    def send_joint_targets(self, q_target):
        """Send 12 position targets (rad). Motor drivers should run PD with kp/kd from policy_meta.json."""
        raise NotImplementedError

    def estop(self):
        """Go limp / safe state."""
        raise NotImplementedError


def main(robot: RobotInterface, vx_cmd=0.3, ramp_seconds=2.0, max_step_rad=0.15):
    meta = json.load(open("policy_meta.json"))
    sess = ort.InferenceSession("policy.onnx")
    default_q = np.array(meta["default_q"], dtype=np.float32)
    q_lo, q_hi = np.array(meta["q_lo"]), np.array(meta["q_hi"])
    hist, dt = meta["history"], 1.0 / meta["control_hz"]

    # 1) slowly move from the current pose to the standing pose
    q0, _ = robot.read_joints()
    for i in range(int(ramp_seconds / dt)):
        a = (i + 1) / int(ramp_seconds / dt)
        robot.send_joint_targets((1 - a) * q0 + a * default_q)
        time.sleep(dt)

    # 2) policy loop. Start with vx_cmd=0 (stand), then raise it gradually.
    last_action = np.zeros(12, dtype=np.float32)
    cmd = np.array([vx_cmd, 0.0, 0.0], dtype=np.float32)
    frames = deque(maxlen=hist)
    prev_target = default_q.copy()
    nxt = time.perf_counter()

    def frame():
        gyro, accel = robot.read_imu()
        parts = [gyro, accel]
        if not meta["imu_only"]:
            q, qd = robot.read_joints()
            parts += [q - default_q, qd * 0.1]
        parts += [last_action, cmd]
        return np.concatenate(parts).astype(np.float32)

    f = frame()
    for _ in range(hist):
        frames.append(f)

    try:
        while True:
            frames.append(frame())
            obs = np.concatenate(frames)[None]
            action = sess.run(None, {"obs": obs})[0][0]
            target = np.clip(default_q + meta["action_scale"] * action, q_lo, q_hi)
            target = np.clip(target, prev_target - max_step_rad, prev_target + max_step_rad)  # slew limit
            robot.send_joint_targets(target)
            prev_target, last_action = target, action.astype(np.float32)

            gyro, accel = robot.read_imu()
            if np.linalg.norm(gyro) > 15.0 or np.linalg.norm(accel) > 60.0:   # tune on the real robot
                robot.estop()
                raise RuntimeError("Safety trip: abnormal IMU reading")

            nxt += dt
            time.sleep(max(0.0, nxt - time.perf_counter()))
    except KeyboardInterrupt:
        robot.estop()
