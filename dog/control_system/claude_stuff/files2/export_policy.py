"""
Export the trained policy to ONNX with observation normalisation baked in.

  python export_policy.py            ->  policy.onnx + policy_meta.json

policy_meta.json holds everything the on-robot code needs (observation layout, joint order,
default pose, action scale, gains, limits) so sim and robot cannot silently drift apart.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quadruped_env import JOINT_NAMES, QuadrupedEnv
from train import MODEL_PATH, STATS_PATH


class DeployPolicy(nn.Module):
    def __init__(self, policy, mean, var, eps, clip):
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.sqrt(var + eps), dtype=torch.float32))
        self.clip = clip

    def forward(self, obs):
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        latent = self.policy.mlp_extractor.forward_actor(x)
        return torch.clamp(self.policy.action_net(latent), -1.0, 1.0)   # deterministic mean action


def main(args):
    env = QuadrupedEnv(imu_only=args.imu_only)
    vn = VecNormalize.load(STATS_PATH, DummyVecEnv([lambda: env]))
    model = PPO.load(MODEL_PATH, device="cpu")
    net = DeployPolicy(model.policy, vn.obs_rms.mean, vn.obs_rms.var, vn.epsilon, vn.clip_obs).eval()

    dummy = torch.zeros(1, env.observation_space.shape[0])
    torch.onnx.export(net, dummy, "policy.onnx", input_names=["obs"], output_names=["action"], opset_version=17)

    cfg = env.cfg
    meta = dict(
        obs_dim=int(env.observation_space.shape[0]), history=env.history, frame_dim=env.frame_dim,
        imu_only=env.imu_only,
        frame_layout=["gyro[3] rad/s", "accel[3] m/s^2"] + ([] if env.imu_only else
                      ["q - default_q[12] rad", "qd*0.1[12] rad/s"]) + ["last_action[12]", "cmd[vx,vy,wz]"],
        history_order="oldest first; at start, fill all history slots with the first frame",
        joint_order=JOINT_NAMES, default_q=env.default_q.tolist(), action_scale=env.action_scale,
        q_lo=env.q_lo.tolist(), q_hi=env.q_hi.tolist(), control_hz=cfg.control_hz,
        kp=cfg.kp, kd=cfg.kd, torque_limit=cfg.torque_limit, max_joint_speed=cfg.max_joint_speed,
        target_rule="q_target = clip(default_q + action_scale * action, q_lo, q_hi)",
    )
    json.dump(meta, open("policy_meta.json", "w"), indent=2)

    try:  # parity check against the SB3 policy
        import onnxruntime as ort
        sess = ort.InferenceSession("policy.onnx")
        x = np.random.randn(8, dummy.shape[1]).astype(np.float32)
        ref = model.predict(np.clip((x - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                                    -vn.clip_obs, vn.clip_obs), deterministic=True)[0]
        out = sess.run(None, {"obs": x})[0]
        print("max |onnx - sb3| =", float(np.abs(np.clip(ref, -1, 1) - out).max()))
    except ImportError:
        print("pip install onnxruntime to verify the export")
    print("Wrote policy.onnx and policy_meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--imu_only", action="store_true")
    main(p.parse_args())
