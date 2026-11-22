"""
Train / watch / stress-test the quadruped.

  python train.py check                      # smoke test: obs shape, holds stance with zero action
  python train.py train --steps 10_000_000   # PPO training (all CPU cores, capped at 16)
  python train.py eval --vx 0.5              # watch the trained policy in the viewer (mjpython on macOS)
  python train.py stress                     # 50 randomised episodes: survival rate + speed tracking
  python train.py record --vx 0.5            # save walk.mp4

  --imu_only   remove joint encoders from the observations
  --no_random  disable domain randomisation (debugging only; do NOT train a real-robot policy this way)
  tensorboard --logdir runs
"""
import argparse
import os

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from quadruped_env import QuadrupedEnv

RUN_DIR = "runs"
MODEL_PATH = os.path.join(RUN_DIR, "ppo_quadruped.zip")
STATS_PATH = os.path.join(RUN_DIR, "vecnormalize.pkl")


def train(args):
    os.makedirs(RUN_DIR, exist_ok=True)
    n_envs = args.n_envs or min(os.cpu_count() or 4, 16)
    env = make_vec_env(QuadrupedEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                       env_kwargs=dict(imu_only=args.imu_only, randomize=not args.no_random))
    if args.resume and os.path.exists(MODEL_PATH):
        env = VecNormalize.load(STATS_PATH, env)
        model = PPO.load(MODEL_PATH, env=env, tensorboard_log=RUN_DIR)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=RUN_DIR,
            n_steps=256, batch_size=1024, n_epochs=5,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                               activation_fn=nn.ELU, log_std_init=-1.0),
        )
    ckpt = CheckpointCallback(save_freq=max(500_000 // n_envs, 1), save_path=RUN_DIR,
                              name_prefix="ckpt", save_vecnormalize=True)
    model.learn(total_timesteps=args.steps, callback=ckpt, reset_num_timesteps=not args.resume)
    model.save(MODEL_PATH)
    env.save(STATS_PATH)
    print(f"Saved {MODEL_PATH} and {STATS_PATH}")


def load_policy(args, render_mode):
    kw = dict(imu_only=args.imu_only, render_mode=render_mode, randomize=not args.no_random,
              cmd_range=(args.vx, args.vx), stand_prob=0.0)
    venv = DummyVecEnv([lambda: QuadrupedEnv(**kw)])
    venv = VecNormalize.load(STATS_PATH, venv)
    venv.training, venv.norm_reward = False, False
    return PPO.load(MODEL_PATH, env=venv), venv


def evaluate(args):
    model, venv = load_policy(args, "human")
    obs = venv.reset()
    ep_ret, ep = 0.0, 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = venv.step(action)
        ep_ret += float(rew[0])
        if done[0]:
            ep += 1
            print(f"episode {ep}: return {ep_ret:.1f}  (cmd {args.vx} m/s)")
            ep_ret = 0.0


def stress(args):
    """Run many randomised episodes; the survival rate is your best pre-hardware go/no-go number."""
    model, venv = load_policy(args, None)
    obs = venv.reset()
    survived, errs, n_ep = 0, [], 0
    while n_ep < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(action)
        errs.append(abs(info[0]["vx"] - info[0]["cmd_vx"]))
        if done[0]:
            n_ep += 1
            survived += int(info[0].get("TimeLimit.truncated", False))
    print(f"survived full episode: {survived}/{n_ep} ({100 * survived / n_ep:.0f}%)")
    print(f"mean |vx - cmd|: {np.mean(errs):.3f} m/s at cmd {args.vx} m/s")
    print("Aim for >95% survival before trying hardware.")


def record(args):
    import imageio
    model, venv = load_policy(args, "rgb_array")
    obs = venv.reset()
    frames = []
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(action)
        frames.append(venv.venv.envs[0].unwrapped.render())
    imageio.mimsave("walk.mp4", frames, fps=50)
    print("Wrote walk.mp4")


def check(args):
    env = QuadrupedEnv(imu_only=args.imu_only, randomize=not args.no_random)
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape, "| action:", env.action_space.shape, "| policy dt:", env.dt,
          "| stand height: %.3f m" % env.stand_height, "| default pose:", np.round(env.default_leg, 3))
    ret, n = 0.0, 0
    for _ in range(300):
        obs, r, term, trunc, _ = env.step(np.zeros(12))   # zero action = hold default stance
        ret += r; n += 1
        if term or trunc:
            break
    print(f"holding default pose: survived {n}/300 steps, return {ret:.1f}")
    print("IMU at rest (gyro, accel):", np.round(obs[-env.frame_dim:][:6], 2), "(accel z should be ~ +9.8)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "train", "eval", "stress", "record"])
    p.add_argument("--steps", type=lambda s: int(s.replace("_", "")), default=10_000_000)
    p.add_argument("--n_envs", type=int, default=0)
    p.add_argument("--vx", type=float, default=0.5, help="forward speed command for eval/stress/record")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--imu_only", action="store_true")
    p.add_argument("--no_random", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    {"check": check, "train": train, "eval": evaluate, "stress": stress, "record": record}[a.mode](a)
