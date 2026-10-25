"""学習した方策を MuJoCo のビューアで実時間で再生する。

    uv run python play_ppo.py ppo_myrun_s0
    uv run python play_ppo.py ppo_myrun_s0 --dr --seconds 20

ウィンドウを閉じるか Ctrl-C で終わる。転覆するか --seconds 経つとリセットして繰り返す。
"""

import argparse
import json
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import CTRL_HZ, InchwormEnv
from norm import MaskedVecNormalize


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("tag", help="例: ppo_myrun_s0")
  ap.add_argument("--seconds", type=float, default=10.0, help="1 エピソードの長さ")
  ap.add_argument("--dr", action="store_true", help="ドメインランダマイゼーションをかけて再生する")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  # 学習時の環境設定を復元する
  path = f"cfg_{args.tag.replace('ppo_', '')}.json"
  cfg = json.load(open(path)) if os.path.exists(path) else {}
  env = InchwormEnv(delta_max_deg=cfg.get("delta_max_deg", 0.0), w_lift=cfg.get("w_lift", 0.0),
                    init_tilt_deg=cfg.get("init_tilt_deg", 0.0), dr=args.dr)

  model = PPO.load(args.tag, device="cpu")
  # 正規化パラメータだけ使う。ステップは VecEnv を通さず env を直接回す
  # （VecEnv の自動リセットを挟むとビューアと data の対応が崩れるため）
  dummy = DummyVecEnv([lambda: TimeLimit(InchwormEnv(), max_episode_steps=10)])
  vn = MaskedVecNormalize.load(args.tag.replace("ppo_", "vecnorm_") + ".pkl", dummy)
  vn.training = False

  steps = int(args.seconds * CTRL_HZ)
  dt = 1.0 / CTRL_HZ
  with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = env.model.body("mid").id
    viewer.cam.distance = 0.6
    ep = 0
    while viewer.is_running():
      obs, _ = env.reset(seed=args.seed + ep)
      x0 = float(env.data.qpos[0])
      flipped = False
      for _ in range(steps):
        t0 = time.perf_counter()
        act, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, _, terminated, _, _ = env.step(act[0])
        viewer.sync()
        if terminated:
          flipped = True
          break
        if not viewer.is_running():
          return
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
      dx = float(env.data.qpos[0]) - x0
      print(f"ep{ep}: 前進 {100 * dx:6.1f} cm{'  ⚠転覆' if flipped else ''}")
      ep += 1


if __name__ == "__main__":
  main()
