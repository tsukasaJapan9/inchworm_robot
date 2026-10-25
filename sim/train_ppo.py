"""PPO でシャクトリムシの歩容を学習する。

    uv run python train_ppo.py --steps 1000000 --tag myrun   # result/myrun_s0/ に保存
"""

import argparse
import json

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from norm import MaskedVecNormalize
from result import RESULT_DIR

from env import InchwormEnv

EP_LEN = 500  # 100Hz なので 5 秒


def make_env_factory(w_rate, w_torque, delta_max_deg, w_lift, dr, obs_noise, init_tilt):
  def make_env():
    return TimeLimit(
      InchwormEnv(w_rate=w_rate, w_torque=w_torque, delta_max_deg=delta_max_deg, w_lift=w_lift,
                  dr=dr, obs_noise=obs_noise, init_tilt_deg=init_tilt),
      max_episode_steps=EP_LEN,
    )

  return make_env


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", type=int, default=1_000_000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--n-envs", type=int, default=8)
  ap.add_argument("--w-rate", type=float, default=0.0)
  ap.add_argument("--w-torque", type=float, default=0.0)
  ap.add_argument("--delta-max", type=float, default=0.0, help="目標角の増分の上限[deg/step]。0 で絶対角モード")
  ap.add_argument("--w-lift", type=float, default=0.0, help="中の節を持ち上げることへの報酬")
  ap.add_argument("--dr", action="store_true", help="ドメインランダマイゼーションを有効化")
  ap.add_argument("--obs-noise", type=float, default=0.0, help="観測ノイズの標準偏差")
  ap.add_argument("--init-tilt", type=float, default=0.0, help="初期の傾きのランダム化[deg]")
  ap.add_argument("--tag", default="inchworm")
  args = ap.parse_args()

  venv = make_vec_env(make_env_factory(args.w_rate, args.w_torque, args.delta_max, args.w_lift,
                                      args.dr, args.obs_noise, args.init_tilt), n_envs=args.n_envs,
                      seed=args.seed, vec_env_cls=SubprocVecEnv)
  venv = MaskedVecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

  model = PPO(
    "MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
    n_steps=512, batch_size=2048, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
    ent_coef=0.0, tensorboard_log="../tb_logs",
  )
  name = f"{args.tag}_s{args.seed}"
  model.learn(total_timesteps=args.steps, tb_log_name=name)
  out = RESULT_DIR / name
  out.mkdir(parents=True, exist_ok=True)
  model.save(out / "model.zip")
  venv.save(str(out / "vecnorm.pkl"))
  # 評価時に同じ環境設定で再現できるように残す（これが無いと増分モードの方策を
  # 絶対角モードで評価してしまい、まったく違う挙動になる）
  json.dump(
    {"w_rate": args.w_rate, "w_torque": args.w_torque, "delta_max_deg": args.delta_max,
     "w_lift": args.w_lift, "dr": args.dr, "obs_noise": args.obs_noise,
     "init_tilt_deg": args.init_tilt},
    open(out / "cfg.json", "w"),
  )
  print(f"saved {out}")
