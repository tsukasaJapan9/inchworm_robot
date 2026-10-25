"""学習結果の置き場所と読み込み。

1 回の学習の結果は result/<tag>_s<seed>/ にまとめて置く。

  model.zip     方策の重み
  vecnorm.pkl   観測の正規化パラメータ
  cfg.json      学習時の環境設定

この 3 つが揃って 1 つの方策になる。どれが欠けても再現できない。
"""

import json
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "result"


def run_dir(name):
  """`myrun_s0` のような名前から結果のディレクトリを返す。ディレクトリのパスを直接渡してもよい。"""
  p = Path(name)
  return p if p.is_dir() else RESULT_DIR / name


def load(name):
  """方策・正規化・環境設定をまとめて読む。"""
  from env import InchwormEnv
  from gymnasium.wrappers import TimeLimit
  from norm import MaskedVecNormalize
  from stable_baselines3 import PPO
  from stable_baselines3.common.vec_env import DummyVecEnv

  d = run_dir(name)
  if not d.is_dir():
    raise SystemExit(f"{d} がありません")
  model = PPO.load(d / "model.zip", device="cpu")
  # VecNormalize は pickle なので SB3 経由で読む。中の環境は正規化に関係しない
  venv = DummyVecEnv([lambda: TimeLimit(InchwormEnv(), max_episode_steps=10)])
  vn = MaskedVecNormalize.load(str(d / "vecnorm.pkl"), venv)
  vn.training = False
  vn.norm_reward = False
  cfg = json.loads((d / "cfg.json").read_text())
  return model, vn, cfg
