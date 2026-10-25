"""重力方向と角速度を VecNormalize の対象から外すラッパ。

学習時の分散で割る正規化は、**分散が小さい観測ほど過敏になる**。
シミュで方策がうまく歩くほど機体は水平に保たれ、重力方向の分散は
1.5 度ほどまで縮む。するとその狭い分散で正規化されるので、実機で
わずかに傾いただけで入力が ±10 のクリップに張り付き、方策は姿勢が
見えなくなる（実機で正規化後 RMS 29 まで飛んだ）。

重力方向は定義上 -1〜+1 の単位ベクトル、角速度も env 側で 10 rad/s で
割って無次元化してあるので、どちらも素のまま渡せばよい。
"""

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

RAW_SLICE = slice(0, 6)  # [0:3] 重力方向 / [3:6] 角速度（env 側で 1/10 済み）


class MaskedVecNormalize(VecNormalize):
  def normalize_obs(self, obs):
    out = super().normalize_obs(obs)
    if isinstance(out, np.ndarray) and isinstance(obs, np.ndarray):
      out = np.array(out, copy=True)
      out[..., RAW_SLICE] = obs[..., RAW_SLICE]
    return out
