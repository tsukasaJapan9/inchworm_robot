"""シャクトリムシ実機モデルの Gymnasium 環境。

報酬は「前進速度」＋「ひっくり返らない」だけ。歩容は PPO に任せる。

  観測 (14): 重力方向を機体座標で見たもの 3 ＋ 角速度 3 ＋ 関節角 2 ＋ 関節速度 2
             ＋ 前回の行動 2 ＋ いまの目標角 2
  行動 (2) : delta_max_deg > 0 なら「目標角の増分」、0 なら「目標角そのもの」
             増分モードでは 1 ステップの変化量が構造的に制限されるので、
             報酬でなめらかさを買わなくても滑らかな指令になる

報酬は既定では「前進速度 ＋ 正立」だけ。w_rate / w_torque を正にすると
行動の急変とトルクにペナルティがかかり、実機に載せられる滑らかな方策になる。
"""

from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from model import Params, build

CTRL_HZ = 100
STALL = 0.52  # XL330-M288-T の失速トルク [N·m]
NO_LOAD = np.deg2rad(618)  # 無負荷速度 [rad/s]
KP, KD = 2.0, 0.02  # サーボ内蔵 PD の仮定値。実機のステップ応答で要同定
Q_MAX = np.deg2rad(90.0)

# ドメインランダマイゼーションの範囲。
# 床の摩擦・関節の乾性摩擦・サーボの kp・反射慣性は実機で未確定なので、
# 手書き歩容で感度を測ったときの範囲をそのまま使う。
DR_RANGES = {
  "friction": (0.15, 0.8),  # PLA と木の床。実測で速度は ±20% 程度しか動かない
  "frictionloss": (0.016, 0.08),  # 「自重で落ちてこない」から下限 0.0161 は確定
  "kp": (1.0, 6.0),  # 最大の未知。ステップ応答を取るまで詰められない
  "armature": (0.0002, 0.002),  # 288:1 の反射慣性
  "mass_scale": (0.9, 1.1),  # 実測値はあるが、組み直しや電池の差を見込む
}
DELAY_MAX = 2  # 通信＋推論の遅延[制御ステップ]。100Hz なので 1 ステップ = 10ms

# 初期状態のランダム化。
# 物理パラメータだけ振っていると、方策が「きれいに歩く状態」しか経験せず、
# 重力方向の分散が 1.5 度ほどしかなくなる。すると VecNormalize の正規化が
# 極端に敏感になり、実機で機体が傾いた瞬間に入力がクリップに張り付いて
# 姿勢が見えなくなる（実機で平均 44.9 度傾き、正規化後 RMS 29 まで飛んだ）。
# 角速度を割る値。学習時の分散で正規化すると過敏になるので、固定値で無次元化する
GYRO_SCALE = 10.0  # rad/s

INIT_TILT_DEG = 60.0  # 初期の傾き（ロール・ピッチ）。実機は平均 44.9 度傾いていた
INIT_Q_DEG = (-90.0, 30.0)  # 初期の関節角。方策が実際に使う範囲に合わせる


class InchwormEnv(gym.Env):
  metadata = {"render_modes": [], "render_fps": CTRL_HZ}

  def __init__(
    self,
    params: Params | None = None,
    init_noise=0.1,
    w_rate=0.0,
    w_torque=0.0,
    delta_max_deg=0.0,
    w_lift=0.0,
    dr=False,
    obs_noise=0.0,
    delay_steps=0,
    init_tilt_deg=0.0,
  ):
    self.p = params or Params()
    self.model, self.data = build(self.p)
    self.frame_skip = round((1 / self.model.opt.timestep) / CTRL_HZ)
    self.init_noise = init_noise
    self.w_rate = w_rate  # 行動の急変へのペナルティ
    self.w_torque = w_torque  # トルクへのペナルティ
    # 1 制御ステップで目標角を動かせる上限。XL330 の無負荷速度 618 deg/s は
    # 100Hz 換算で 6.18 deg/step なので、それ以下にしておくと追従できる範囲に収まる
    self.delta_max = np.deg2rad(delta_max_deg)
    self.w_lift = w_lift  # 中の節を持ち上げることへの報酬。躍動的な歩容を狙う
    self.target = np.zeros(2)
    # 脚を真下に向けたときの中の節の高さの上限（実測 69mm）までを 1.0 に正規化する
    self.z_rest = self.p.mid_h / 2
    self.z_span = 0.069 - self.z_rest
    self.dr = dr
    self.init_tilt_deg = init_tilt_deg
    self.obs_noise = obs_noise
    self.delay_steps = delay_steps
    self.kp = KP
    # ランダム化の基準値を控えておく（毎回ここから作り直す。掛け続けない）
    self._nom = {
      "friction": self.model.geom_friction[:, 0].copy(),
      "frictionloss": self.model.dof_frictionloss.copy(),
      "armature": self.model.dof_armature.copy(),
      "body_mass": self.model.body_mass.copy(),
      "body_inertia": self.model.body_inertia.copy(),
    }
    self._queue = deque()
    self.prev_action = np.zeros(2, dtype=np.float32)
    # 重力方向は本来 ±1 だが、観測ノイズを足すので余裕を持たせる
    high = np.array(
      [2] * 3 + [5] * 3 + [2] * 2 + [50] * 2 + [1] * 2 + [1] * 2, dtype=np.float32
    )
    self.observation_space = spaces.Box(-high, high, dtype=np.float32)
    self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

  def _upright(self):
    """機体の上向きベクトルの z 成分。1 で正立、負でひっくり返り。"""
    return float(self.data.body("mid").xmat.reshape(3, 3)[2, 2])

  def _randomize(self):
    """reset のたびに物理パラメータを振り直す。MJCF を作り直さず配列を書き換える。"""
    m, n, rng = self.model, self._nom, self.np_random
    lo, hi = DR_RANGES["friction"]
    m.geom_friction[:, 0] = n["friction"] * 0.0 + rng.uniform(lo, hi)
    lo, hi = DR_RANGES["frictionloss"]
    m.dof_frictionloss[:] = n["frictionloss"]
    m.dof_frictionloss[6:8] = rng.uniform(lo, hi)
    lo, hi = DR_RANGES["armature"]
    m.dof_armature[:] = n["armature"]
    m.dof_armature[6:8] = rng.uniform(lo, hi)
    scale = rng.uniform(*DR_RANGES["mass_scale"])
    m.body_mass[:] = n["body_mass"] * scale
    m.body_inertia[:] = n["body_inertia"] * scale
    self.kp = rng.uniform(*DR_RANGES["kp"])
    self.delay = int(rng.integers(0, DELAY_MAX + 1))

  def _randomize_pose(self):
    """初期の傾きと関節角を振って、床の上に置き直す。"""
    rng = self.np_random
    t = np.deg2rad(self.init_tilt_deg)
    roll, pitch = rng.uniform(-t, t), rng.uniform(-t, t)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    self.data.qpos[3:7] = [cp * cr, cp * sr, sp * cr, -sp * sr]
    self.data.qpos[7:9] = np.deg2rad(rng.uniform(*INIT_Q_DEG, size=2))
    # いちばん低い geom が床に触れる高さまで下ろす
    mujoco.mj_forward(self.model, self.data)
    floor = self.model.geom("floor").id
    low = min(
      self.data.geom_xpos[g][2] - self.model.geom_rbound[g]
      for g in range(self.model.ngeom)
      if g != floor
    )
    self.data.qpos[2] -= low

  def _restore(self):
    m, n = self.model, self._nom
    m.geom_friction[:, 0] = n["friction"]
    m.dof_frictionloss[:] = n["frictionloss"]
    m.dof_armature[:] = n["armature"]
    m.body_mass[:] = n["body_mass"]
    m.body_inertia[:] = n["body_inertia"]
    self.kp = KP
    self.delay = self.delay_steps

  def _obs(self):
    d = self.data
    R = d.body("mid").xmat.reshape(3, 3)
    g_body = R.T @ np.array([0.0, 0.0, -1.0])  # 重力方向を機体座標で
    obs = np.concatenate(
      [
        g_body,
        d.sensor("gyro").data / GYRO_SCALE,  # 無次元化。正規化を通さずそのまま渡す
        d.qpos[7:9],
        d.qvel[6:8],
        self.prev_action,
        self.target / Q_MAX,
      ]
    )
    obs = obs.astype(np.float32)
    if self.obs_noise > 0:
      # IMU のノイズと関節角の量子化を模す。センサ由来の 8 つだけに効かせ、
      # 自分で作った値（前回の行動・いまの目標角）には乗せない
      noise = self.np_random.normal(0.0, self.obs_noise, size=obs.shape).astype(
        np.float32
      )
      noise[8:] = 0.0
      obs = obs + noise
    return obs

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    if self.dr:
      self._randomize()
    else:
      self._restore()
    self._queue = deque([np.zeros(2)] * self.delay, maxlen=max(self.delay, 1))
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[0:3] = [0.0, 0.0, self.p.mid_h / 2]
    self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    self.data.qpos[7:9] = self.np_random.uniform(
      -self.init_noise, self.init_noise, size=2
    )
    if self.init_tilt_deg > 0.0:
      self._randomize_pose()
    mujoco.mj_forward(self.model, self.data)
    self.prev_action = np.zeros(2, dtype=np.float32)
    self.target = self.data.qpos[7:9].copy()  # 初期ジャンプを避ける
    return self._obs(), {}

  def step(self, action):
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    prev_target = self.target.copy()
    if self.delta_max > 0:
      self.target = np.clip(self.target + a * self.delta_max, -Q_MAX, Q_MAX)
    else:
      self.target = a * Q_MAX
    # 遅延: 指令は delay ステップ後にサーボへ届く
    if self.delay > 0:
      self._queue.append(self.target.copy())
      target = self._queue[0] if len(self._queue) == self._queue.maxlen else self.target
    else:
      target = self.target
    d = self.data
    tau_sq = 0.0
    for _ in range(self.frame_skip):
      for k in range(2):
        tau = self.kp * (target[k] - d.qpos[7 + k]) - KD * d.qvel[6 + k]
        qd = d.qvel[6 + k]
        lim = STALL * max(0.0, 1 - abs(qd) / NO_LOAD) if tau * qd > 0 else STALL
        d.ctrl[k] = float(np.clip(tau, -lim, lim))
      mujoco.mj_step(self.model, d)
      tau_sq += float(np.sum(np.square(d.ctrl)))
    tau_sq /= self.frame_skip  # 1 制御ステップあたりの平均 Στ²

    vx = float(d.qvel[0])  # 前向きの速度 [m/s]
    up = self._upright()
    d_act = float(np.sum(np.square(a - self.prev_action)))
    # 実際にサーボへ送る指令がどれだけ動いたか。モードによらず比較できる指標
    d_cmd = float(np.sum(np.square((self.target - prev_target) / Q_MAX)))
    lift = float(np.clip((d.body("mid").xpos[2] - self.z_rest) / self.z_span, 0.0, 1.0))
    reward = (
      10.0 * vx
      + 0.2 * up
      + self.w_lift * lift
      - self.w_rate * d_cmd
      - self.w_torque * tau_sq
    )
    terminated = up < 0.0  # ひっくり返ったら終わり
    if terminated:
      reward -= 5.0
    self.prev_action = a
    info = {
      "vx": vx,
      "upright": up,
      "d_act": d_act,
      "d_cmd": d_cmd,
      "tau_sq": tau_sq,
      "lift": lift,
      "z_mid": float(d.body("mid").xpos[2]),
    }
    return self._obs(), float(reward), bool(terminated), False, info
