"""学習した方策を M5Stack 用の C ヘッダに書き出す。

    uv run python export_policy.py ppo_dr_s2 ../firmware/include/policy_weights.h

出力するもの:
  - MLP の重み（14 -> Tanh(64) -> Tanh(64) -> 2）
  - VecNormalize の平均と標準偏差（**これを忘れると方策が別物になる**）
  - 学習時の環境設定（増分の上限など）
"""

import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO


def c_array(name, arr):
  """float32 の 1 次元配列を C のリテラルにする。"""
  flat = np.asarray(arr, dtype=np.float32).ravel()
  body = ",\n  ".join(
    ", ".join(f"{v:+.8e}f" for v in flat[i : i + 6]) for i in range(0, len(flat), 6)
  )
  return f"static const float {name}[{len(flat)}] = {{\n  {body}\n}};\n"


def main(tag, out_path):
  model = PPO.load(tag, device="cpu")
  sd = {k: v.numpy() for k, v in model.policy.state_dict().items()}

  # VecNormalize は pickle なので SB3 経由で読む
  from stable_baselines3.common.vec_env import DummyVecEnv

  from norm import RAW_SLICE, MaskedVecNormalize
  from gymnasium.wrappers import TimeLimit
  from env import InchwormEnv

  cfg = json.load(open(f"cfg_{tag.replace('ppo_', '')}.json"))
  venv = DummyVecEnv([lambda: TimeLimit(InchwormEnv(), max_episode_steps=10)])
  vn = MaskedVecNormalize.load(tag.replace("ppo_", "vecnorm_") + ".pkl", venv)
  mean = vn.obs_rms.mean.astype(np.float32)
  std = np.sqrt(vn.obs_rms.var + vn.epsilon).astype(np.float32)
  # 正規化から外している次元（重力方向と角速度）は、ファーム側が一律に
  # (obs - mean) / std を掛けるので、素通りになる値を書いておく。
  # ここを忘れると実機だけ観測が別物になる
  mean[RAW_SLICE] = 0.0
  std[RAW_SLICE] = 1.0

  # PyTorch の Linear は (out, in) 並び。C 側では行優先でそのまま使う
  w0, b0 = sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"]
  w1, b1 = sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"]
  w2, b2 = sd["action_net.weight"], sd["action_net.bias"]
  n_in, n_h, n_out = w0.shape[1], w0.shape[0], w2.shape[0]

  with open(out_path, "w") as f:
    f.write(f"""// {tag} から自動生成。手で編集しないこと。
// 生成: export_policy.py
//
// 方策: {n_in} -> Tanh({n_h}) -> Tanh({n_h}) -> {n_out}（決定論的に平均を使う）
// 観測の並び:
//   [0:3]   重力方向（機体座標・単位ベクトル）
//   [3:6]   角速度 [rad/s]（機体座標）
//   [6:8]   関節角 q1, q2 [rad]（水平が 0、持ち上げる向きが正）
//   [8:10]  関節角速度 [rad/s]
//   [10:12] 前回の行動（[-1,1]）
//   [12:14] いまの目標角 / {np.degrees(np.pi / 2):.0f}deg
// 行動: 目標角の増分。a * {cfg["delta_max_deg"]:.1f}deg を毎制御ステップ積分する

#pragma once

#define POLICY_N_IN {n_in}
#define POLICY_N_HIDDEN {n_h}
#define POLICY_N_OUT {n_out}
#define POLICY_DELTA_MAX_DEG {cfg["delta_max_deg"]:.4f}f
#define POLICY_Q_MAX_DEG 90.0f
#define POLICY_CLIP_OBS {vn.clip_obs:.1f}f
#define POLICY_CTRL_HZ 100

""")
    f.write(c_array("kObsMean", mean))
    f.write("\n")
    f.write(c_array("kObsStd", std))
    f.write("\n")
    f.write(c_array("kW0", w0))
    f.write("\n")
    f.write(c_array("kB0", b0))
    f.write("\n")
    f.write(c_array("kW1", w1))
    f.write("\n")
    f.write(c_array("kB1", b1))
    f.write("\n")
    f.write(c_array("kW2", w2))
    f.write("\n")
    f.write(c_array("kB2", b2))

  n_param = sum(a.size for a in (w0, b0, w1, b1, w2, b2)) + mean.size + std.size
  print(f"wrote {out_path}")
  print(f"  パラメータ {n_param} 個 / {n_param * 4 / 1024:.1f} KB (float32)")
  print(f"  増分の上限 {cfg['delta_max_deg']}deg/step @ {int(1 / 0.01)}Hz")

  # C 実装の検算用に、既知の入出力ペアを 1 つ出しておく。
  # 乱数だと分布外に出て行動が ±1 に飽和し、バグがあっても通ってしまうので、
  # 実際のロールアウトから「飽和していない」観測を拾う。
  env = InchwormEnv(delta_max_deg=cfg["delta_max_deg"],
                    init_tilt_deg=cfg.get("init_tilt_deg", 0.0))
  best, best_sat = None, 1e9
  for ep in range(6):  # 飽和しない観測が見つかるまでエピソードを変えて探す
    o, _ = env.reset(seed=ep)
    for _ in range(300):
      z_ = np.clip((o - mean) / std, -vn.clip_obs, vn.clip_obs)
      h_ = np.tanh(w1 @ np.tanh(w0 @ z_ + b0) + b1)
      a_raw = w2 @ h_ + b2
      a_ = np.clip(a_raw, -1, 1)
      sat = float(np.max(np.abs(a_raw)))  # 小さいほど飽和から遠い
      if sat < best_sat:
        best_sat, best = sat, o.copy()
      o, _, te, tr, _ = env.step(a_.astype(np.float32))
      if te or tr:
        break
    if best_sat < 0.9:
      break
  obs = best
  if best_sat > 0.95:
    print(f"  ※ 飽和しない観測が見つかりませんでした（最小 |a| = {best_sat:.3f}）。"
          "検証としては弱いので注意")
  z = np.clip((obs - mean) / std, -vn.clip_obs, vn.clip_obs)
  h = np.tanh(w0 @ z + b0)
  h = np.tanh(w1 @ h + b1)
  act = w2 @ h + b2  # クリップ前の生の出力。ファーム側もこれと突き合わせる
  print("\n  === C 実装の検算用テストベクタ（クリップ前の生出力）===")
  print("  obs =", ", ".join(f"{v:+.6f}f" for v in obs))
  print("  期待される行動 =", ", ".join(f"{v:+.6f}" for v in act))


if __name__ == "__main__":
  tag = sys.argv[1] if len(sys.argv) > 1 else "ppo_dr_s2"
  out = sys.argv[2] if len(sys.argv) > 2 else "../firmware/include/policy_weights.h"
  os.makedirs(os.path.dirname(out), exist_ok=True)
  main(tag, out)
