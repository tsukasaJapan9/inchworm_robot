# inchworm_robot

サーボ 2 個のシャクトリムシ型ロボットを MuJoCo でモデル化し、PPO で歩き方を学習させて、
M5Stack Core2 に載せて実機を歩かせるまでのコードです。

## 構成

```
sim/
  model.py           実機の実測値から MuJoCo モデルを組み立てる
  env.py             Gymnasium 環境（サーボモデル・ドメインランダマイゼーション込み）
  norm.py            重力方向と角速度を正規化の対象から外す VecNormalize
  train_ppo.py       PPO で学習する
  export_policy.py   学習した方策を C ヘッダに書き出す
firmware/            M5Stack Core2 のファームウェア（PlatformIO）
model/inchworm.xml   MJCF（参考用。sim/ からは読んでいない）
```

## セットアップ

Python パッケージはuvで管理しています。

```sh
uv sync
```

## 学習する

```sh
cd sim
uv run python train_ppo.py --steps 2000000 --delta-max 5 --dr --obs-noise 0.02 --init-tilt 60 --tag myrun
```

CPU で 10 分ほどです。GPU は使いません（計算のほとんどが MuJoCo の物理計算で CPU 側にあるため）。

| オプション | 意味 |
|---|---|
| `--delta-max 5` | 行動を目標角の増分として扱い、1 ステップ 5 度までにする |
| `--dr` | ドメインランダマイゼーション。摩擦・関節摩擦・kp・armature・質量・遅延を振る |
| `--obs-noise 0.02` | 観測にノイズを足す |
| `--init-tilt 60` | 初期姿勢を ±60 度ランダムにする |
| `--seed` | 乱数シード（既定 0） |

シードを変えると結果がかなり変わります。複数シードで比べてください。

学習が終わると `sim/` に次の 3 つができます。この 3 つが揃って 1 つの方策です。

| ファイル | 中身 |
|---|---|
| `ppo_myrun_s0.zip` | 方策の重み |
| `vecnorm_myrun_s0.pkl` | 観測の正規化パラメータ |
| `cfg_myrun_s0.json` | 学習時の環境設定 |

学習曲線はリポジトリ直下の `tb_logs/` に出ます（直下で `uv run tensorboard --logdir tb_logs`）。

## 実機に載せる

```sh
cd sim
uv run python export_policy.py ppo_myrun_s0    # firmware/include/policy_weights.h を生成
cd ../firmware
pio run -t upload
```

重みだけでなく観測の正規化パラメータも同じヘッダに書き出されます。どちらが欠けても方策は動きません。

| ボタン | 動作 |
|---|---|
| A | 方策を走らせる |
| B | 停止（長押しでステップ応答の測定） |
| C | ゼロ点キャリブレーション |

走らせる前にゼロ点のキャリブレーションが要ります。立ち上げ手順と実機合わせの詳細は
[firmware/README.md](firmware/README.md) にあります。
