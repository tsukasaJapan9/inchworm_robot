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
  play_ppo.py        学習した方策を MuJoCo のビューアで再生する
  export_policy.py   学習した方策を C ヘッダに書き出す
  result.py          学習結果の置き場所と読み込み
  result/            学習結果（git 管理外）
firmware/            M5Stack Core2 のファームウェア（PlatformIO）
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

CPU で 10 分ほどです。

| オプション | 意味 |
|---|---|
| `--delta-max 5` | 行動を目標角の増分として扱い、1 ステップ 5 度までにする |
| `--dr` | ドメインランダマイゼーション。摩擦・関節摩擦・kp・armature・質量・遅延を振る |
| `--obs-noise 0.02` | 観測にノイズを足す |
| `--init-tilt 60` | 初期姿勢を ±60 度ランダムにする |
| `--seed` | 乱数シード（既定 0） |

シードを変えると結果がかなり変わります。複数シードで比べてください。

学習が終わると `sim/result/<tag>_s<seed>/`（この例では `sim/result/myrun_s0/`）に次の 3 つができます。
この 3 つが揃って 1 つの方策です。以降のスクリプトにはこのディレクトリ名（`myrun_s0`）を渡します。

| ファイル | 中身 |
|---|---|
| `model.zip` | 方策の重み |
| `vecnorm.pkl` | 観測の正規化パラメータ |
| `cfg.json` | 学習時の環境設定 |

学習曲線はリポジトリ直下の `tb_logs/` に出ます（直下で `uv run tensorboard --logdir tb_logs`）。

## シミュレーションで再生する

```sh
cd sim
uv run python play_ppo.py myrun_s0                    # 学習時と同じ環境で再生
uv run python play_ppo.py myrun_s0 --dr --seconds 20  # 物理パラメータを振って再生
```

転倒するか `--seconds` 経つとリセットして繰り返し、エピソードごとの前進距離をターミナルに出します。

## 実機に載せる

```sh
cd sim
uv run python export_policy.py myrun_s0    # firmware/include/policy_weights.h を生成
cd ../firmware
pio run -t upload
```

### いちばん良かった方策を使う

`firmware/include/policy_weights_best.h` は、これまでで一番良かった方策（`ppo_raw_s0`）を書き出したものです。
ファームウェアが読むのは `policy_weights.h` なので、使うときはこの名前にして書き込みます。

```sh
cd firmware
cp include/policy_weights_best.h include/policy_weights.h
pio run -t upload
```

### 実機の操作方法

| ボタン | 動作 |
|---|---|
| A | 方策を走らせる |
| B | 停止 |
| C | ゼロ点キャリブレーション |

走らせる前にゼロ点のキャリブレーションが要ります。前後の節を手で水平にして C ボタンを押すと、
そのときのサーボ角度を 16 回読んで平均し、不揮発メモリに保存します。校正前は A ボタンを押しても走りません。
