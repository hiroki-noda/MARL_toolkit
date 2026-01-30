# 強化学習プロジェクト - Self-Play

MuJoCoを使用した2体のAntエージェントによる相撲タスクの強化学習環境とPPOによるSelf-play学習

## プロジェクト構成

```
reinforcement-learning/
├── docker/
│   ├── Dockerfile          # Ubuntu 24.04 + CUDA + MuJoCo + PyTorch + ROS2
# 強化学習プロジェクト - Self-Play マルチエージェント RL

MuJoCo ベースの Ant Sumo から、2～4 体のマルチエージェント環境（タグ、協調ナビゲーション、行列ゲーム）までを対象とした、
Self-Play 強化学習フレームワークです。

- 複数環境を `--env-id` で切り替え
- PPO / SAC / TD3 とその再帰版 (rPPO / rSAC / rTD3)
- 4 体タグ環境用の「役割予測付き」再帰モデル (rtd3_role / rsac_role / rppo_role)
- TensorBoard ログ (うち role_loss も可視化)
- 学習済みモデルの対戦・可視化スクリプト

---

## ディレクトリ構成（抜粋）

```text
reinforcement-learning/
├── docker/
│   ├── Dockerfile          # Ubuntu + CUDA + MuJoCo + PyTorch 環境
│   ├── build.sh            # Docker イメージビルド
│   └── run.sh              # コンテナ起動
├── src/
│   ├── env_factory.py      # env_id から環境を生成するヘルパ
│   ├── train.py            # Self-Play 学習メインスクリプト
│   ├── test_policy.py      # 学習済みモデルのテスト・可視化
│   ├── sumo_env/
│   │   ├── ant_sumo_env.py # Ant Sumo 2 体相撲環境
│   │   └── README.md
│   ├── ma_envs/            # マルチエージェント環境
│   │   ├── adversarial_tag_env.py  # 2 体 / 4 体タグ環境
│   │   ├── coop_navigation_env.py  # 協調ナビゲーション
│   │   ├── matrix_games.py         # IPD / Matching Pennies / Coordination
│   │   └── README.md (任意)
│   └── ppo_trainer/        # 強化学習アルゴリズム & トレーナー
│       ├── ppo.py, sac.py, td3.py
│       ├── recurrent_ppo.py, recurrent_sac.py, recurrent_td3.py
│       ├── role_rtd3.py, role_rsac.py, role_rppo.py
│       ├── trainer.py, offpolicy_trainer.py
│       ├── sequence_replay.py, vec_env.py
│       └── README.md
├── run_tensorboard.sh      # TensorBoard 起動用ヘルパ
├── checkpoints/            # モデルチェックポイント（自動生成）
└── logs/                   # TensorBoard ログ（自動生成）
```

---

## 利用可能な環境 (env_id)

`src/env_factory.py` で定義されている環境 ID:

- `ant_sumo` : MuJoCo ベースの 2 体 Ant 相撲環境
- `ipd` : Iterated Prisoners Dilemma（繰り返し囚人のジレンマ）
- `matching_pennies` : Matching Pennies 行列ゲーム
- `coordination` : Continuous Coordination ゲーム
- `coop_nav` : 2D 協調ナビゲーション環境
- `adversarial_tag` : 2 体追跡 vs 逃走タグ環境
- `adversarial_tag_4p` : 4 体タグ環境（追跡者 2 体 vs 逃走者 2 体）

いずれも Self-Play 前提で「1 エージェント = 1 物理プレイヤー」の設定になっています。

---

## 利用可能なアルゴリズム (algo)

`src/train.py` の `--algo` で選択できます。

- On-policy
  - `ppo` : 通常の PPO（2 体 / 4 体両方対応）
  - `rppo` : GRU ベースの Recurrent PPO
  - `rppo_role` : 4 体タグ専用の「役割予測付き」Recurrent PPO
- Off-policy（フィードフォワード）
  - `sac` : Soft Actor-Critic
  - `td3` : Twin Delayed DDPG
- Off-policy（再帰型）
  - `rsac` : Recurrent SAC
  - `rtd3` : Recurrent TD3
- Off-policy（役割予測付き / 4 体タグ専用）
  - `rtd3_role` : 役割予測付き Recurrent TD3
  - `rsac_role` : 役割予測付き Recurrent SAC

`*_role` 系は `env_id=adversarial_tag_4p` でのみ利用可能で、
他の環境で指定すると明示的にエラーを出すようになっています。

---

## セットアップ

### 1. Docker イメージのビルド

```bash
cd docker
./build.sh
```

### 2. コンテナの起動

```bash
./run.sh
```

コンテナ内では `/workspace/reinforcement-learning` が本リポジトリです。

---

## 学習の実行

基本形:

```bash
cd /workspace/reinforcement-learning
python src/train.py \
  --env-id ant_sumo \
  --algo ppo \
  --num-envs 8 \
  --total-steps 10000000 \
  --use-gpu
```

主なオプション:

- `--env-id` : 上記の環境 ID から選択
- `--algo` : 上記のアルゴリズム ID から選択
- `--num-envs` : 並列環境数
- `--total-steps` : 総ステップ数 (`num_envs * steps_per_update` 単位で丸められます)
- `--steps-per-update` : 1 回の更新あたりのステップ数
- `--num-epochs` / `--batch-size` : PPO / Recurrent PPO の学習ハイパラ
- `--offpolicy-batch-size` / `--offpolicy-gradient-steps` : SAC / TD3 系のバッチ・更新回数
- `--offpolicy-max-seq-len` : 再帰型 off-policy (rsac / rtd3 / *_role) の BPTT シーケンス長

チェックポイントは `checkpoints/<env_id>_algo/`（PPO は `checkpoints/<env_id>/`）に保存されます。

例: 4 体タグ環境 + 役割予測付き rTD3

```bash
python src/train.py \
  --env-id adversarial_tag_4p \
  --algo rtd3_role \
  --num-envs 8 \
  --steps-per-update 2048 \
  --offpolicy-batch-size 256 \
  --offpolicy-max-seq-len 64 \
  --use-gpu
```

---

## 学習済みモデルのテスト・可視化

`src/test_policy.py` で、チェックポイントをロードして対戦・可視化できます。

```bash
python src/test_policy.py \
  --env-id adversarial_tag_4p \
  --algo rtd3_role \
  --num-episodes 10
```

- `--checkpoint` を省略すると、`checkpoints/<env_id>_algo/`（または旧形式）から最新の checkpoint を自動検出します。
- 4 体タグ環境 + `*_role` のときは、レンダリング画像内に
  - 各エージェントの「他 3 体の役割予測（chaser である確率）」
  - 実際の役割（色・ラベル）
  が同時に表示されます。
- MuJoCo ベースの AntSumoEnv の場合は MuJoCo ビューアでの 3D 可視化、それ以外は Matplotlib を使った 2D 可視化です。


---

## TensorBoard での監視

学習中の損失や勝率などは `logs/` 以下に書き出されます。

### 起動ヘルパスクリプト

```bash
./run_tensorboard.sh            # デフォルト: logdir=logs, port=6006
# あるいは
PORT=7007 ./run_tensorboard.sh logs
```

ブラウザで `http://localhost:6006`（または指定ポート）にアクセスしてください。

主なスカラー:

- `train/policy_loss`, `train/value_loss`, `train/entropy`
- `train/q1_loss`, `train/q2_loss`, `train/critic_loss`, `train/alpha`
- `train/avg_reward`
- `train/win_rate_agent1`, `train/win_rate_agent2`, `train/draw_rate`
- `train/avg_episode_return_agent1`, `train/avg_episode_return_agent2`
- `train/role_loss` : 役割予測付きモデル (rtd3_role / rsac_role / rppo_role) の教師ありロス

---

## 依存ライブラリ（主なもの）

- Python 3.12
- PyTorch + CUDA
- MuJoCo
- Gymnasium
- NumPy
- Matplotlib
- TensorBoard

Dockerfile では上記を含む環境が構築される想定です。

---

## ライセンス

MIT
