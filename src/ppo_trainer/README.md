# PPO Trainer (Self-play)

Self-play 向けに複数の連続値アルゴリズムをまとめたパッケージです。

## 特徴

- **アルゴリズム**
	- PPO (feedforward)
	- Recurrent PPO (`--algo rppo`, GRUベース)
	- Soft Actor-Critic (SAC, 連続値・オフポリシー)
	- TD3 (連続値・オフポリシー)
- **Self-play**: 1つのモデルを両エージェントに適用
- **並列環境**: 複数の環境で同時に学習
- **PyTorch実装**: GPU対応

## 構成

- `ppo.py`: PPOアルゴリズムの実装
- `recurrent_ppo.py`: GRUベースのRecurrent PPOと対応トレーナー
- `sac.py`: Soft Actor-Critic 実装
- `td3.py`: TD3 実装
- `offpolicy_trainer.py`: SAC/TD3 用 Self-play トレーナー
- `vec_env.py`: 並列環境のラッパー
- `trainer.py`: PPO 用 Self-play トレーナー

## 学習の実行

```bash
cd src
python train.py --num-envs 8 --total-steps 10000000 --algo ppo

# Recurrent PPO
python train.py --num-envs 8 --total-steps 10000000 --algo rppo

# SAC / TD3
python train.py --num-envs 8 --total-steps 10000000 --algo sac
python train.py --num-envs 8 --total-steps 10000000 --algo td3
```

### 主要オプション

- `--num-envs`: 並列環境の数（デフォルト: 4）
- `--total-steps`: 総学習ステップ数（デフォルト: 10,000,000）
- `--steps-per-update`: PPO更新ごとのステップ数（デフォルト: 2048）
- `--num-epochs`: PPOエポック数（デフォルト: 10）
- `--batch-size`: バッチサイズ（デフォルト: 64）
- `--lr`: 学習率（デフォルト: 3e-4）
- `--save-interval`: モデル保存間隔（デフォルト: 100更新ごと）
- `--log-dir`: TensorBoardログディレクトリ
- `--checkpoint-dir`: チェックポイント保存ディレクトリ
- `--algo`: `ppo` / `rppo` / `sac` / `td3`
- `--offpolicy-batch-size`, `--offpolicy-gradient-steps`: SAC/TD3 用

## TensorBoardでの監視

```bash
tensorboard --logdir logs
```

## モデルの読み込み

```python
from ppo_trainer import PPOAgent, RecurrentPPOAgent

# PPO
agent = PPOAgent(obs_dim=28, action_dim=8)
agent.load("checkpoints/ant_sumo/model_final.pt")

# Recurrent PPO
ragent = RecurrentPPOAgent(obs_dim=28, action_dim=8)
ragent.load("checkpoints/ant_sumo_rppo/model_final.pt")
```

> Note: 現状の Recurrent PPO は、推論時にはGRUの隠れ状態を時系列に渡って
> 維持しますが、学習時の勾配は各ステップを独立とみなす近似になっています。
> より厳密なシーケンス学習を行うには、シーケンス単位のBPTT対応など、
> さらなる拡張が必要です。
