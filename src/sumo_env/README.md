# Ant Sumo Environment

2体のAntエージェントが相撲で対戦するMuJoCo強化学習環境

## 特徴

- **タスク**: 円形のアリーナ内で相手をひっくり返すか、円の外に押し出す
- **エージェント**: 2体のAnt（各8自由度）
- **報酬**:
  - 勝利: +100
  - 敗北: -100
  - 時間経過: -0.01/step

## 観測空間（28次元）

- 自分の位置 (x, y): 2次元
- 自分の関節角度: 8次元
- 自分の関節トルク: 8次元
- 相手の位置 (x, y): 2次元
- 相手の関節角度: 8次元

## 行動空間

- 8つの関節トルク（-1.0 ~ 1.0）

## 使い方

```python
from sumo_env import AntSumoEnv

env = AntSumoEnv(arena_radius=5.0, max_episode_steps=1000)
obs, info = env.reset()

# 両エージェントの行動を辞書形式で渡す
action = {
    "agent1": np.random.uniform(-1, 1, 8),
    "agent2": np.random.uniform(-1, 1, 8)
}
obs, reward, terminated, truncated, info = env.step(action)
```
