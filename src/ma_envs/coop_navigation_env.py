import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib

# ヘッドレス環境では Qt などのGUIバックエンドを使わず、Agg に固定する
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio

class CooperativeNavigationEnv(gym.Env):
    """2エージェントの協調ナビゲーション環境

    シンプルな 2D 連続空間で、両エージェントが原点 (0, 0) に
    近づくことを目指す協調タスク。

    - 観測 (各エージェント 5次元):
        [self_x, self_y, other_x, other_y, dist_to_origin]
    - 行動: 2次元連続ベクトル (vx, vy) in [-1, 1]
      ただし1ステップの移動量は小さくクリップする。
    - 報酬: 各エージェントごとに
        r_i = -0.1 * dist_to_origin_i - collision_penalty
      （原点に近づくほど高報酬、互いに接近し過ぎるとペナルティ）
    - エピソード長: max_episode_steps ステップ
      途中終了条件は設けず、時間切れで truncated=True になる。

    step の返り値・info 形式は他の環境と揃える:
        obs(agent1), reward1, terminated, truncated, info
    info には "agent1_reward", "agent2_reward", "winner" を含める。
    "winner" はエピソード終了時に、原点に近い方を勝者とする。
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, max_episode_steps: int = 50, world_radius: float = 2.0):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.world_radius = world_radius
        self.current_step = 0

        # 各エージェントの位置 (x, y)
        self.agent1_pos = np.zeros(2, dtype=np.float32)
        self.agent2_pos = np.zeros(2, dtype=np.float32)

        # 2次元連続アクション
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 観測: [self_x, self_y, other_x, other_y, dist_to_origin]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        # レンダリング用（ヘッドレス: 後でGIFにまとめる）
        self._fig = None
        self._ax = None
        self._render_dir = "render_frames_coop_nav"
        self._frames = []  # list[np.ndarray]

    # -----------------
    # ヘルパ
    # -----------------

    def _clip_position(self, pos: np.ndarray) -> np.ndarray:
        """ワールド内に位置をクリップ"""
        r = np.linalg.norm(pos)
        if r > self.world_radius:
            pos = pos * (self.world_radius / (r + 1e-8))
        return pos

    def _get_obs(self, agent_id: int = 1) -> np.ndarray:
        """エージェント視点の観測を返す"""
        if agent_id == 1:
            self_pos = self.agent1_pos
            other_pos = self.agent2_pos
        else:
            self_pos = self.agent2_pos
            other_pos = self.agent1_pos

        dist_to_origin = np.linalg.norm(self_pos)
        obs = np.array(
            [
                self_pos[0],
                self_pos[1],
                other_pos[0],
                other_pos[1],
                dist_to_origin,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_rewards(self):
        """各エージェントの報酬を計算"""
        d1 = np.linalg.norm(self.agent1_pos)
        d2 = np.linalg.norm(self.agent2_pos)

        # 原点からの距離に応じた損失
        r1 = -0.1 * d1
        r2 = -0.1 * d2

        # 互いに接近し過ぎるとペナルティ
        dist_12 = np.linalg.norm(self.agent1_pos - self.agent2_pos)
        if dist_12 < 0.2:
            penalty = 0.1 * (0.2 - dist_12) / 0.2  # 0〜0.1
            r1 -= penalty
            r2 -= penalty

        return float(r1), float(r2)

    # -----------------
    # Gym API
    # -----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        # 初期位置: 原点付近にランダム配置
        self.agent1_pos = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self.agent2_pos = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)

        obs1 = self._get_obs(agent_id=1)
        info = {}
        return obs1, info

    def step(self, action):
        """action: {"agent1": np.array([vx, vy]), "agent2": np.array([vx, vy])}"""
        if isinstance(action, dict):
            a1 = np.asarray(action.get("agent1", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            a2 = np.asarray(action.get("agent2", np.zeros(2, dtype=np.float32)), dtype=np.float32)
        else:
            # 単一エージェント互換（agent1 のみ）
            a1 = np.asarray(action, dtype=np.float32)
            a2 = np.zeros(2, dtype=np.float32)

        # 速度の大きさをクリップしてから適用
        max_step = 0.1
        a1 = np.clip(a1, -1.0, 1.0) * max_step
        a2 = np.clip(a2, -1.0, 1.0) * max_step

        self.agent1_pos = self._clip_position(self.agent1_pos + a1)
        self.agent2_pos = self._clip_position(self.agent2_pos + a2)

        r1, r2 = self._compute_rewards()
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        # エピソード終了時に原点へ近い方を winner にする
        winner = "none"
        if truncated:
            d1 = np.linalg.norm(self.agent1_pos)
            d2 = np.linalg.norm(self.agent2_pos)
            if d1 < d2:
                winner = "agent1"
            elif d2 < d1:
                winner = "agent2"

        info = {
            "agent1_reward": float(r1),
            "agent2_reward": float(r2),
            "winner": winner,
        }

        obs1 = self._get_obs(agent_id=1)
        return obs1, float(r1), terminated, truncated, info

    def render(self):
        """ヘッドレス環境向け 2D 可視化

        各ステップの描画結果をメモリ上に保持し、close() 時に
        GIF アニメーションとして保存する。
        保存先: ./render_frames_coop_nav/coop_nav_animation.gif
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        ax = self._ax
        ax.clear()

        # アリーナ円
        arena = Circle((0.0, 0.0), radius=self.world_radius, fill=False, color="black", linewidth=1.5)
        ax.add_patch(arena)

        # エージェント位置
        ax.scatter(self.agent1_pos[0], self.agent1_pos[1], c="blue", label="agent1")
        ax.scatter(self.agent2_pos[0], self.agent2_pos[1], c="red", label="agent2")

        ax.set_xlim(-self.world_radius * 1.1, self.world_radius * 1.1)
        ax.set_ylim(-self.world_radius * 1.1, self.world_radius * 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"CooperativeNavigationEnv  step={self.current_step}")
        ax.legend(loc="upper right")

        # キャンバスを描画して RGBA 配列として取得
        self._fig.canvas.draw()
        # buffer_rgba() が返すバッファは後で上書きされる可能性があるので copy して保持
        frame = np.asarray(self._fig.canvas.buffer_rgba()).copy()  # (H, W, 4) uint8
        self._frames.append(frame)

    def close(self):
        # フレームがあれば GIF / MP4 として書き出す
        if self._frames:
            os.makedirs(self._render_dir, exist_ok=True)
            fps = self.metadata.get("render_fps", 5)

            # RGBA -> RGB (imageio の一部バックエンドは RGB を想定)
            frames_rgb = [f[..., :3] for f in self._frames]

            # GIF
            gif_path = os.path.join(self._render_dir, "coop_nav_animation.gif")
            imageio.mimsave(gif_path, frames_rgb, fps=fps)

            # MP4（ffmpeg が利用可能ならこちらも作成）
            mp4_path = os.path.join(self._render_dir, "coop_nav_animation.mp4")
            try:
                imageio.mimsave(mp4_path, frames_rgb, fps=fps)
            except Exception as e:
                # MP4 生成に失敗しても学習自体には影響させない
                print(f"[WARN] Failed to write MP4 animation: {e}")

        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        self._frames = []
