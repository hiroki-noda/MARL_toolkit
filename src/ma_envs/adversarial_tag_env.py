import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib

# ヘッドレス環境向けに GUI バックエンドではなく Agg を使用
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio


class AdversarialTagEnv(gym.Env):
    """2エージェントの対戦型タグ（鬼ごっこ）環境

    - 2D 連続空間内で、Agent1(追跡者) が Agent2(逃走者) を追いかける。
    - 観測（各エージェント視点 7次元）:
        [self_x, self_y, other_x, other_y, dist_to_other, dist_to_origin, role_id]
        role_id = +1.0 (Agent1/追跡者), -1.0 (Agent2/逃走者)
    - 行動: 2次元連続ベクトル (vx, vy) in [-1, 1]
    - 報酬:
        * Agent1: タグ成功(+1)、タイムペナルティ(-0.01)
        * Agent2: タグされたら(-1)、生存ボーナス(+0.01)
    - 終了条件:
        * 2者間距離 < tag_radius でエピソード終了
        * max_episode_steps 到達でタイムアウト

    step 返り値・info 形式は他の環境と揃える:
        obs(agent1), reward1, terminated, truncated, info
    info には "agent1_reward", "agent2_reward", "winner" を含める。
    "winner" はタグ成功時に "agent1"、タイムアウト時で未タグなら "agent2"。
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, max_episode_steps: int = 100, world_radius: float = 2.0, tag_radius: float = 0.15):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.world_radius = world_radius
        self.tag_radius = tag_radius
        self.current_step = 0

        # 各エージェントの位置 (x, y)
        self.agent1_pos = np.zeros(2, dtype=np.float32)
        self.agent2_pos = np.zeros(2, dtype=np.float32)

        # アクション: 2次元連続速度ベクトル
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 観測: [self_x, self_y, other_x, other_y, dist_to_other, dist_to_origin, role_id]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

        # レンダリング用（ヘッドレス: 後でGIFにまとめる）
        self._fig = None
        self._ax = None
        self._render_dir = "render_frames_adversarial_tag"
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
            role_id = 1.0
        else:
            self_pos = self.agent2_pos
            other_pos = self.agent1_pos
            role_id = -1.0

        dist_to_other = np.linalg.norm(self_pos - other_pos)
        dist_to_origin = np.linalg.norm(self_pos)

        obs = np.array(
            [
                self_pos[0],
                self_pos[1],
                other_pos[0],
                other_pos[1],
                dist_to_other,
                dist_to_origin,
                role_id,
            ],
            dtype=np.float32,
        )
        return obs

    # -----------------
    # Gym API
    # -----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        # 追跡者は原点近く、逃走者はランダムな方向に少し離して配置
        self.agent1_pos = np.random.uniform(-0.2, 0.2, size=2).astype(np.float32)
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        radius = np.random.uniform(0.5, 1.0)
        self.agent2_pos = np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
        )
        self.agent2_pos = self._clip_position(self.agent2_pos)

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

        self.current_step += 1

        # 距離と終了判定
        dist_12 = np.linalg.norm(self.agent1_pos - self.agent2_pos)
        tagged = dist_12 < self.tag_radius

        terminated = tagged
        truncated = (not tagged) and (self.current_step >= self.max_episode_steps)

        # 報酬
        r1 = -0.01  # 時間ペナルティ
        r2 = +0.01  # 生存ボーナス

        if tagged:
            r1 += 1.0
            r2 -= 1.0

        # winner 判定
        winner = "none"
        if tagged:
            winner = "agent1"
        elif truncated:
            winner = "agent2"  # 逃げ切り

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
        保存先: ./render_frames_adversarial_tag/adversarial_tag_animation.gif
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        ax = self._ax
        ax.clear()

        # アリーナ円
        arena = Circle((0.0, 0.0), radius=self.world_radius, fill=False, color="black", linewidth=1.5)
        ax.add_patch(arena)

        # エージェント位置（追跡者: 青, 逃走者: 赤）
        ax.scatter(self.agent1_pos[0], self.agent1_pos[1], c="blue", label="agent1 (chaser)")
        ax.scatter(self.agent2_pos[0], self.agent2_pos[1], c="red", label="agent2 (evader)")

        ax.set_xlim(-self.world_radius * 1.1, self.world_radius * 1.1)
        ax.set_ylim(-self.world_radius * 1.1, self.world_radius * 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"AdversarialTagEnv  step={self.current_step}")
        ax.legend(loc="upper right")

        # キャンバスを描画して RGBA 配列として取得
        self._fig.canvas.draw()
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
            gif_path = os.path.join(self._render_dir, "adversarial_tag_animation.gif")
            imageio.mimsave(gif_path, frames_rgb, fps=fps)

            # MP4（ffmpeg が利用可能ならこちらも作成）
            mp4_path = os.path.join(self._render_dir, "adversarial_tag_animation.mp4")
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


class AdversarialTagFourAgentEnv(gym.Env):
    """4体版の対戦型タグ環境（追跡者2体 vs 逃走者2体）

    - 2D 連続空間に 4 体のエージェントが存在する。
      agent1, agent3: 追跡者 (chaser)
      agent2, agent4: 逃走者 (evader)
    - 各エージェントは「自分自身の役割」だけを観測し、
      他エージェントの役割は観測に含めない。
    - 行動は各エージェントごとに 2 次元連続ベクトル (vx, vy) in [-1, 1]。
      -> 1体につき1つのエージェント（方策）を対応させられる構造。
    - 終了条件:
        * 逃走者2体がいずれかの追跡者に両方とも捕まったら
            -> 追跡者チームの勝利 (winner="agent1")
        * 逃走者同士が接触したら
            -> 逃走者チームの勝利 (winner="agent2")
        * 追跡者同士が接触しても何も起こらない
        * max_episode_steps 到達でタイムアウト
            -> 逃走者チームの勝利 (逃げ切り)
    - 報酬:
        * 各追跡者: 時間ペナルティ -0.01
        * 各逃走者: 生存ボーナス +0.01
        * 終了時に勝利チームの全エージェントへ +1.0、
          敗北チームの全エージェントへ -1.0 を加算。

    step の返り値・info 形式:
        obs(agent1), reward1, terminated, truncated, info
    info には "agent1_reward"〜"agent4_reward" と "winner" を含める。
    "winner" は "agent1" (追跡チーム) / "agent2" (逃走チーム) / "none"。
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, max_episode_steps: int = 100, world_radius: float = 2.0, tag_radius: float = 0.15):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.world_radius = world_radius
        self.tag_radius = tag_radius
        self.current_step = 0

        # 4体分の位置 (agent1..4)
        # agent1,3: chaser / agent2,4: evader
        self.positions = np.zeros((4, 2), dtype=np.float32)

        # 逃走者が捕まれたかどうか（agent2, agent4 のみ意味を持つ）
        self.evader_captured = np.zeros(4, dtype=bool)

        # 各エージェントごとの 2 次元連続アクション
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 観測: [self_x, self_y, other1_x, other1_y, other2_x, other2_y,
        #        other3_x, other3_y, dist_to_origin, role_id]
        # 役割ID: +1.0 (chaser), -1.0 (evader)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        # レンダリング用
        self._fig = None
        self._ax = None
        self._render_dir = "render_frames_adversarial_tag_4p"
        self._frames = []  # list[np.ndarray]

        # 各エージェントによる他3体の役割予測 (chaser である確率)
        # role_predictions[i] は agent(i+1) の予測を表し、
        # 長さ3の配列で「自分以外の agent ID を昇順に並べた順」の確率を持つ。
        self.role_predictions = [None, None, None, None]

    # -----------------
    # ヘルパ
    # -----------------

    def _clip_position(self, pos: np.ndarray) -> np.ndarray:
        """ワールド内に位置をクリップ"""
        r = np.linalg.norm(pos)
        if r > self.world_radius:
            pos = pos * (self.world_radius / (r + 1e-8))
        return pos

    def _role_id(self, agent_id: int) -> float:
        """agent_id に応じた役割IDを返す (+1: chaser, -1: evader)"""
        # agent1,3 => chaser / agent2,4 => evader
        return 1.0 if agent_id in (1, 3) else -1.0

    def _get_obs(self, agent_id: int = 1) -> np.ndarray:
        """各エージェント視点の観測を返す

        - 自身の位置
        - 他の3体の位置
        - 自身の原点からの距離
        - 自身の役割ID (role_id)

        他エージェントの役割は一切含めない。
        """
        assert 1 <= agent_id <= 4
        idx = agent_id - 1

        self_pos = self.positions[idx]
        other_indices = [i for i in range(4) if i != idx]
        other_pos = [self.positions[i] for i in other_indices]

        dist_to_origin = np.linalg.norm(self_pos)
        role_id = self._role_id(agent_id)

        obs = np.array(
            [
                self_pos[0],
                self_pos[1],
                other_pos[0][0], other_pos[0][1],
                other_pos[1][0], other_pos[1][1],
                other_pos[2][0], other_pos[2][1],
                dist_to_origin,
                role_id,
            ],
            dtype=np.float32,
        )
        return obs

    # -----------------
    # Gym API
    # -----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        # 追跡者 (agent1, agent3) は原点近くに配置
        self.positions[0] = np.random.uniform(-0.2, 0.2, size=2).astype(np.float32)
        self.positions[2] = np.random.uniform(-0.2, 0.2, size=2).astype(np.float32)

        # 逃走者 (agent2, agent4) はやや外側のリング上に配置
        for idx in [1, 3]:
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            radius = np.random.uniform(0.5, 1.0)
            pos = np.array(
                [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
            )
            self.positions[idx] = self._clip_position(pos)

        self.evader_captured[:] = False

        obs1 = self._get_obs(agent_id=1)
        info = {}
        return obs1, info

    def step(self, action):
        """action: {"agent1": [vx, vy], ..., "agent4": [vx, vy]}

        4体それぞれに 1 つの行動ベクトルを割り当てる。
        """
        if isinstance(action, dict):
            acts = []
            for aid in range(1, 5):
                a = np.asarray(action.get(f"agent{aid}", np.zeros(2, dtype=np.float32)), dtype=np.float32)
                acts.append(a)
            acts = np.stack(acts, axis=0)
        else:
            # 単一エージェント互換（agent1 のみ）
            a1 = np.asarray(action, dtype=np.float32)
            acts = np.zeros((4, 2), dtype=np.float32)
            acts[0] = a1

        # 速度の大きさをクリップしてから適用
        max_step = 0.1
        acts = np.clip(acts, -1.0, 1.0) * max_step

        # 逃走者で既に捕まっている個体は移動させない
        for idx in range(4):
            # agent2,4 が evader
            is_evader = (idx in (1, 3))
            if is_evader and self.evader_captured[idx]:
                continue
            self.positions[idx] = self._clip_position(self.positions[idx] + acts[idx])

        self.current_step += 1

        # 追跡者 (0,2) と逃走者 (1,3) 間の距離
        chasers = [0, 2]
        evaders = [1, 3]

        # 新たに捕まった逃走者を判定
        for e in evaders:
            if self.evader_captured[e]:
                continue
            for c in chasers:
                dist_ce = np.linalg.norm(self.positions[c] - self.positions[e])
                if dist_ce < self.tag_radius:
                    self.evader_captured[e] = True
                    break

        both_captured = all(self.evader_captured[e] for e in evaders)

        # 逃走者同士の接触
        dist_e1_e2 = np.linalg.norm(self.positions[1] - self.positions[3])
        evaders_collided = dist_e1_e2 < self.tag_radius

        terminated = both_captured or evaders_collided
        truncated = (not terminated) and (self.current_step >= self.max_episode_steps)

        # 報酬（時間ペナルティ / 生存ボーナス）
        rewards = np.zeros(4, dtype=np.float32)
        for idx in range(4):
            is_chaser = (idx in chasers)
            if is_chaser:
                rewards[idx] += -0.01
            else:
                rewards[idx] += +0.01

        # 勝敗に応じたボーナス
        winner = "none"
        if both_captured:
            # 追跡者チームの勝利
            for idx in range(4):
                if idx in chasers:
                    rewards[idx] += 1.0
                else:
                    rewards[idx] -= 1.0
            winner = "agent1"  # chaser team
        elif evaders_collided:
            # 逃走者チームの勝利
            for idx in range(4):
                if idx in evaders:
                    rewards[idx] += 1.0
                else:
                    rewards[idx] -= 1.0
            winner = "agent2"  # evader team
        elif truncated:
            # 時間切れ -> 逃走者チームの勝利（逃げ切り）
            winner = "agent2"

        info = {
            "agent1_reward": float(rewards[0]),
            "agent2_reward": float(rewards[1]),
            "agent3_reward": float(rewards[2]),
            "agent4_reward": float(rewards[3]),
            "winner": winner,
        }

        obs1 = self._get_obs(agent_id=1)
        return obs1, float(rewards[0]), terminated, truncated, info

    def set_agent_role_prediction(self, agent_id: int, probs: np.ndarray):
        """指定したエージェントが他3体を chaser と予測する確率をセットする。

        Args:
            agent_id: 1〜4
            probs: shape (3,) の配列。順番は「agent_id 以外の ID を昇順に並べた順」。
        """
        if not (1 <= agent_id <= 4):
            return
        if probs is None:
            self.role_predictions[agent_id - 1] = None
            return

        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 3:
            self.role_predictions[agent_id - 1] = arr
        else:
            # 形がおかしい場合は無視
            self.role_predictions[agent_id - 1] = None

    def render(self):
        """ヘッドレス環境向け 2D 可視化"""
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        ax = self._ax
        ax.clear()

        # アリーナ円
        arena = Circle((0.0, 0.0), radius=self.world_radius, fill=False, color="black", linewidth=1.5)
        ax.add_patch(arena)

        # 追跡者 (agent1,3): 青 / 逃走者 (agent2,4): 赤
        ax.scatter(self.positions[0, 0], self.positions[0, 1], c="blue", label="chaser1 (agent1)")
        ax.scatter(self.positions[2, 0], self.positions[2, 1], c="blue", marker="x", label="chaser2 (agent3)")

        e1_color = "gray" if self.evader_captured[1] else "red"
        e2_color = "gray" if self.evader_captured[3] else "red"
        ax.scatter(self.positions[1, 0], self.positions[1, 1], c=e1_color, label="evader1 (agent2)")
        ax.scatter(self.positions[3, 0], self.positions[3, 1], c=e2_color, marker="x", label="evader2 (agent4)")

        ax.set_xlim(-self.world_radius * 1.1, self.world_radius * 1.1)
        ax.set_ylim(-self.world_radius * 1.1, self.world_radius * 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"AdversarialTagFourAgentEnv  step={self.current_step}")

        # 各エージェントによる役割予測を図中に表示
        lines = []
        for idx in range(4):
            probs = self.role_predictions[idx]
            if probs is None:
                continue
            agent_id = idx + 1
            other_ids = [aid for aid in range(1, 5) if aid != agent_id]
            lines.append(f"Agent{agent_id} prediction (P[chaser]):")
            for i, aid in enumerate(other_ids):
                p = float(probs[i])
                role = "chaser" if p >= 0.5 else "evader"
                lines.append(f"  agent{aid}: {role} (p={p:.2f})")

        if lines:
            text = "\n".join(lines)
            ax.text(
                0.02,
                0.02,
                text,
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7),
            )
        ax.legend(loc="upper right")

        # キャンバスを描画して RGBA 配列として取得
        self._fig.canvas.draw()
        frame = np.asarray(self._fig.canvas.buffer_rgba()).copy()
        self._frames.append(frame)

    def close(self):
        # フレームがあれば GIF / MP4 として書き出す
        if self._frames:
            os.makedirs(self._render_dir, exist_ok=True)
            fps = self.metadata.get("render_fps", 5)

            frames_rgb = [f[..., :3] for f in self._frames]

            gif_path = os.path.join(self._render_dir, "adversarial_tag_4p_animation.gif")
            imageio.mimsave(gif_path, frames_rgb, fps=fps)

            mp4_path = os.path.join(self._render_dir, "adversarial_tag_4p_animation.mp4")
            try:
                imageio.mimsave(mp4_path, frames_rgb, fps=fps)
            except Exception as e:
                print(f"[WARN] Failed to write MP4 animation (4p): {e}")

        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        self._frames = []
