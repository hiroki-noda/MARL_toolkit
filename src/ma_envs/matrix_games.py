import numpy as np
import gymnasium as gym
from gymnasium import spaces


class _BaseMatrixGameEnv(gym.Env):
    """2エージェント用の簡易マトリクスゲーム基底クラス

    - アクション: 各エージェント1次元の連続値 (-1 ~ 1)
      -> env側で二値化して離散行動にマッピング
    - 観測: 前ステップの自分/相手の二値行動と自分の報酬、正規化ステップ数
      例: [a_self, a_opp, reward_self, step / max_step]
    - stepの返り値・infoの構造は AntSumoEnv と同じ形式に合わせる:
      obs(agent1用), reward1, terminated, truncated, info
      info["agent1_reward"], info["agent2_reward"], info["winner"]
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, max_episode_steps: int = 10):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # 連続1次元アクション（符号で離散行動に変換）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # 観測: [a_self, a_opp, reward_self, step_norm]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self._last_actions = None  # (a1, a2) in {-1.0, +1.0}
        self._last_rewards = None  # (r1, r2)
        self._ep_return1 = 0.0
        self._ep_return2 = 0.0

    # --- ヘルパ ---

    def _continuous_to_binary(self, a: float) -> int:
        """連続値 [-1,1] -> 離散 {0,1}
        0: 行動0, 1: 行動1
        """
        return 1 if a >= 0.0 else 0

    def _binary_to_obs_value(self, b: int) -> float:
        """離散 {0,1} -> 観測用の値 {-1,+1}"""
        return 1.0 if b == 1 else -1.0

    # --- Gym API ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self._ep_return1 = 0.0
        self._ep_return2 = 0.0
        self._last_actions = None
        self._last_rewards = None

        obs1 = self._get_obs(agent_id=1)
        info = {}
        return obs1, info

    def _get_obs(self, agent_id: int = 1):
        # 初期ステップはゼロ埋め
        if self._last_actions is None or self._last_rewards is None:
            a_self = 0.0
            a_opp = 0.0
            r_self = 0.0
        else:
            a1, a2 = self._last_actions
            r1, r2 = self._last_rewards
            if agent_id == 1:
                a_self = a1
                a_opp = a2
                r_self = r1
            else:
                a_self = a2
                a_opp = a1
                r_self = r2

        step_norm = float(self.current_step) / float(max(1, self.max_episode_steps))
        obs = np.array([a_self, a_opp, r_self, step_norm], dtype=np.float32)
        return obs

    # 具象クラスで _payoff(a1_bin, a2_bin) を実装する

    def _payoff(self, a1_bin: int, a2_bin: int):  # pragma: no cover - 実装はサブクラス
        raise NotImplementedError

    def step(self, action):
        # action: {"agent1": np.array([..]), "agent2": np.array([..])}
        if isinstance(action, dict):
            a1_cont = float(np.clip(action.get("agent1", np.zeros(1))[0], -1.0, 1.0))
            a2_cont = float(np.clip(action.get("agent2", np.zeros(1))[0], -1.0, 1.0))
        else:
            # 単一エージェント用（agent1のみ）の互換性確保
            a1_cont = float(np.clip(action[0], -1.0, 1.0))
            a2_cont = 0.0

        a1_bin = self._continuous_to_binary(a1_cont)
        a2_bin = self._continuous_to_binary(a2_cont)

        # 観測用には -1 / +1 にマッピング
        a1_obs = self._binary_to_obs_value(a1_bin)
        a2_obs = self._binary_to_obs_value(a2_bin)

        r1, r2 = self._payoff(a1_bin, a2_bin)

        self._ep_return1 += r1
        self._ep_return2 += r2
        self.current_step += 1

        self._last_actions = (a1_obs, a2_obs)
        self._last_rewards = (r1, r2)

        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        # エピソード終了時のみ winner を決める
        winner = "none"
        if truncated:
            if self._ep_return1 > self._ep_return2:
                winner = "agent1"
            elif self._ep_return2 > self._ep_return1:
                winner = "agent2"
            else:
                winner = "none"

        info = {
            "agent1_reward": float(r1),
            "agent2_reward": float(r2),
            "winner": winner,
        }

        obs1 = self._get_obs(agent_id=1)
        return obs1, float(r1), terminated, truncated, info

    def render(self):
        if self._last_actions is None or self._last_rewards is None:
            print("[Render] step=0 (no actions yet)")
            return
        a1, a2 = self._last_actions
        r1, r2 = self._last_rewards
        print(
            f"[Render] step={self.current_step} "
            f"a1={a1:+.1f}, a2={a2:+.1f}, r1={r1:+.2f}, r2={r2:+.2f}"
        )

    def close(self):
        pass


class IteratedPrisonersDilemmaEnv(_BaseMatrixGameEnv):
    """繰り返し囚人のジレンマ (IPD)

    離散行動:
      0: 裏切り (Defect)
      1: 協調 (Cooperate)

    利得行列 (エージェント1視点):
      - 両者協調 (C, C): (3, 3)
      - 両者裏切り (D, D): (1, 1)
      - 自分だけ裏切り (D, C): (5, 0)
      - 自分だけ協調 (C, D): (0, 5)

    報酬は 0〜5 を 0〜1 くらいのスケールにするため 5 で割る。
    """

    def __init__(self, max_episode_steps: int = 10):
        super().__init__(max_episode_steps=max_episode_steps)

    def _payoff(self, a1_bin: int, a2_bin: int):
        # 0: D, 1: C
        if a1_bin == 1 and a2_bin == 1:  # C, C
            r1, r2 = 3.0, 3.0
        elif a1_bin == 0 and a2_bin == 0:  # D, D
            r1, r2 = 1.0, 1.0
        elif a1_bin == 0 and a2_bin == 1:  # D, C
            r1, r2 = 5.0, 0.0
        else:  # C, D
            r1, r2 = 0.0, 5.0

        # スケールを揃えるために 5 で割る
        return r1 / 5.0, r2 / 5.0


class MatchingPenniesEnv(_BaseMatrixGameEnv):
    """マッチングペニーズ (ゼロ和ゲーム)

    離散行動:
      0: 表 (Heads)
      1: 裏 (Tails)

    ルール:
      - a1 == a2: エージェント1の勝ち (+1, -1)
      - a1 != a2: エージェント2の勝ち (-1, +1)
    """

    def __init__(self, max_episode_steps: int = 10):
        super().__init__(max_episode_steps=max_episode_steps)

    def _payoff(self, a1_bin: int, a2_bin: int):
        if a1_bin == a2_bin:
            return 1.0, -1.0
        else:
            return -1.0, 1.0


class ContinuousCoordinationEnv(_BaseMatrixGameEnv):
    """連続値を揃える協調ゲーム

    - 各エージェントの内部連続行動 a1, a2 in [-1, 1]
    - env 内部では符号のみを二値化して観測に使うが、
      報酬は元の連続値を使って距離ベースで与える。

    報酬:
      r1 = r2 = 1 - |a1 - a2|
      (完全一致で +1, 反対側で -1)
    """

    def __init__(self, max_episode_steps: int = 10):
        super().__init__(max_episode_steps=max_episode_steps)

    def step(self, action):
        # 元の連続値を保持しておきたいので、ここだけオーバーライド
        if isinstance(action, dict):
            a1_cont = float(np.clip(action.get("agent1", np.zeros(1))[0], -1.0, 1.0))
            a2_cont = float(np.clip(action.get("agent2", np.zeros(1))[0], -1.0, 1.0))
        else:
            a1_cont = float(np.clip(action[0], -1.0, 1.0))
            a2_cont = 0.0

        a1_bin = self._continuous_to_binary(a1_cont)
        a2_bin = self._continuous_to_binary(a2_cont)
        a1_obs = self._binary_to_obs_value(a1_bin)
        a2_obs = self._binary_to_obs_value(a2_bin)

        # 連続値の距離に基づく協調報酬
        diff = abs(a1_cont - a2_cont)
        reward = 1.0 - diff  # diff=0 -> 1, diff=2 -> -1
        r1 = r2 = float(np.clip(reward, -1.0, 1.0))

        self._ep_return1 += r1
        self._ep_return2 += r2
        self.current_step += 1

        self._last_actions = (a1_obs, a2_obs)
        self._last_rewards = (r1, r2)

        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        # 協調タスクなので winner は常に "none" とする
        info = {
            "agent1_reward": float(r1),
            "agent2_reward": float(r2),
            "winner": "none",
        }

        obs1 = self._get_obs(agent_id=1)
        return obs1, float(r1), terminated, truncated, info