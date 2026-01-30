"""Twin Delayed Deep Deterministic Policy Gradient (TD3).

A strong off-policy actor-critic algorithm for continuous control,
introduced after DDPG and competitive with SAC.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float = 1.0, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.net(obs))


class Critic(nn.Module):
    """Twin Q-networks in a single module."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


@dataclass
class TD3Config:
    """TD3 のハイパーパラメータ設定用データクラス。

    Attributes:
        obs_dim: 観測ベクトルの次元数。
        action_dim: 連続行動ベクトルの次元数。
        max_action: 行動の絶対値の上限（tanh 出力に掛けるスケール）。
        hidden_dim: Actor / Critic の隠れ層ユニット数。
        gamma: 割引率 γ。
        tau: ターゲットネットワークのソフト更新係数 (0〜1)。
        policy_noise: ターゲットアクションに加えるガウスノイズの標準偏差。
        noise_clip: 上記ノイズをクリップする絶対値上限。
        policy_delay: Actor を更新する頻度（Critic 更新何回ごとに 1 回）。
        lr: Adam オプティマイザの学習率。
        device: 使用デバイス。"cpu" / "cuda" など。
    """

    obs_dim: int
    action_dim: int
    max_action: float = 1.0
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    lr: float = 3e-4
    device: str = "cpu"


class TD3Agent:
    """TD3 エージェント。

    SAC と同様に `select_action` と `update` を通じて
    自己対戦トレーナーから利用されます。
    """

    def __init__(self, config: TD3Config):
        """コンストラクタ

        Args:
            config: TD3Config でまとめた TD3 のハイパーパラメータとデバイス設定。
        """
        self.device = config.device
        self.gamma = config.gamma
        self.tau = config.tau
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_delay = config.policy_delay

        self.actor = Actor(config.obs_dim, config.action_dim, config.max_action, config.hidden_dim).to(self.device)
        self.actor_target = Actor(config.obs_dim, config.action_dim, config.max_action, config.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.lr)

        self.max_action = config.max_action
        self.total_it = 0

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size: int, gradient_steps: int = 1):
        if len(replay_buffer) < batch_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
            }

        total_critic_loss = 0.0
        total_actor_loss = 0.0

        for _ in range(gradient_steps):
            self.total_it += 1

            batch = replay_buffer.sample(batch_size)
            obs = batch["obs"]
            action = batch["action"]
            reward = batch["reward"]
            next_obs = batch["next_obs"]
            done = batch["done"]

            with torch.no_grad():
                # target policy smoothing
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                next_action = (
                    self.actor_target(next_obs) + noise
                ).clamp(-self.max_action, self.max_action)

                target_q1, target_q2 = self.critic_target(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1.0 - done) * self.gamma * target_q

            current_q1, current_q2 = self.critic(obs, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            actor_loss_value = 0.0
            if self.total_it % self.policy_delay == 0:
                # Actor update
                actor_action = self.actor(obs)
                actor_q1, _ = self.critic(obs, actor_action)
                actor_loss = -actor_q1.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Update target networks
                with torch.no_grad():
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

                actor_loss_value = actor_loss.item()

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss_value

        steps = float(gradient_steps)
        return {
            "critic_loss": total_critic_loss / steps,
            "actor_loss": total_actor_loss / steps,
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
