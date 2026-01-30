"""Soft Actor-Critic (SAC) implementation for continuous control.

This module is designed to be compatible with the existing self-play
infrastructure. It shares a similar interface to PPOAgent so that
Self-play specific trainers can use it.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous control.

    Outputs mean and log_std, and supports reparameterized sampling.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.base = MLP(obs_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.base(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        # Change of variables formula for tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


@dataclass
class SACConfig:
    """SAC のハイパーパラメータ設定用データクラス。

    Attributes:
        obs_dim: 観測ベクトルの次元数。
        action_dim: 連続行動ベクトルの次元数。
        hidden_dim: ポリシー／Qネットワークの隠れ層ユニット数。
        gamma: 割引率 γ。
        tau: ターゲットネットワークのソフト更新係数 (0〜1)。
        lr: すべての Adam オプティマイザで用いる学習率。
        alpha: エントロピー温度の初期値（自動調整しない場合は固定値）。
        automatic_entropy_tuning: True の場合、target_entropy に基づき α を学習。
        target_entropy: 目標エントロピー。None の場合は −action_dim を自動設定。
        device: 使用デバイス。"cpu" または "cuda" など。
    """

    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2  # initial entropy temperature (may be tuned automatically)
    automatic_entropy_tuning: bool = True
    target_entropy: Optional[float] = None
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.tensor(self.obs[idxs], device=self.device),
            action=torch.tensor(self.action[idxs], device=self.device),
            reward=torch.tensor(self.reward[idxs], device=self.device),
            next_obs=torch.tensor(self.next_obs[idxs], device=self.device),
            done=torch.tensor(self.done[idxs], device=self.device),
        )
        return batch

    def __len__(self):
        return self.size


class SoftActorCriticAgent:
    """Soft Actor-Critic agent for continuous control.

    Interface is similar to PPOAgent where possible:
    - select_action(obs, deterministic=False) -> np.ndarray
    - update(replay_buffer, batch_size, gradient_steps) -> metrics dict
    """

    def __init__(self, config: SACConfig):
        """コンストラクタ

        Args:
            config: SACConfig でまとめたハイパーパラメータとデバイス設定。
                主に obs_dim / action_dim / hidden_dim / gamma / tau / lr /
                alpha / automatic_entropy_tuning / target_entropy / device を使用します。
        """
        self.device = config.device
        self.gamma = config.gamma
        self.tau = config.tau

        self.policy = GaussianPolicy(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q1 = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2 = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q1_target = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2_target = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=config.lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=config.lr)

        # Entropy temperature
        if config.automatic_entropy_tuning:
            if config.target_entropy is None:
                # heuristic: -dim(A)
                target_entropy = -float(config.action_dim)
            else:
                target_entropy = config.target_entropy

            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=config.lr)
            self.target_entropy = target_entropy
        else:
            self.log_alpha = None
            self.alpha_optim = None
            self.target_entropy = None

        self.alpha = config.alpha

    def _get_alpha(self) -> float:
        if self.log_alpha is None:
            return self.alpha
        return float(self.log_alpha.exp().item())

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.policy.deterministic(obs_tensor)
            else:
                action, _ = self.policy.sample(obs_tensor)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer: ReplayBuffer, batch_size: int, gradient_steps: int = 1):
        """Update SAC networks from replay buffer.

        Returns a dict of average losses over gradient_steps.
        """
        if len(replay_buffer) < batch_size:
            return {
                "q1_loss": 0.0,
                "q2_loss": 0.0,
                "policy_loss": 0.0,
                "alpha": self._get_alpha(),
            }

        total_q1_loss = 0.0
        total_q2_loss = 0.0
        total_policy_loss = 0.0

        for _ in range(gradient_steps):
            batch = replay_buffer.sample(batch_size)
            obs = batch["obs"]
            action = batch["action"]
            reward = batch["reward"]
            next_obs = batch["next_obs"]
            done = batch["done"]

            # Critic update
            with torch.no_grad():
                next_action, next_log_prob = self.policy.sample(next_obs)
                target_q1 = self.q1_target(next_obs, next_action)
                target_q2 = self.q2_target(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2)
                alpha = self._get_alpha()
                target_value = target_q - alpha * next_log_prob
                target_q_value = reward + (1.0 - done) * self.gamma * target_value

            current_q1 = self.q1(obs, action)
            current_q2 = self.q2(obs, action)

            q1_loss = F.mse_loss(current_q1, target_q_value)
            q2_loss = F.mse_loss(current_q2, target_q_value)

            self.q1_optim.zero_grad()
            q1_loss.backward()
            self.q1_optim.step()

            self.q2_optim.zero_grad()
            q2_loss.backward()
            self.q2_optim.step()

            # Policy update
            new_action, log_prob = self.policy.sample(obs)
            q1_new = self.q1(obs, new_action)
            q2_new = self.q2(obs, new_action)
            q_new = torch.min(q1_new, q2_new)

            alpha = self._get_alpha()
            policy_loss = (alpha * log_prob - q_new).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Temperature update
            if self.log_alpha is not None:
                alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

            # Soft update target networks
            with torch.no_grad():
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

            total_q1_loss += q1_loss.item()
            total_q2_loss += q2_loss.item()
            total_policy_loss += policy_loss.item()

        steps = float(gradient_steps)
        return {
            "q1_loss": total_q1_loss / steps,
            "q2_loss": total_q2_loss / steps,
            "policy_loss": total_policy_loss / steps,
            "alpha": self._get_alpha(),
        }

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
