"""Recurrent Soft Actor-Critic for continuous control.

Uses a GRU-based Gaussian policy and sequence replay buffer.
Critic (Q networks) remain feed-forward for simplicity.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo_trainer.sequence_replay import SequenceReplayBuffer, SequenceSample


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class RecurrentGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward_seq(self, obs_seq: torch.Tensor, h0: torch.Tensor):
        """Forward over a sequence.

        Args:
            obs_seq: (T, B, obs_dim)
            h0: (1, B, hidden_dim)
        Returns:
            mean_seq: (T, B, action_dim)
            log_std_seq: (T, B, action_dim)
            hT: (1, B, hidden_dim)
        """

        T, B, _ = obs_seq.shape
        x = self.fc(obs_seq.view(T * B, self.obs_dim))
        x = torch.relu(x)
        x = x.view(T, B, self.hidden_dim)
        out, hT = self.gru(x, h0)
        mean = self.mean_layer(out)
        log_std = self.log_std_layer(out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, hT

    def step(self, obs: torch.Tensor, h: torch.Tensor):
        """Single-step forward for acting.

        Args:
            obs: (B, obs_dim)
            h: (1, B, hidden_dim)
        Returns:
            action: (B, action_dim)
            log_prob: (B, 1)
            h_new: (1, B, hidden_dim)
        """

        obs_seq = obs.unsqueeze(0)
        mean_seq, log_std_seq, h_new = self.forward_seq(obs_seq, h)
        mean = mean_seq.squeeze(0)
        log_std = log_std_seq.squeeze(0)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, h_new


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
class RecurrentSACConfig:
    """Recurrent SAC のハイパーパラメータ設定用データクラス。

    Attributes:
        obs_dim: 観測ベクトルの次元数。
        action_dim: 連続行動ベクトルの次元数。
        hidden_dim: GRU ポリシーおよび Q ネットワークの隠れ層ユニット数。
        gamma: 割引率 γ。
        tau: ターゲット Q ネットワークのソフト更新係数 (0〜1)。
        lr: すべての Adam オプティマイザに用いる学習率。
        alpha: エントロピー温度の初期値（自動調整しない場合は固定）。
        automatic_entropy_tuning: True なら target_entropy に基づき α を学習。
        target_entropy: 目標エントロピー。None の場合は −action_dim を自動設定。
        device: 使用デバイス。"cpu" / "cuda" など。
    """

    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float | None = None
    device: str = "cpu"


class RecurrentSACAgent:
    """GRU ベースのポリシーを用いる Recurrent SAC エージェント。"""

    def __init__(self, config: RecurrentSACConfig):
        """コンストラクタ

        Args:
            config: RecurrentSACConfig でまとめたハイパーパラメータとデバイス設定。
        """
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.tau = config.tau

        self.policy = RecurrentGaussianPolicy(config.obs_dim, config.action_dim, config.hidden_dim).to(
            self.device
        )
        self.q1 = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2 = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q1_target = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2_target = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=config.lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=config.lr)

        if config.automatic_entropy_tuning:
            if config.target_entropy is None:
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

        self.eval_hidden: torch.Tensor | None = None

    def _alpha(self) -> torch.Tensor:
        if self.log_alpha is None:
            return torch.tensor(self.alpha, device=self.device)
        return self.log_alpha.exp()

    # Acting -----------------------------------------------------------------
    def reset_eval_hidden(self):
        self.eval_hidden = None

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.eval_hidden is None:
            self.eval_hidden = self.policy.initial_hidden(1, self.device)

        with torch.no_grad():
            if deterministic:
                # Use mean action
                mean_seq, log_std_seq, h_new = self.policy.forward_seq(obs_tensor.unsqueeze(0), self.eval_hidden)
                action = torch.tanh(mean_seq.squeeze(0))
            else:
                action, _, h_new = self.policy.step(obs_tensor, self.eval_hidden)
        self.eval_hidden = h_new
        return action.cpu().numpy()[0]

    # Training ----------------------------------------------------------------
    def update(
        self,
        replay_buffer: SequenceReplayBuffer,
        batch_size: int,
        max_seq_len: int,
        gradient_steps: int = 1,
    ):
        if len(replay_buffer) == 0:
            return {"q1_loss": 0.0, "q2_loss": 0.0, "policy_loss": 0.0, "alpha": float(self._alpha())}

        total_q1_loss = 0.0
        total_q2_loss = 0.0
        total_policy_loss = 0.0

        for _ in range(gradient_steps):
            seq_batch: List[SequenceSample] = replay_buffer.sample(batch_size, max_seq_len)
            if not seq_batch:
                break

            # Flatten all timesteps from all sequences for critic update
            obs_list = []
            action_list = []
            reward_list = []
            next_obs_list = []
            done_list = []
            for seq in seq_batch:
                obs_list.append(seq.obs)
                action_list.append(seq.action)
                reward_list.append(seq.reward)
                next_obs_list.append(seq.next_obs)
                done_list.append(seq.done)

            obs = torch.as_tensor(np.concatenate(obs_list, axis=0), device=self.device)
            action = torch.as_tensor(np.concatenate(action_list, axis=0), device=self.device)
            reward = torch.as_tensor(np.concatenate(reward_list, axis=0), device=self.device)
            next_obs = torch.as_tensor(np.concatenate(next_obs_list, axis=0), device=self.device)
            done = torch.as_tensor(np.concatenate(done_list, axis=0), device=self.device)

            with torch.no_grad():
                # Next actions from policy (no recurrence used for critic target)
                # next_obs: (N, obs_dim) -> (T=N, B=1, obs_dim)
                obs_seq = next_obs.unsqueeze(1)
                h0 = self.policy.initial_hidden(batch_size=1, device=self.device)
                next_mean, next_log_std, _ = self.policy.forward_seq(obs_seq, h0)
                # 出力 (T,1,*) を (N,*) に戻す
                next_mean = next_mean.squeeze(1)
                next_log_std = next_log_std.squeeze(1)
                next_std = next_log_std.exp()
                normal = torch.distributions.Normal(next_mean, next_std)
                x_t = normal.rsample()
                next_action = torch.tanh(x_t)
                next_log_prob = normal.log_prob(x_t) - torch.log(1 - next_action.pow(2) + 1e-6)
                next_log_prob = next_log_prob.sum(dim=-1, keepdim=True)

                target_q1 = self.q1_target(next_obs, next_action)
                target_q2 = self.q2_target(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2)
                alpha = self._alpha()
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

            # Policy update with BPTT over each sequence
            policy_loss_acc = 0.0
            alpha_loss_acc = 0.0
            count = 0

            for seq in seq_batch:
                T = seq.obs.shape[0]
                obs_seq = torch.as_tensor(seq.obs, device=self.device).unsqueeze(1)  # (T, 1, obs_dim)
                h0 = self.policy.initial_hidden(1, self.device)
                mean_seq, log_std_seq, _ = self.policy.forward_seq(obs_seq, h0)
                std_seq = log_std_seq.exp()
                normal = torch.distributions.Normal(mean_seq, std_seq)
                x_t = normal.rsample()
                action_seq = torch.tanh(x_t)
                log_prob_seq = normal.log_prob(x_t) - torch.log(1 - action_seq.pow(2) + 1e-6)
                log_prob_seq = log_prob_seq.sum(dim=-1, keepdim=True)  # (T, 1, 1)

                q1_new = self.q1(obs_seq.view(T, -1), action_seq.view(T, -1))
                q2_new = self.q2(obs_seq.view(T, -1), action_seq.view(T, -1))
                q_new = torch.min(q1_new, q2_new)

                alpha = self._alpha()
                policy_loss_seq = (alpha * log_prob_seq.view(T, 1) - q_new).mean()
                policy_loss_acc += policy_loss_seq

                if self.log_alpha is not None:
                    alpha_loss_seq = -(self.log_alpha * (log_prob_seq.detach().view(T, 1) + self.target_entropy)).mean()
                    alpha_loss_acc += alpha_loss_seq

                count += 1

            if count > 0:
                policy_loss = policy_loss_acc / count
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                if self.log_alpha is not None:
                    alpha_loss = alpha_loss_acc / count
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

                total_policy_loss += float(policy_loss.item())

            # Soft update targets
            with torch.no_grad():
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

            total_q1_loss += float(q1_loss.item())
            total_q2_loss += float(q2_loss.item())

        steps = max(1, gradient_steps)
        return {
            "q1_loss": total_q1_loss / steps,
            "q2_loss": total_q2_loss / steps,
            "policy_loss": total_policy_loss / steps,
            "alpha": float(self._alpha()),
        }

    def save(self, path: str):
        """モデルパラメータを保存する。

        - ポリシーネットワーク (RecurrentGaussianPolicy)
        - Q1 / Q2 ネットワーク

        Optimizer やターゲットネットワークは含めていません。
        """
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """保存されたパラメータを読み込む。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
