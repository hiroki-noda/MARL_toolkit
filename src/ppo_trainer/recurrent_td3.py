"""Recurrent TD3 (Twin Delayed DDPG) for continuous control.

Uses a GRU-based deterministic policy and sequence replay buffer.
Critics remain feed-forward.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo_trainer.sequence_replay import SequenceReplayBuffer, SequenceSample


class RecurrentActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float = 1.0, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action

        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.out = nn.Linear(hidden_dim, action_dim)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward_seq(self, obs_seq: torch.Tensor, h0: torch.Tensor):
        T, B, _ = obs_seq.shape
        x = self.fc(obs_seq.view(T * B, self.obs_dim))
        x = torch.relu(x)
        x = x.view(T, B, self.hidden_dim)
        out, hT = self.gru(x, h0)
        action = self.max_action * torch.tanh(self.out(out))
        return action, hT

    def step(self, obs: torch.Tensor, h: torch.Tensor):
        obs_seq = obs.unsqueeze(0)
        act_seq, h_new = self.forward_seq(obs_seq, h)
        return act_seq.squeeze(0), h_new


class Critic(nn.Module):
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

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


@dataclass
class RecurrentTD3Config:
    """Recurrent TD3 のハイパーパラメータ設定用データクラス。

    Attributes:
        obs_dim: 観測ベクトルの次元数。
        action_dim: 連続行動ベクトルの次元数。
        max_action: 行動の絶対値の上限（tanh 出力に掛けるスケール）。
        hidden_dim: GRU アクターおよび Critic の隠れ層ユニット数。
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


class RecurrentTD3Agent:
    """GRU ベースのアクターを用いる Recurrent TD3 エージェント。"""

    def __init__(self, config: RecurrentTD3Config):
        """コンストラクタ

        Args:
            config: RecurrentTD3Config でまとめたハイパーパラメータとデバイス設定。
        """
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.tau = config.tau
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_delay = config.policy_delay

        self.actor = RecurrentActor(config.obs_dim, config.action_dim, config.max_action, config.hidden_dim).to(
            self.device
        )
        self.actor_target = RecurrentActor(
            config.obs_dim, config.action_dim, config.max_action, config.hidden_dim
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.lr)

        self.max_action = config.max_action
        self.total_it = 0
        self.eval_hidden: torch.Tensor | None = None

    # Acting -----------------------------------------------------------------
    def reset_eval_hidden(self):
        self.eval_hidden = None

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.eval_hidden is None:
            self.eval_hidden = self.actor.initial_hidden(1, self.device)

        with torch.no_grad():
            action, h_new = self.actor.step(obs_tensor, self.eval_hidden)
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
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        total_critic_loss = 0.0
        total_actor_loss = 0.0

        for _ in range(gradient_steps):
            seq_batch: List[SequenceSample] = replay_buffer.sample(batch_size, max_seq_len)
            if not seq_batch:
                break

            # Flatten transitions for critic update
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
                # target policy smoothing using recurrent actor on next_obs (no sequence info)
                # next_obs: (N, obs_dim) を「長さ1のシーケンス, バッチN」として処理する
                obs_seq = next_obs.unsqueeze(0)  # (T=1, B=N, obs_dim)
                h0 = self.actor_target.initial_hidden(batch_size=next_obs.shape[0], device=self.device)
                next_action_seq, _ = self.actor_target.forward_seq(obs_seq, h0)
                next_action = next_action_seq.squeeze(0)  # (N, action_dim)
                noise = (
                    torch.randn_like(next_action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

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
                # Actor update with BPTT over sequences
                actor_loss_acc = 0.0
                count = 0
                for seq in seq_batch:
                    T = seq.obs.shape[0]
                    obs_seq = torch.as_tensor(seq.obs, device=self.device).unsqueeze(1)  # (T, 1, obs_dim)
                    h0 = self.actor.initial_hidden(1, self.device)
                    act_seq, _ = self.actor.forward_seq(obs_seq, h0)
                    q1, _ = self.critic(obs_seq.view(T, -1), act_seq.view(T, -1))
                    actor_loss_seq = -q1.mean()
                    actor_loss_acc += actor_loss_seq
                    count += 1

                if count > 0:
                    actor_loss = actor_loss_acc / count
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    actor_loss_value = float(actor_loss.item())

                # Soft update targets
                with torch.no_grad():
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

            self.total_it += 1
            total_critic_loss += float(critic_loss.item())
            total_actor_loss += actor_loss_value

        steps = max(1, gradient_steps)
        return {
            "critic_loss": total_critic_loss / steps,
            "actor_loss": total_actor_loss / steps,
        }

    def save(self, path: str):
        """モデルパラメータを保存する。

        - 再帰アクター (RecurrentActor)
        - クリティック (双子 Q ネットワーク)

        Optimizer やターゲットネットワークは含めていません。
        """
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """保存されたパラメータを読み込む。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
