"""Role-aware Recurrent TD3 agent.

This variant extends Recurrent TD3 with an auxiliary task that
predicts the roles (chaser vs evader) of the *other* three agents
in the 4-agent adversarial_tag_4p environment.

- A GRU-based feature extractor processes observations.
- A role predictor head outputs Gaussian parameters over role labels
  for the three other agents.
- An actor head conditions on both the raw observation and the
  predicted role probabilities to produce actions.

Training combines:
- Standard TD3-style actor-critic updates (off-policy, recurrent)
- Supervised learning on role labels, backpropagated through the GRU
  and predictor only (actor head is not used for this loss).
"""

from dataclasses import dataclass
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo_trainer.sequence_replay import SequenceReplayBuffer, SequenceSample
from ppo_trainer.recurrent_td3 import Critic


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class FeatureGRU(nn.Module):
    """GRU-based feature extractor over observations.

    This mirrors the recurrent actor core in RecurrentTD3 but outputs
    a hidden representation instead of actions directly.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward_seq(self, obs_seq: torch.Tensor, h0: torch.Tensor):
        """Forward over a sequence.

        Args:
            obs_seq: (T, B, obs_dim)
            h0: (1, B, hidden_dim)
        Returns:
            feat_seq: (T, B, hidden_dim)
            hT: (1, B, hidden_dim)
        """

        T, B, _ = obs_seq.shape
        x = self.fc(obs_seq.view(T * B, self.obs_dim))
        x = torch.relu(x)
        x = x.view(T, B, self.hidden_dim)
        out, hT = self.gru(x, h0)
        return out, hT

    def step(self, obs: torch.Tensor, h: torch.Tensor):
        """Single-step forward.

        Args:
            obs: (B, obs_dim)
            h: (1, B, hidden_dim)
        Returns:
            feat: (B, hidden_dim)
            h_new: (1, B, hidden_dim)
        """

        obs_seq = obs.unsqueeze(0)  # (1, B, obs_dim)
        feat_seq, h_new = self.forward_seq(obs_seq, h)
        return feat_seq.squeeze(0), h_new


class RolePredictor(nn.Module):
    """Predicts roles (chaser vs evader) of the other three agents.

    For each of the 3 other agents, outputs mean and log-std of a
    Gaussian over a scalar label y in {0, 1}, where y=1 corresponds
    to chaser and y=0 to evader.
    """

    def __init__(self, hidden_dim: int, num_others: int = 3):
        super().__init__()
        self.num_others = num_others
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, num_others)
        self.log_std_head = nn.Linear(hidden_dim, num_others)

    def forward(self, feat: torch.Tensor):
        """Args:
            feat: (T or B, hidden_dim)
        Returns:
            mean: (T or B, num_others)
            log_std: (T or B, num_others)
        """

        x = self.net(feat)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class RoleAwareActor(nn.Module):
    """Actor that conditions on observation and predicted role probabilities."""

    def __init__(
        self,
        obs_dim: int,
        num_others: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_action = max_action
        input_dim = obs_dim + num_others
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, role_probs: torch.Tensor) -> torch.Tensor:
        """Compute action from observation and role probabilities.

        Args:
            obs: (N, obs_dim)
            role_probs: (N, num_others)
        Returns:
            action: (N, action_dim)
        """

        x = torch.cat([obs, role_probs], dim=-1)
        return self.max_action * torch.tanh(self.net(x))


@dataclass
class RoleAwareRecurrentTD3Config:
    """Config for the role-aware recurrent TD3 agent.

    This is intended for adversarial_tag_4p where there are four agents
    and each agent observes the other three.
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
    num_other_agents: int = 3


class RoleAwareRecurrentTD3Agent:
    """Recurrent TD3 agent with auxiliary role prediction.

    - GRU features are shared between actor and role predictor.
    - Critic is the same as in RecurrentTD3 and sees only the raw
      observation and action.
    - Actor conditions on both observation and predicted role
      probabilities.
    """

    def __init__(self, config: RoleAwareRecurrentTD3Config):
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.tau = config.tau
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_delay = config.policy_delay
        self.num_others = config.num_other_agents

        # Shared recurrent feature extractor
        self.feature = FeatureGRU(config.obs_dim, config.hidden_dim).to(self.device)
        self.feature_target = FeatureGRU(config.obs_dim, config.hidden_dim).to(self.device)
        self.feature_target.load_state_dict(self.feature.state_dict())

        # Role predictor (no target network; auxiliary task only)
        self.role_predictor = RolePredictor(config.hidden_dim, self.num_others).to(self.device)

        # Actor head (conditioned on obs + role probs)
        self.actor = RoleAwareActor(
            config.obs_dim,
            self.num_others,
            config.action_dim,
            config.max_action,
            config.hidden_dim,
        ).to(self.device)
        self.actor_target = RoleAwareActor(
            config.obs_dim,
            self.num_others,
            config.action_dim,
            config.max_action,
            config.hidden_dim,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic and target critic (same as RecurrentTD3, obs only)
        self.critic = Critic(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.lr)
        # GRU receives gradients from both supervised and RL losses
        self.predictor_optim = optim.Adam(
            list(self.feature.parameters()) + list(self.role_predictor.parameters()),
            lr=config.lr,
        )
        # Actor (and GRU) for RL
        self.actor_optim = optim.Adam(
            list(self.feature.parameters()) + list(self.actor.parameters()),
            lr=config.lr,
        )

        self.max_action = config.max_action
        self.total_it = 0
        self.eval_hidden: torch.Tensor | None = None
        # 評価時に直近の役割確率を保持（test_policy から可視化用に参照）
        self.last_role_probs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Helper: Gaussian NLL for role prediction
    # ------------------------------------------------------------------
    def _gaussian_nll(self, mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean Gaussian negative log-likelihood.

        Args:
            mean: (T, num_others)
            log_std: (T, num_others)
            target: (T, num_others)
        """

        var = torch.exp(2.0 * log_std)
        nll = 0.5 * (((target - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + math.log(2.0 * math.pi))
        return nll.mean()

    def _role_labels_for_agent(self, agent_index: int, T: int) -> torch.Tensor:
        """Return role labels for the 3 other agents for a given agent index.

        In adversarial_tag_4p we assume global indices:
            0, 2 -> chaser (label 1.0)
            1, 3 -> evader (label 0.0)
        Other-agent order follows env convention: ascending indices
        excluding self.
        """

        all_ids = [0, 1, 2, 3]
        other_ids = [i for i in all_ids if i != agent_index]
        labels = []
        for idx in other_ids:
            is_chaser = idx in (0, 2)
            labels.append(1.0 if is_chaser else 0.0)
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device).view(1, self.num_others)
        return labels_tensor.expand(T, self.num_others)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def reset_eval_hidden(self):
        self.eval_hidden = None

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action for a single observation.

        Deterministic policy: no exploration noise is added here; the
        trainer is responsible for any exploration strategy.
        """

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.eval_hidden is None:
            self.eval_hidden = self.feature.initial_hidden(1, self.device)

        with torch.no_grad():
            feat, h_new = self.feature.step(obs_tensor, self.eval_hidden)
            mean_role, _ = self.role_predictor(feat)
            role_probs = torch.sigmoid(mean_role)  # (1, num_others)
            action = self.actor(obs_tensor, role_probs)

        self.eval_hidden = h_new
        # 可視化用に numpy で保存
        self.last_role_probs = role_probs.cpu().numpy()[0]
        return action.cpu().numpy()[0]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def update(
        self,
        replay_buffer: SequenceReplayBuffer,
        batch_size: int,
        max_seq_len: int,
        gradient_steps: int = 1,
    ):
        if len(replay_buffer) == 0:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "role_loss": 0.0}

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_role_loss = 0.0

        for _ in range(gradient_steps):
            seq_batch: List[SequenceSample] = replay_buffer.sample(batch_size, max_seq_len)
            if not seq_batch:
                break

            # -------------------- Critic update (flattened transitions) --------------------
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
                # target policy: feature_target + role_predictor + actor_target
                # next_obs: (N, obs_dim) -> (T=1, B=N, obs_dim)
                N = next_obs.shape[0]
                obs_seq = next_obs.unsqueeze(0)
                h0 = self.feature_target.initial_hidden(batch_size=N, device=self.device)
                feat_seq, _ = self.feature_target.forward_seq(obs_seq, h0)
                feat = feat_seq.squeeze(0)  # (N, hidden_dim)
                mean_role, _ = self.role_predictor(feat)
                role_probs = torch.sigmoid(mean_role)

                next_action = self.actor_target(next_obs, role_probs)
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

            # -------------------- Supervised role prediction update --------------------
            self.predictor_optim.zero_grad()
            role_loss_acc = 0.0
            count = 0
            for seq in seq_batch:
                if seq.agent_index is None:
                    continue
                T = seq.obs.shape[0]
                obs_seq = torch.as_tensor(seq.obs, device=self.device).unsqueeze(1)  # (T,1,obs_dim)
                h0 = self.feature.initial_hidden(1, self.device)
                feat_seq, _ = self.feature.forward_seq(obs_seq, h0)
                feat = feat_seq.squeeze(1)  # (T, hidden_dim)

                mean_role, log_std_role = self.role_predictor(feat)
                role_targets = self._role_labels_for_agent(seq.agent_index, T)
                role_loss_seq = self._gaussian_nll(mean_role, log_std_role, role_targets)
                role_loss_acc += role_loss_seq
                count += 1

            if count > 0:
                role_loss = role_loss_acc / count
                role_loss.backward()
                self.predictor_optim.step()
                total_role_loss += float(role_loss.item())

            # -------------------- Actor update (with frozen predictor params) --------------------
            actor_loss_value = 0.0
            if self.total_it % self.policy_delay == 0:
                for p in self.role_predictor.parameters():
                    p.requires_grad = False

                self.actor_optim.zero_grad()
                actor_loss_acc = 0.0
                count = 0
                for seq in seq_batch:
                    T = seq.obs.shape[0]
                    obs_seq = torch.as_tensor(seq.obs, device=self.device).unsqueeze(1)  # (T,1,obs_dim)
                    h0 = self.feature.initial_hidden(1, self.device)
                    feat_seq, _ = self.feature.forward_seq(obs_seq, h0)
                    feat = feat_seq.squeeze(1)  # (T, hidden_dim)

                    mean_role, _ = self.role_predictor(feat)
                    role_probs = torch.sigmoid(mean_role)  # (T, num_others)

                    obs_flat = obs_seq.view(T, -1)
                    role_flat = role_probs.view(T, -1)
                    act_flat = self.actor(obs_flat, role_flat)
                    q1, _ = self.critic(obs_flat, act_flat)
                    actor_loss_seq = -q1.mean()
                    actor_loss_acc += actor_loss_seq
                    count += 1

                if count > 0:
                    actor_loss = actor_loss_acc / count
                    actor_loss.backward()
                    self.actor_optim.step()
                    actor_loss_value = float(actor_loss.item())

                for p in self.role_predictor.parameters():
                    p.requires_grad = True

                # Soft update target networks
                with torch.no_grad():
                    for param, target_param in zip(self.feature.parameters(), self.feature_target.parameters()):
                        target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
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
            "role_loss": total_role_loss / max(1.0, float(gradient_steps)),
        }

    def save(self, path: str):
        """Save model parameters (feature GRU, role predictor, actor, critic)."""
        torch.save(
            {
                "feature": self.feature.state_dict(),
                "role_predictor": self.role_predictor.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.feature.load_state_dict(checkpoint["feature"])
        self.feature_target.load_state_dict(self.feature.state_dict())
        self.role_predictor.load_state_dict(checkpoint["role_predictor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(self.critic.state_dict())
