"""Role-aware Recurrent PPO agent.

This variant extends Recurrent PPO with an auxiliary task that
predicts the roles (chaser vs evader) of the *other* three agents
in the 4-agent adversarial_tag_4p environment.

- A GRU-based feature extractor processes observations.
- A role predictor head outputs Gaussian parameters over role labels
  for the three other agents.
- An actor head conditions on GRU features and predicted role
  probabilities, while the critic uses GRU features only.

The interface is kept compatible with RecurrentPPOAgent so that
existing recurrent self-play trainers can be reused.
"""

from dataclasses import dataclass
import math
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo_trainer.role_rtd3 import FeatureGRU, RolePredictor, LOG_STD_MIN, LOG_STD_MAX


class RoleAwareRecurrentActorCritic(nn.Module):
    """GRU-based Actor-Critic with auxiliary role prediction.

    - feature: GRU over observations
    - role_predictor: predicts roles of other agents from features
    - actor: uses [feature, role_probs] to produce action means
    - critic: uses features to produce state values
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, num_others: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_others = num_others

        self.feature = FeatureGRU(obs_dim, hidden_dim)
        self.role_predictor = RolePredictor(hidden_dim, num_others)

        actor_input_dim = hidden_dim + num_others
        self.actor_mlp = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.feature.initial_hidden(batch_size, device)

    def forward(self, obs_seq: torch.Tensor, h0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward through GRU for a sequence.

        Args:
            obs_seq: (T, B, obs_dim)
            h0: (1, B, hidden_dim)
        Returns:
            action_mean_seq: (T, B, action_dim)
            value_seq: (T, B, 1)
            hT: (1, B, hidden_dim)
        """

        T, B, _ = obs_seq.shape
        feat_seq, hT = self.feature.forward_seq(obs_seq, h0)  # (T,B,H)
        feat_flat = feat_seq.view(T * B, self.hidden_dim)

        mean_role, _ = self.role_predictor(feat_flat)
        role_probs = torch.sigmoid(mean_role)
        actor_in = torch.cat([feat_flat, role_probs], dim=-1)
        action_mean_flat = self.actor_mlp(actor_in)
        value_flat = self.critic(feat_flat)

        action_mean = action_mean_flat.view(T, B, self.action_dim)
        value = value_flat.view(T, B, 1)
        return action_mean, value, hT

    def step(self, obs: torch.Tensor, h: torch.Tensor):
        """Single-step forward for acting.

        Args:
            obs: (B, obs_dim)
            h: (1, B, hidden_dim)
        Returns:
            action_mean: (B, action_dim)
            value: (B, 1)
            h_new: (1, B, hidden_dim)
        """

        obs_seq = obs.unsqueeze(0)
        action_mean_seq, value_seq, h_new = self.forward(obs_seq, h)
        return action_mean_seq.squeeze(0), value_seq.squeeze(0), h_new


@dataclass
class RoleAwareRecurrentPPOConfig:
    """Config for RoleAwareRecurrentPPOAgent.

    Mirrors RecurrentPPOAgent's constructor arguments, plus:
        num_other_agents: 自分以外のエージェント数（4体環境では3）。
        num_agents: マルチエージェント環境での総エージェント数（4体環境では4）。
    """

    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str | None = None
    num_other_agents: int = 3
    num_agents: int = 4


class RoleAwareRecurrentPPOAgent:
    """Recurrent PPO agent with auxiliary role prediction.

    The public interface is compatible with RecurrentPPOAgent so that
    existing trainers (`SelfPlayRecurrentPPOTrainer` and
    `SelfPlayRecurrentMultiAgentTrainer`) can be reused as-is.
    """

    def __init__(self, config: RoleAwareRecurrentPPOConfig):
        if config.device is None:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = config.device

        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.num_agents = config.num_agents
        self.num_others = config.num_other_agents

        self.network = RoleAwareRecurrentActorCritic(
            config.obs_dim,
            config.action_dim,
            config.hidden_dim,
            num_others=config.num_other_agents,
        ).to(self.device)

        # Policy optimizer: feature + actor + critic
        policy_params = (
            list(self.network.feature.parameters())
            + list(self.network.actor_mlp.parameters())
            + list(self.network.critic.parameters())
            + [self.network.actor_logstd]
        )
        self.policy_optim = optim.Adam(policy_params, lr=config.lr)

        # Predictor optimizer: feature + role_predictor
        predictor_params = list(self.network.feature.parameters()) + list(
            self.network.role_predictor.parameters()
        )
        self.predictor_optim = optim.Adam(predictor_params, lr=config.lr)

        # For compatibility with existing save/load usage
        self.optimizer = self.policy_optim

        # For evaluation convenience
        self.eval_hidden = None
        self.last_role_probs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Acting helpers
    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return self.network.initial_hidden(batch_size, torch.device(self.device))

    def reset_eval_hidden(self):
        self.eval_hidden = None

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for single observation (used in test_policy)."""

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.eval_hidden is None:
            self.eval_hidden = self.init_hidden(batch_size=1)

        with torch.no_grad():
            # Manually run through feature + role predictor + actor for visibility
            feat, h_new = self.network.feature.step(obs_tensor, self.eval_hidden)
            mean_role, _ = self.network.role_predictor(feat)
            role_probs = torch.sigmoid(mean_role)  # (1, num_others)
            actor_in = torch.cat([feat, role_probs], dim=-1)
            action_mean = self.network.actor_mlp(actor_in)

            if deterministic:
                action = action_mean
            else:
                action_std = torch.exp(self.network.actor_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()

        self.eval_hidden = h_new
        self.last_role_probs = role_probs.cpu().numpy()[0]
        return action.cpu().numpy()[0]

    def get_action_and_value_batch(
        self,
        obs_batch: torch.Tensor,
        h: torch.Tensor,
        actions: torch.Tensor = None,
    ):
        """Used by the recurrent self-play trainer.

        Args:
            obs_batch: (B, obs_dim)
            h: (1, B, hidden_dim)
        Returns:
            action: (B, action_dim)
            log_prob: (B,)
            entropy: (B,)
            value: (B, 1)
            h_new: (1, B, hidden_dim)
        """

        action_mean, value, h_new = self.network.step(obs_batch, h)
        action_std = torch.exp(self.network.actor_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)

        if actions is None:
            actions = dist.sample()

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return actions, log_prob, entropy, value, h_new

    # ------------------------------------------------------------------
    # Supervised role prediction helpers
    # ------------------------------------------------------------------
    def _gaussian_nll(self, mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean Gaussian negative log-likelihood.

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
    # PPO update with auxiliary role loss
    # ------------------------------------------------------------------
    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        hiddens: np.ndarray,
        dones: np.ndarray,
        num_epochs: int = 10,
        batch_size: int = 64,
    ):
        """Update PPO parameters using recurrent BPTT over sequences.

        Args are identical to RecurrentPPOAgent.update.
        """

        # Convert to tensors
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        hiddens_tensor = torch.as_tensor(hiddens, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Normalize advantages over all timesteps and agents
        advantages_flat = advantages_tensor.view(-1)
        advantages_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        advantages_tensor = advantages_norm.view_as(advantages_tensor)

        T, B, _ = obs_tensor.shape

        # Build list of variable-length segments per agent, separated by dones
        segments = []  # list of (t_start, t_end, b)
        for b in range(B):
            t0 = 0
            while t0 < T:
                t = t0
                while True:
                    # stop at final timestep or when this step is terminal
                    if t >= T - 1 or dones_tensor[t, b].item() == 1.0:
                        t_end = t
                        break
                    t += 1

                segments.append((t0, t_end, b))
                t0 = t_end + 1

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_role_loss = 0.0
        num_updates = 0

        self.network.train()

        for _ in range(num_epochs):
            np.random.shuffle(segments)

            # Supervised role loss over this epoch
            role_loss_acc = 0.0
            role_count = 0

            for t_start, t_end, b in segments:
                L = t_end - t_start + 1

                # Slice sequence for a single agent (batch dimension = 1)
                obs_seq = obs_tensor[t_start : t_end + 1, b : b + 1, :]
                act_seq = actions_tensor[t_start : t_end + 1, b : b + 1, :]
                old_logp_seq = old_log_probs_tensor[t_start : t_end + 1, b : b + 1]
                adv_seq = advantages_tensor[t_start : t_end + 1, b : b + 1]
                ret_seq = returns_tensor[t_start : t_end + 1, b : b + 1]

                # Hidden state before the first step in this segment
                h0 = hiddens_tensor[t_start, b].unsqueeze(0).unsqueeze(1)  # (1,1,H)

                # Forward through GRU for the whole segment
                action_mean_seq, value_seq, _ = self.network(obs_seq, h0)

                action_std = torch.exp(self.network.actor_logstd)
                dist = torch.distributions.Normal(action_mean_seq, action_std)
                log_prob_seq = dist.log_prob(act_seq).sum(dim=-1)  # (L,1)
                entropy_seq = dist.entropy().sum(dim=-1)  # (L,1)

                # Flatten time and batch dims for loss computation
                log_prob_flat = log_prob_seq.view(-1)
                old_logp_flat = old_logp_seq.view(-1)
                adv_flat = adv_seq.view(-1)
                ret_flat = ret_seq.view(-1)
                value_flat = value_seq.view(-1)
                entropy_flat = entropy_seq.view(-1)

                ratio = torch.exp(log_prob_flat - old_logp_flat)
                surr1 = ratio * adv_flat
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_flat
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value_flat, ret_flat)
                entropy_loss = -entropy_flat.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.policy_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.policy_optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_flat.mean().item()
                num_updates += 1

                # Accumulate role loss for this segment (supervised, per-agent)
                # Agent index inferred from flattened index b (env-major, agent-inner)
                agent_index = int(b % self.num_agents)
                feat_seq, _ = self.network.feature.forward_seq(obs_seq, h0)
                feat = feat_seq.squeeze(1)  # (L,H)
                mean_role, log_std_role = self.network.role_predictor(feat)
                role_targets = self._role_labels_for_agent(agent_index, L)
                role_loss_seq = self._gaussian_nll(mean_role, log_std_role, role_targets)
                role_loss_acc += role_loss_seq
                role_count += 1

            if role_count > 0:
                role_loss = role_loss_acc / role_count
                self.predictor_optim.zero_grad()
                role_loss.backward()
                self.predictor_optim.step()
                total_role_loss += float(role_loss.item())

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "role_loss": total_role_loss / max(1.0, float(num_epochs)),
        }

    def save(self, path: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "policy_optim": self.policy_optim.state_dict(),
                "predictor_optim": self.predictor_optim.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        if "policy_optim" in checkpoint:
            self.policy_optim.load_state_dict(checkpoint["policy_optim"])
        if "predictor_optim" in checkpoint:
            self.predictor_optim.load_state_dict(checkpoint["predictor_optim"])
