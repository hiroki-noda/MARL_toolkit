"""Role-aware Recurrent SAC agent.

This variant extends Recurrent SAC with an auxiliary task that
predicts the roles (chaser vs evader) of the *other* three agents
in the 4-agent adversarial_tag_4p environment.

The design mirrors RoleAwareRecurrentTD3Agent:
- A GRU-based feature extractor processes observations.
- A role predictor head outputs Gaussian parameters over role labels
  for the three other agents.
- A Gaussian policy head conditions on the GRU features and predicted
  role probabilities to produce actions.

Training combines:
- Standard recurrent SAC actor-critic updates (off-policy).
- Supervised learning on role labels, backpropagated through the GRU
  and predictor (but not the policy head).
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
from ppo_trainer.role_rtd3 import FeatureGRU, RolePredictor, LOG_STD_MIN, LOG_STD_MAX
from ppo_trainer.recurrent_sac import QNetwork, RecurrentSACConfig


@dataclass
class RoleAwareRecurrentSACConfig(RecurrentSACConfig):
    """Config for the role-aware recurrent SAC agent.

    Inherits standard RecurrentSACConfig and adds:

    Attributes:
        num_other_agents: 自分以外のエージェント数（adversarial_tag_4p では常に3）。
    """

    num_other_agents: int = 3


class RoleAwareRecurrentSACAgent:
    """Recurrent SAC agent with auxiliary role prediction.

    - GRU features are shared between actor and role predictor.
    - Critics are identical to RecurrentSACAgent and see only
      the raw observation and action.
    - Actor conditions on GRU features and predicted role
      probabilities.
    """

    def __init__(self, config: RoleAwareRecurrentSACConfig):
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.tau = config.tau
        self.num_others = config.num_other_agents

        # Shared recurrent feature extractor
        self.feature = FeatureGRU(config.obs_dim, config.hidden_dim).to(self.device)

        # Role predictor
        self.role_predictor = RolePredictor(config.hidden_dim, self.num_others).to(self.device)

        # Policy head: from [feat, role_probs] -> mean/log_std
        input_dim = config.hidden_dim + self.num_others
        self.policy_mean = nn.Linear(input_dim, config.action_dim).to(self.device)
        self.policy_log_std = nn.Linear(input_dim, config.action_dim).to(self.device)

        # Q networks and targets (same as RecurrentSAC)
        self.q1 = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2 = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q1_target = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2_target = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        # Critics
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=config.lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=config.lr)
        # Policy (GRU feature + policy head)
        policy_params = list(self.feature.parameters()) + list(self.policy_mean.parameters()) + list(
            self.policy_log_std.parameters()
        )
        self.policy_optim = optim.Adam(policy_params, lr=config.lr)
        # Predictor (GRU feature + role predictor)
        predictor_params = list(self.feature.parameters()) + list(self.role_predictor.parameters())
        self.predictor_optim = optim.Adam(predictor_params, lr=config.lr)

        # Entropy temperature
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

        # For compatibility with existing trainer code
        self.eval_hidden: torch.Tensor | None = None
        self.last_role_probs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _alpha(self) -> torch.Tensor:
        if self.log_alpha is None:
            return torch.tensor(self.alpha, device=self.device)
        return self.log_alpha.exp()

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

    def _policy_from_feat(self, feat: torch.Tensor):
        """Compute policy mean and log_std from features.

        Args:
            feat: (N, hidden_dim)
        Returns:
            mean: (N, action_dim)
            log_std: (N, action_dim)
        """

        mean_role, _ = self.role_predictor(feat)
        role_probs = torch.sigmoid(mean_role)
        x = torch.cat([feat, role_probs], dim=-1)
        mean = self.policy_mean(x)
        log_std = self.policy_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def reset_eval_hidden(self):
        self.eval_hidden = None

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.eval_hidden is None:
            self.eval_hidden = self.feature.initial_hidden(1, self.device)

        with torch.no_grad():
            feat, h_new = self.feature.step(obs_tensor, self.eval_hidden)
            mean_role, _ = self.role_predictor(feat)
            role_probs = torch.sigmoid(mean_role)  # (1, num_others)

            x = torch.cat([feat, role_probs], dim=-1)
            mean = self.policy_mean(x)
            log_std = self.policy_log_std(x)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = log_std.exp()

            if deterministic:
                action = torch.tanh(mean)
            else:
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)

        self.eval_hidden = h_new
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
            return {
                "q1_loss": 0.0,
                "q2_loss": 0.0,
                "policy_loss": 0.0,
                "alpha": float(self._alpha()),
                "role_loss": 0.0,
            }

        total_q1_loss = 0.0
        total_q2_loss = 0.0
        total_policy_loss = 0.0
        total_role_loss = 0.0

        for _ in range(gradient_steps):
            seq_batch: List[SequenceSample] = replay_buffer.sample(batch_size, max_seq_len)
            if not seq_batch:
                break

            # ---------------- Critic update (flattened transitions) ----------------
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
                # Next actions from current policy using GRU features
                # next_obs: (N, obs_dim) -> (T=N, B=1, obs_dim)
                N = next_obs.shape[0]
                obs_seq = next_obs.unsqueeze(1)
                h0 = self.feature.initial_hidden(batch_size=1, device=self.device)
                feat_seq, _ = self.feature.forward_seq(obs_seq, h0)
                feat = feat_seq.squeeze(1)  # (N, hidden_dim)
                mean, log_std = self._policy_from_feat(feat)
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
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

            total_q1_loss += float(q1_loss.item())
            total_q2_loss += float(q2_loss.item())

            # ---------------- Supervised role prediction update ----------------
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

            # ---------------- Policy (actor) update with BPTT ----------------
            policy_loss_acc = 0.0
            alpha_loss_acc = 0.0
            count = 0

            for seq in seq_batch:
                T = seq.obs.shape[0]
                obs_seq = torch.as_tensor(seq.obs, device=self.device).unsqueeze(1)  # (T,1,obs_dim)
                h0 = self.feature.initial_hidden(1, self.device)
                feat_seq, _ = self.feature.forward_seq(obs_seq, h0)
                feat = feat_seq.squeeze(1)  # (T, hidden_dim)

                mean, log_std = self._policy_from_feat(feat)
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                action_seq = torch.tanh(x_t)
                log_prob_seq = normal.log_prob(x_t) - torch.log(1 - action_seq.pow(2) + 1e-6)
                log_prob_seq = log_prob_seq.sum(dim=-1, keepdim=True)  # (T,1)

                obs_flat = obs_seq.view(T, -1)
                act_flat = action_seq.view(T, -1)
                q1_new = self.q1(obs_flat, act_flat)
                q2_new = self.q2(obs_flat, act_flat)
                q_new = torch.min(q1_new, q2_new)

                alpha = self._alpha()
                policy_loss_seq = (alpha * log_prob_seq - q_new).mean()
                policy_loss_acc += policy_loss_seq

                if self.log_alpha is not None:
                    alpha_loss_seq = -(self.log_alpha * (log_prob_seq.detach() + self.target_entropy)).mean()
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

        steps = max(1, gradient_steps)
        return {
            "q1_loss": total_q1_loss / steps,
            "q2_loss": total_q2_loss / steps,
            "policy_loss": total_policy_loss / steps,
            "alpha": float(self._alpha()),
            "role_loss": total_role_loss / max(1.0, float(gradient_steps)),
        }

    def save(self, path: str):
        """Save model parameters (feature GRU, role predictor, policy head, Q networks)."""
        torch.save(
            {
                "feature": self.feature.state_dict(),
                "role_predictor": self.role_predictor.state_dict(),
                "policy_mean": self.policy_mean.state_dict(),
                "policy_log_std": self.policy_log_std.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.feature.load_state_dict(checkpoint["feature"])
        self.role_predictor.load_state_dict(checkpoint["role_predictor"])
        self.policy_mean.load_state_dict(checkpoint["policy_mean"])
        self.policy_log_std.load_state_dict(checkpoint["policy_log_std"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
