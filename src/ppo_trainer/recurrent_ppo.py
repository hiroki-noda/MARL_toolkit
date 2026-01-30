"""Recurrent PPO agent and self-play trainer.

This module provides a GRU-based PPO variant that can maintain
hidden state across timesteps during acting. The current
implementation keeps the training update similar to the feedforward
PPO for simplicity, so temporal credit assignment through the GRU
is approximate. It is, however, suitable for experimenting with
recurrent policies in the existing self-play setup.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo_trainer.vec_env import VectorizedEnv, MultiAgentVectorizedEnv


class RecurrentActorCritic(nn.Module):
    """GRU-based Actor-Critic network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        obs_seq: torch.Tensor,
        h0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        x = self.fc(obs_seq.view(T * B, self.obs_dim))
        x = torch.tanh(x)
        x = x.view(T, B, self.hidden_dim)

        out, hT = self.gru(x, h0)

        action_mean = self.actor_mean(out)
        value = self.critic(out)
        return action_mean, value, hT

    def step(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
    ):
        """Single-step forward for acting.

        Args:
            obs: (B, obs_dim)
            h: (1, B, hidden_dim)
        Returns:
            action_mean: (B, action_dim)
            value: (B, 1)
            h_new: (1, B, hidden_dim)
        """

        obs_seq = obs.unsqueeze(0)  # (1, B, obs_dim)
        action_mean_seq, value_seq, h_new = self.forward(obs_seq, h)
        return action_mean_seq.squeeze(0), value_seq.squeeze(0), h_new


class RecurrentPPOAgent:
    """PPO agent with a GRU-based Actor-Critic network.

    Notes
    -----
    - During acting (self-play collection or evaluation), hidden
      state is carried across timesteps.
    - During the PPO update, we approximate the recurrent training
      by treating each step as independent, which keeps the code
      simple and stable in this codebase.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = None,
    ):
        """コンストラクタ

        Args:
            obs_dim: 観測ベクトルの次元数。
            action_dim: 連続行動ベクトルの次元数。
            hidden_dim: GRU ベース Actor-Critic の隠れ層ユニット数。
            lr: Adam オプティマイザの学習率。
            gamma: 割引率 γ。
            gae_lambda: GAE (Generalized Advantage Estimation) の λ。
            clip_epsilon: PPO のクリッピング係数 (ε)。
            value_coef: 価値損失の重み係数。
            entropy_coef: エントロピー正則化の重み係数。
            max_grad_norm: 勾配クリッピングの上限ノルム。
            device: 使用するデバイス。"cuda" / "cpu"。None の場合は自動判定。
        """
        if device is None:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.network = RecurrentActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # For evaluation convenience
        self.eval_hidden = None

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
            action_mean, _, h_new = self.network.step(obs_tensor, self.eval_hidden)
            if deterministic:
                action = action_mean
            else:
                action_std = torch.exp(self.network.actor_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()

        self.eval_hidden = h_new
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
    # PPO update (on-policy, sequence-wise with BPTT over segments)
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

        Args:
            obs: (T, B, obs_dim)
            actions: (T, B, action_dim)
            old_log_probs: (T, B)
            advantages: (T, B)
            returns: (T, B)
            hiddens: (T, B, hidden_dim)  # hidden *before* each step
            dones: (T, B)  # 1.0 where episode ended after this step
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
        num_updates = 0

        self.network.train()

        for _ in range(num_epochs):
            np.random.shuffle(segments)

            for t_start, t_end, b in segments:
                # Sequence length for this segment
                L = t_end - t_start + 1

                # Slice sequence for a single agent (batch dimension = 1)
                obs_seq = obs_tensor[t_start : t_end + 1, b : b + 1, :]
                act_seq = actions_tensor[t_start : t_end + 1, b : b + 1, :]
                old_logp_seq = old_log_probs_tensor[t_start : t_end + 1, b : b + 1]
                adv_seq = advantages_tensor[t_start : t_end + 1, b : b + 1]
                ret_seq = returns_tensor[t_start : t_end + 1, b : b + 1]

                # Hidden state before the first step in this segment
                h0 = hiddens_tensor[t_start, b].unsqueeze(0).unsqueeze(1)  # (1, 1, hidden_dim)

                # Forward through GRU for the whole segment
                action_mean_seq, value_seq, _ = self.network(obs_seq, h0)

                action_std = torch.exp(self.network.actor_logstd)
                dist = torch.distributions.Normal(action_mean_seq, action_std)
                log_prob_seq = dist.log_prob(act_seq).sum(dim=-1)  # (L, 1)
                entropy_seq = dist.entropy().sum(dim=-1)  # (L, 1)

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

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_flat.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
        }

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class SelfPlayRecurrentPPOTrainer:
    """Self-play trainer that uses RecurrentPPOAgent.

    This is structurally similar to the existing SelfPlayTrainer but
    manages GRU hidden states across timesteps for both agents.
    """

    def __init__(
        self,
        agent: RecurrentPPOAgent,
        envs: VectorizedEnv,
        steps_per_update: int = 2048,
        num_epochs: int = 10,
        batch_size: int = 64,
    ):
        self.agent = agent
        self.envs = envs
        self.steps_per_update = steps_per_update
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_envs = envs.num_envs

    def train_step(self):
        obs1, obs2 = self.envs.reset()

        B = self.num_envs * 2
        h = self.agent.init_hidden(B)

        # Time-major buffers: each entry is shape (B, ...)
        obs_seq = []
        actions_seq = []
        logp_seq = []
        rewards_seq = []
        dones_seq = []
        values_seq = []
        hiddens_seq = []  # hidden state *before* each step

        ep_returns1 = np.zeros(self.num_envs, dtype=np.float32)
        ep_returns2 = np.zeros(self.num_envs, dtype=np.float32)
        completed_returns1 = []
        completed_returns2 = []
        completed_winners = []

        for _ in range(self.steps_per_update):
            # Build batch of observations for both agents
            obs_batch = np.concatenate([obs1, obs2], axis=0)  # (2*num_envs, obs_dim)
            obs_batch_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.agent.device)

            # Save hidden state before acting so that update() can
            # reconstruct the same recurrent dynamics.
            h_before = h.clone().detach()

            with torch.no_grad():
                acts_batch, logp_batch, _, values_batch, h = self.agent.get_action_and_value_batch(
                    obs_batch_t, h
                )

            acts_batch_np = acts_batch.cpu().numpy()  # (B, action_dim)
            values_batch_np = values_batch.squeeze(-1).cpu().numpy()  # (B,)
            logp_batch_np = logp_batch.cpu().numpy()  # (B,)
            h_before_np = h_before.squeeze(0).cpu().numpy()  # (B, hidden_dim)

            # Split actions for stepping environments
            actions1 = acts_batch_np[: self.num_envs]
            actions2 = acts_batch_np[self.num_envs :]

            # Step environments
            (next_obs1, next_obs2), (rewards1, rewards2), dones_arr, infos, _ = self.envs.step(
                actions1, actions2
            )

            ep_returns1 += rewards1
            ep_returns2 += rewards2

            # Per-agent rewards/dones for training (length B)
            reward_batch = np.concatenate([rewards1, rewards2], axis=0)
            done_batch = np.concatenate([dones_arr, dones_arr], axis=0).astype(np.float32)

            # Store time-step data
            obs_seq.append(obs_batch)
            actions_seq.append(acts_batch_np)
            logp_seq.append(logp_batch_np)
            rewards_seq.append(reward_batch)
            dones_seq.append(done_batch)
            values_seq.append(values_batch_np)
            hiddens_seq.append(h_before_np)

            # Episode statistics and hidden reset per environment
            for env_idx, done_flag in enumerate(dones_arr):
                if done_flag:
                    completed_returns1.append(ep_returns1[env_idx])
                    completed_returns2.append(ep_returns2[env_idx])

                    winner = infos[env_idx].get("winner", "none")
                    if winner == "agent1":
                        completed_winners.append(1)
                    elif winner == "agent2":
                        completed_winners.append(2)
                    else:
                        completed_winners.append(0)

                    ep_returns1[env_idx] = 0.0
                    ep_returns2[env_idx] = 0.0

                    # Reset hidden state for this environment (both agents)
                    h[:, env_idx, :] = 0.0
                    h[:, env_idx + self.num_envs, :] = 0.0

            obs1 = next_obs1
            obs2 = next_obs2

        # Compute last state values for GAE (agent1/agent2 separately)
        with torch.no_grad():
            next_values1 = []
            next_values2 = []
            for i in range(self.num_envs):
                obs1_tensor = torch.FloatTensor(obs1[i]).unsqueeze(0).to(self.agent.device)
                obs2_tensor = torch.FloatTensor(obs2[i]).unsqueeze(0).to(self.agent.device)

                h1 = self.agent.init_hidden(1)
                h2 = self.agent.init_hidden(1)
                _, v1, _ = self.agent.network.step(obs1_tensor, h1)
                _, v2, _ = self.agent.network.step(obs2_tensor, h2)
                next_values1.append(v1.item())
                next_values2.append(v2.item())

        # Convert buffers to time-major arrays
        obs_seq = np.stack(obs_seq, axis=0)  # (T, B, obs_dim)
        actions_seq = np.stack(actions_seq, axis=0)  # (T, B, action_dim)
        logp_seq = np.stack(logp_seq, axis=0)  # (T, B)
        rewards_seq = np.stack(rewards_seq, axis=0)  # (T, B)
        dones_seq = np.stack(dones_seq, axis=0)  # (T, B)
        values_seq = np.stack(values_seq, axis=0)  # (T, B)
        hiddens_seq = np.stack(hiddens_seq, axis=0)  # (T, B, hidden_dim)

        # Next values per agent (agent1 then agent2)
        next_values = np.array(next_values1 + next_values2)  # (B,)

        # GAE per agent over time
        T = obs_seq.shape[0]
        B = obs_seq.shape[1]
        advantages = np.zeros_like(rewards_seq)
        for b in range(B):
            last_gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = next_values[b]
                else:
                    next_val = values_seq[t + 1, b]

                done = dones_seq[t, b]
                delta = (
                    rewards_seq[t, b]
                    + self.agent.gamma * next_val * (1.0 - done)
                    - values_seq[t, b]
                )
                last_gae = delta + self.agent.gamma * self.agent.gae_lambda * (1.0 - done) * last_gae
                advantages[t, b] = last_gae

        returns = advantages + values_seq

        metrics = self.agent.update(
            obs_seq,
            actions_seq,
            logp_seq,
            advantages,
            returns,
            hiddens_seq,
            dones_seq,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )

        num_episodes = len(completed_winners)
        if num_episodes > 0:
            winners_arr = np.asarray(completed_winners, dtype=np.int32)
            win_rate_agent1 = float(np.mean(winners_arr == 1))
            win_rate_agent2 = float(np.mean(winners_arr == 2))
            draw_rate = float(np.mean(winners_arr == 0))
            avg_ep_return_agent1 = float(np.mean(completed_returns1))
            avg_ep_return_agent2 = float(np.mean(completed_returns2))
        else:
            win_rate_agent1 = 0.0
            win_rate_agent2 = 0.0
            draw_rate = 0.0
            avg_ep_return_agent1 = 0.0
            avg_ep_return_agent2 = 0.0

        avg_reward = float(rewards_seq.mean()) if rewards_seq.size > 0 else 0.0

        stats = {
            "episodes_in_batch": float(num_episodes),
            "win_rate_agent1": win_rate_agent1,
            "win_rate_agent2": win_rate_agent2,
            "draw_rate": draw_rate,
            "avg_episode_return_agent1": avg_ep_return_agent1,
            "avg_episode_return_agent2": avg_ep_return_agent2,
        }

        metrics.update(stats)

        return metrics, avg_reward


class SelfPlayRecurrentMultiAgentTrainer:
    """Self-play trainer for recurrent PPO in multi-agent environments.

    This trainer mirrors `SelfPlayMultiAgentTrainer` but uses a
    `RecurrentPPOAgent` and explicitly tracks GRU hidden state for
    each (env, agent) pair across timesteps.
    """

    def __init__(
        self,
        agent: RecurrentPPOAgent,
        envs: MultiAgentVectorizedEnv,
        num_agents: int,
        steps_per_update: int = 2048,
        num_epochs: int = 10,
        batch_size: int = 64,
        team1_indices: Optional[Sequence[int]] = None,
        team2_indices: Optional[Sequence[int]] = None,
    ):
        self.agent = agent
        self.envs = envs
        self.num_agents = num_agents
        self.steps_per_update = steps_per_update
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_envs = envs.num_envs

        self.team1_indices = list(team1_indices) if team1_indices is not None else None
        self.team2_indices = list(team2_indices) if team2_indices is not None else None

    def train_step(self):
        # obs: (num_envs, num_agents, obs_dim)
        obs = self.envs.reset()

        B = self.num_envs * self.num_agents
        h = self.agent.init_hidden(B)

        # Time-major buffers: each entry uses env-major, agent-inner flattening
        obs_seq: List[np.ndarray] = []      # (B, obs_dim)
        actions_seq: List[np.ndarray] = []  # (B, action_dim)
        logp_seq: List[np.ndarray] = []     # (B,)
        rewards_seq: List[np.ndarray] = []  # (B,)
        dones_seq: List[np.ndarray] = []    # (B,)
        values_seq: List[np.ndarray] = []   # (B,)
        hiddens_seq: List[np.ndarray] = []  # (B, hidden_dim)

        # Episode statistics per env & team
        ep_returns = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        completed_returns_team1: List[float] = []
        completed_returns_team2: List[float] = []
        completed_winners: List[int] = []  # 1: team1, 2: team2, 0: draw/none

        for _ in range(self.steps_per_update):
            # Flatten observations to (B, obs_dim) in env-major order
            obs_batch = obs.reshape(B, obs.shape[-1])
            obs_batch_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.agent.device)

            # Hidden state before acting (for training update)
            h_before = h.clone().detach()

            with torch.no_grad():
                acts_batch, logp_batch, _, values_batch, h = self.agent.get_action_and_value_batch(
                    obs_batch_t, h
                )

            acts_batch_np = acts_batch.cpu().numpy()  # (B, action_dim)
            values_batch_np = values_batch.squeeze(-1).cpu().numpy()  # (B,)
            logp_batch_np = logp_batch.cpu().numpy()  # (B,)
            h_before_np = h_before.squeeze(0).cpu().numpy()  # (B, hidden_dim)

            # Reshape actions back to (num_envs, num_agents, action_dim)
            actions_step = acts_batch_np.reshape(self.num_envs, self.num_agents, -1)

            # Step environments
            next_obs, rewards_step, dones_env, infos, _ = self.envs.step(actions_step)

            # Update per-env, per-agent returns
            ep_returns += rewards_step

            # Flatten rewards/dones for training
            reward_batch = rewards_step.reshape(B)
            done_batch = np.repeat(dones_env.astype(np.float32), self.num_agents)

            # Store time-step data (env-major flattening)
            obs_seq.append(obs_batch)
            actions_seq.append(acts_batch_np)
            logp_seq.append(logp_batch_np)
            rewards_seq.append(reward_batch)
            dones_seq.append(done_batch)
            values_seq.append(values_batch_np)
            hiddens_seq.append(h_before_np)

            # Episode statistics & hidden resets for finished envs
            for env_idx, done_flag in enumerate(dones_env):
                if not done_flag:
                    continue

                if self.team1_indices is not None and self.team2_indices is not None:
                    team1_ret = float(np.mean(ep_returns[env_idx, self.team1_indices]))
                    team2_ret = float(np.mean(ep_returns[env_idx, self.team2_indices]))
                    completed_returns_team1.append(team1_ret)
                    completed_returns_team2.append(team2_ret)

                winner = infos[env_idx].get("winner", "none")
                if winner == "agent1":
                    completed_winners.append(1)
                elif winner == "agent2":
                    completed_winners.append(2)
                else:
                    completed_winners.append(0)

                ep_returns[env_idx, :] = 0.0

                # Reset hidden state for all agents in this environment
                for agent_idx in range(self.num_agents):
                    b = env_idx * self.num_agents + agent_idx
                    h[:, b, :] = 0.0

            obs = next_obs

        # Compute last state values for each (env, agent)
        next_values_list: List[float] = []
        with torch.no_grad():
            for env_idx in range(self.num_envs):
                for agent_idx in range(self.num_agents):
                    o = obs[env_idx, agent_idx]
                    obs_tensor = torch.as_tensor(o, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
                    h0 = self.agent.init_hidden(1)
                    _, v, _ = self.agent.network.step(obs_tensor, h0)
                    next_values_list.append(v.item())

        next_values = np.asarray(next_values_list, dtype=np.float32)  # (B,)

        # Convert buffers to time-major arrays
        obs_seq_arr = np.stack(obs_seq, axis=0)      # (T, B, obs_dim)
        actions_seq_arr = np.stack(actions_seq, axis=0)  # (T, B, action_dim)
        logp_seq_arr = np.stack(logp_seq, axis=0)    # (T, B)
        rewards_seq_arr = np.stack(rewards_seq, axis=0)  # (T, B)
        dones_seq_arr = np.stack(dones_seq, axis=0)  # (T, B)
        values_seq_arr = np.stack(values_seq, axis=0)    # (T, B)
        hiddens_seq_arr = np.stack(hiddens_seq, axis=0)  # (T, B, hidden_dim)

        T, B = rewards_seq_arr.shape
        advantages = np.zeros_like(rewards_seq_arr)

        # GAE per (env, agent) index b
        for b in range(B):
            last_gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = next_values[b]
                else:
                    next_val = values_seq_arr[t + 1, b]

                done = dones_seq_arr[t, b]
                delta = (
                    rewards_seq_arr[t, b]
                    + self.agent.gamma * next_val * (1.0 - done)
                    - values_seq_arr[t, b]
                )
                last_gae = (
                    delta
                    + self.agent.gamma * self.agent.gae_lambda * (1.0 - done) * last_gae
                )
                advantages[t, b] = last_gae

        returns = advantages + values_seq_arr

        metrics = self.agent.update(
            obs_seq_arr,
            actions_seq_arr,
            logp_seq_arr,
            advantages,
            returns,
            hiddens_seq_arr,
            dones_seq_arr,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )

        num_episodes = len(completed_winners)
        if num_episodes > 0 and self.team1_indices is not None and self.team2_indices is not None:
            winners_arr = np.asarray(completed_winners, dtype=np.int32)
            win_rate_team1 = float(np.mean(winners_arr == 1))
            win_rate_team2 = float(np.mean(winners_arr == 2))
            draw_rate = float(np.mean(winners_arr == 0))
            avg_ep_return_team1 = float(np.mean(completed_returns_team1))
            avg_ep_return_team2 = float(np.mean(completed_returns_team2))
        else:
            win_rate_team1 = 0.0
            win_rate_team2 = 0.0
            draw_rate = 0.0
            avg_ep_return_team1 = 0.0
            avg_ep_return_team2 = 0.0

        avg_reward = float(rewards_seq_arr.mean()) if rewards_seq_arr.size > 0 else 0.0

        stats = {
            "episodes_in_batch": float(num_episodes),
            "win_rate_agent1": win_rate_team1,
            "win_rate_agent2": win_rate_team2,
            "draw_rate": draw_rate,
            "avg_episode_return_agent1": avg_ep_return_team1,
            "avg_episode_return_agent2": avg_ep_return_team2,
        }

        metrics.update(stats)

        return metrics, avg_reward
