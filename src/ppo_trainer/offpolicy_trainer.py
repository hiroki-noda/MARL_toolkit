"""Off-policy self-play trainers for continuous-control algorithms.

Includes trainers for both two-agent and multi-agent (N-agent)
settings, for algorithms such as SAC and TD3 that learn from a
replay buffer instead of on-policy rollouts.
"""

from typing import Any, Dict, List, Sequence, Optional

import numpy as np
import torch

from ppo_trainer.vec_env import VectorizedEnv, MultiAgentVectorizedEnv
from ppo_trainer.sequence_replay import SequenceReplayBuffer


class SelfPlayOffPolicyTrainer:
    """Self-play trainer for off-policy algorithms (e.g., SAC, TD3).

    This trainer assumes a symmetric two-agent setting where a single
    agent instance controls both players (agent1/agent2). Transitions
    from both agents are stored into a shared replay buffer.
    """

    def __init__(
        self,
        agent: Any,
        envs: VectorizedEnv,
        replay_buffer: Any,
        steps_per_update: int = 2048,
        batch_size: int = 256,
        gradient_steps: int = 1,
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.replay_buffer = replay_buffer
        self.steps_per_update = steps_per_update
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.num_envs = envs.num_envs

    def train_step(self) -> (Dict[str, float], float):
        """Collect experience and run off-policy updates once.

        Returns:
            metrics: dictionary of training statistics
            avg_reward: average per-step reward of agent1 over the batch
        """

        obs1, obs2 = self.envs.reset()

        # Episode statistics
        ep_returns1 = np.zeros(self.num_envs, dtype=np.float32)
        ep_returns2 = np.zeros(self.num_envs, dtype=np.float32)
        completed_returns1 = []
        completed_returns2 = []
        completed_winners = []  # 1: agent1, 2: agent2, 0: draw/none

        all_step_rewards1 = []

        for _ in range(self.steps_per_update):
            # Select actions for both agents using the same policy
            actions1 = []
            actions2 = []

            for i in range(self.num_envs):
                o1 = obs1[i]
                o2 = obs2[i]

                a1 = self.agent.select_action(o1, deterministic=False)
                a2 = self.agent.select_action(o2, deterministic=False)

                actions1.append(a1)
                actions2.append(a2)

            actions1 = np.asarray(actions1, dtype=np.float32)
            actions2 = np.asarray(actions2, dtype=np.float32)

            # Environment step
            (next_obs1, next_obs2), (rewards1, rewards2), dones_arr, infos, _ = self.envs.step(actions1, actions2)

            all_step_rewards1.append(rewards1.copy())

            # Store transitions for both agents in replay buffer
            for env_idx in range(self.num_envs):
                done_flag = float(dones_arr[env_idx])

                # Agent 1 transition
                self.replay_buffer.add(
                    obs1[env_idx],
                    actions1[env_idx],
                    np.array([rewards1[env_idx]], dtype=np.float32),
                    next_obs1[env_idx],
                    np.array([done_flag], dtype=np.float32),
                )

                # Agent 2 transition
                self.replay_buffer.add(
                    obs2[env_idx],
                    actions2[env_idx],
                    np.array([rewards2[env_idx]], dtype=np.float32),
                    next_obs2[env_idx],
                    np.array([done_flag], dtype=np.float32),
                )

            # Update episode statistics
            ep_returns1 += rewards1
            ep_returns2 += rewards2

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

            obs1 = next_obs1
            obs2 = next_obs2

        # Run off-policy updates
        metrics = self.agent.update(
            replay_buffer=self.replay_buffer,
            batch_size=self.batch_size,
            gradient_steps=self.gradient_steps,
        )

        # Aggregate episode statistics
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

        step_rewards1 = np.concatenate(all_step_rewards1, axis=0)
        avg_reward = float(step_rewards1.mean()) if step_rewards1.size > 0 else 0.0

        stats: Dict[str, float] = {
            "episodes_in_batch": float(num_episodes),
            "win_rate_agent1": win_rate_agent1,
            "win_rate_agent2": win_rate_agent2,
            "draw_rate": draw_rate,
            "avg_episode_return_agent1": avg_ep_return_agent1,
            "avg_episode_return_agent2": avg_ep_return_agent2,
            "avg_reward": avg_reward,
        }

        metrics.update(stats)

        return metrics, avg_reward


class SelfPlayOffPolicyRecurrentTrainer:
    """Self-play trainer for recurrent off-policy algorithms (e.g., Recurrent SAC/TD3).

    Collects rollouts and converts them into per-env, per-agent sequences,
    which are then stored into a SequenceReplayBuffer.
    """

    def __init__(
        self,
        agent: Any,
        envs: VectorizedEnv,
        replay_buffer: SequenceReplayBuffer,
        steps_per_update: int = 2048,
        batch_size: int = 4,
        max_seq_len: int = 64,
        gradient_steps: int = 1,
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.replay_buffer = replay_buffer
        self.steps_per_update = steps_per_update
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.gradient_steps = gradient_steps
        self.num_envs = envs.num_envs

    def train_step(self) -> (Dict[str, float], float):
        obs1, obs2 = self.envs.reset()

        # Rollout storage (time-major)
        obs1_seq = []
        obs2_seq = []
        actions1_seq = []
        actions2_seq = []
        rewards1_seq = []
        rewards2_seq = []
        dones_seq = []
        next_obs1_seq = []
        next_obs2_seq = []

        ep_returns1 = np.zeros(self.num_envs, dtype=np.float32)
        ep_returns2 = np.zeros(self.num_envs, dtype=np.float32)
        completed_returns1 = []
        completed_returns2 = []
        completed_winners = []

        for _ in range(self.steps_per_update):
            # Agents act independently using the same recurrent policy; for
            # simplicity, we do not track hidden state here and rely on
            # sequence training to capture temporal structure.
            actions1 = []
            actions2 = []
            for i in range(self.num_envs):
                a1 = self.agent.select_action(obs1[i], deterministic=False)
                a2 = self.agent.select_action(obs2[i], deterministic=False)
                actions1.append(a1)
                actions2.append(a2)

            actions1 = np.asarray(actions1, dtype=np.float32)
            actions2 = np.asarray(actions2, dtype=np.float32)

            (next_obs1, next_obs2), (rewards1, rewards2), dones_arr, infos, _ = self.envs.step(
                actions1, actions2
            )

            # Store time step
            obs1_seq.append(obs1)
            obs2_seq.append(obs2)
            actions1_seq.append(actions1)
            actions2_seq.append(actions2)
            rewards1_seq.append(rewards1)
            rewards2_seq.append(rewards2)
            dones_seq.append(dones_arr.astype(np.float32))
            next_obs1_seq.append(next_obs1)
            next_obs2_seq.append(next_obs2)

            ep_returns1 += rewards1
            ep_returns2 += rewards2

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

            obs1 = next_obs1
            obs2 = next_obs2

        # Convert rollout buffers to arrays
        obs1_seq = np.stack(obs1_seq, axis=0)       # (T, N, obs_dim)
        obs2_seq = np.stack(obs2_seq, axis=0)
        actions1_seq = np.stack(actions1_seq, axis=0)  # (T, N, act_dim)
        actions2_seq = np.stack(actions2_seq, axis=0)
        rewards1_seq = np.stack(rewards1_seq, axis=0)  # (T, N)
        rewards2_seq = np.stack(rewards2_seq, axis=0)
        dones_seq = np.stack(dones_seq, axis=0)        # (T, N)
        next_obs1_seq = np.stack(next_obs1_seq, axis=0)
        next_obs2_seq = np.stack(next_obs2_seq, axis=0)

        T, N, _ = obs1_seq.shape

        # Build per-env, per-agent sequences and push into replay buffer
        for env_idx in range(self.num_envs):
            for agent_id in (1, 2):
                if agent_id == 1:
                    obs_env = obs1_seq[:, env_idx, :]
                    act_env = actions1_seq[:, env_idx, :]
                    rew_env = rewards1_seq[:, env_idx]
                    next_obs_env = next_obs1_seq[:, env_idx, :]
                else:
                    obs_env = obs2_seq[:, env_idx, :]
                    act_env = actions2_seq[:, env_idx, :]
                    rew_env = rewards2_seq[:, env_idx]
                    next_obs_env = next_obs2_seq[:, env_idx, :]

                done_env = dones_seq[:, env_idx]

                # Split into segments at episode boundaries
                start = 0
                for t in range(T):
                    if done_env[t] == 1.0:
                        end = t
                        if end >= start:
                            self.replay_buffer.add_sequence(
                                obs_env[start : end + 1],
                                act_env[start : end + 1],
                                rew_env[start : end + 1],
                                next_obs_env[start : end + 1],
                                done_env[start : end + 1],
                            )
                        start = t + 1
                # Tail fragment (if any)
                if start < T:
                    self.replay_buffer.add_sequence(
                        obs_env[start:T],
                        act_env[start:T],
                        rew_env[start:T],
                        next_obs_env[start:T],
                        done_env[start:T],
                    )

        # Run recurrent off-policy updates
        metrics = self.agent.update(
            replay_buffer=self.replay_buffer,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            gradient_steps=self.gradient_steps,
        )

        # Episode statistics
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

        avg_reward = float(rewards1_seq.mean()) if rewards1_seq.size > 0 else 0.0

        stats: Dict[str, float] = {
            "episodes_in_batch": float(num_episodes),
            "win_rate_agent1": win_rate_agent1,
            "win_rate_agent2": win_rate_agent2,
            "draw_rate": draw_rate,
            "avg_episode_return_agent1": avg_ep_return_agent1,
            "avg_episode_return_agent2": avg_ep_return_agent2,
            "avg_reward": avg_reward,
        }

        metrics.update(stats)

        return metrics, avg_reward


class SelfPlayOffPolicyMultiAgentTrainer:
    """Off-policy self-play trainer for N-agent symmetric games (feed-forward).

    A single agent instance is shared across all num_agents agents in
    each environment. Transitions for every (env, agent) pair are
    stored into a shared replay buffer.

    This is used for adversarial_tag_4p with SAC/TD3 (non-recurrent).
    """

    def __init__(
        self,
        agent: Any,
        envs: MultiAgentVectorizedEnv,
        replay_buffer: Any,
        steps_per_update: int = 2048,
        batch_size: int = 256,
        gradient_steps: int = 1,
        team1_indices: Optional[Sequence[int]] = None,
        team2_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.replay_buffer = replay_buffer
        self.steps_per_update = steps_per_update
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.num_envs = envs.num_envs
        self.num_agents = envs.num_agents

        self.team1_indices = list(team1_indices) if team1_indices is not None else None
        self.team2_indices = list(team2_indices) if team2_indices is not None else None

    def train_step(self) -> (Dict[str, float], float):
        obs = self.envs.reset()  # (N_env, N_agent, obs_dim)

        ep_returns = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        completed_returns_team1: List[float] = []
        completed_returns_team2: List[float] = []
        completed_winners: List[int] = []

        all_step_rewards_team1: List[np.ndarray] = []

        for _ in range(self.steps_per_update):
            actions = np.zeros_like(obs[..., :2], dtype=np.float32)

            # 全env×agentに対して行動をサンプリング
            for env_idx in range(self.num_envs):
                for agent_idx in range(self.num_agents):
                    a = self.agent.select_action(obs[env_idx, agent_idx], deterministic=False)
                    actions[env_idx, agent_idx] = a

            next_obs, rewards, dones_arr, infos, _ = self.envs.step(actions)

            # すべての (env, agent) ペアの遷移をリプレイバッファに追加
            for env_idx in range(self.num_envs):
                done_flag = float(dones_arr[env_idx])
                for agent_idx in range(self.num_agents):
                    self.replay_buffer.add(
                        obs[env_idx, agent_idx],
                        actions[env_idx, agent_idx],
                        np.array([rewards[env_idx, agent_idx]], dtype=np.float32),
                        next_obs[env_idx, agent_idx],
                        np.array([done_flag], dtype=np.float32),
                    )

            ep_returns += rewards

            # チーム単位の1ステップ平均報酬
            if self.team1_indices is not None and self.team2_indices is not None:
                team1_step = rewards[:, self.team1_indices].mean(axis=1)
                all_step_rewards_team1.append(team1_step)

            # エピソード終了時の統計
            for env_idx, done_flag in enumerate(dones_arr):
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

            obs = next_obs

        metrics = self.agent.update(
            replay_buffer=self.replay_buffer,
            batch_size=self.batch_size,
            gradient_steps=self.gradient_steps,
        )

        num_episodes = len(completed_winners)
        if num_episodes > 0 and self.team1_indices is not None and self.team2_indices is not None:
            winners_arr = np.asarray(completed_winners, dtype=np.int32)
            win_rate_agent1 = float(np.mean(winners_arr == 1))
            win_rate_agent2 = float(np.mean(winners_arr == 2))
            draw_rate = float(np.mean(winners_arr == 0))
            avg_ep_return_agent1 = float(np.mean(completed_returns_team1))
            avg_ep_return_agent2 = float(np.mean(completed_returns_team2))
        else:
            win_rate_agent1 = 0.0
            win_rate_agent2 = 0.0
            draw_rate = 0.0
            avg_ep_return_agent1 = 0.0
            avg_ep_return_agent2 = 0.0

        if all_step_rewards_team1:
            step_rewards_team1 = np.concatenate(all_step_rewards_team1, axis=0)
            avg_reward = float(step_rewards_team1.mean())
        else:
            avg_reward = 0.0

        stats: Dict[str, float] = {
            "episodes_in_batch": float(num_episodes),
            "win_rate_agent1": win_rate_agent1,
            "win_rate_agent2": win_rate_agent2,
            "draw_rate": draw_rate,
            "avg_episode_return_agent1": avg_ep_return_agent1,
            "avg_episode_return_agent2": avg_ep_return_agent2,
            "avg_reward": avg_reward,
        }

        metrics.update(stats)
        return metrics, avg_reward


class SelfPlayOffPolicyRecurrentMultiAgentTrainer:
    """Recurrent off-policy self-play trainer for N-agent games.

    Similar to SelfPlayOffPolicyRecurrentTrainer but generalized to
    num_agents agents per environment and using MultiAgentVectorizedEnv.
    Sequences are stored in a SequenceReplayBuffer.
    """

    def __init__(
        self,
        agent: Any,
        envs: MultiAgentVectorizedEnv,
        replay_buffer: SequenceReplayBuffer,
        steps_per_update: int = 2048,
        batch_size: int = 4,
        max_seq_len: int = 64,
        gradient_steps: int = 1,
        team1_indices: Optional[Sequence[int]] = None,
        team2_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.replay_buffer = replay_buffer
        self.steps_per_update = steps_per_update
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.gradient_steps = gradient_steps
        self.num_envs = envs.num_envs
        self.num_agents = envs.num_agents

        self.team1_indices = list(team1_indices) if team1_indices is not None else None
        self.team2_indices = list(team2_indices) if team2_indices is not None else None

    def train_step(self) -> (Dict[str, float], float):
        obs = self.envs.reset()  # (T0, N_env, N_agent, obs_dim) at T0=0 -> (N_env,N_agent,obs_dim)

        obs_seq: List[np.ndarray] = []
        actions_seq: List[np.ndarray] = []
        rewards_seq: List[np.ndarray] = []
        dones_seq: List[np.ndarray] = []
        next_obs_seq: List[np.ndarray] = []

        ep_returns = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        completed_returns_team1: List[float] = []
        completed_returns_team2: List[float] = []
        completed_winners: List[int] = []

        for _ in range(self.steps_per_update):
            actions = np.zeros_like(obs[..., :2], dtype=np.float32)
            for env_idx in range(self.num_envs):
                for agent_idx in range(self.num_agents):
                    a = self.agent.select_action(obs[env_idx, agent_idx], deterministic=False)
                    actions[env_idx, agent_idx] = a

            next_obs, rewards, dones_arr, infos, _ = self.envs.step(actions)

            obs_seq.append(obs)
            actions_seq.append(actions)
            rewards_seq.append(rewards)
            dones_seq.append(dones_arr.astype(np.float32))
            next_obs_seq.append(next_obs)

            ep_returns += rewards

            for env_idx, done_flag in enumerate(dones_arr):
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

            obs = next_obs

        # (T, N_env, N_agent, ...)
        obs_seq = np.stack(obs_seq, axis=0)
        actions_seq = np.stack(actions_seq, axis=0)
        rewards_seq = np.stack(rewards_seq, axis=0)
        dones_seq = np.stack(dones_seq, axis=0)
        next_obs_seq = np.stack(next_obs_seq, axis=0)

        T, N, A, _ = obs_seq.shape

        # env×agent ごとのシーケンスを切り出し、done で分割してバッファに追加
        for env_idx in range(self.num_envs):
            for agent_idx in range(self.num_agents):
                obs_env = obs_seq[:, env_idx, agent_idx, :]
                act_env = actions_seq[:, env_idx, agent_idx, :]
                rew_env = rewards_seq[:, env_idx, agent_idx]
                next_obs_env = next_obs_seq[:, env_idx, agent_idx, :]
                done_env = dones_seq[:, env_idx]

                start = 0
                for t in range(T):
                    if done_env[t] == 1.0:
                        end = t
                        if end >= start:
                            self.replay_buffer.add_sequence(
                                obs_env[start : end + 1],
                                act_env[start : end + 1],
                                rew_env[start : end + 1],
                                next_obs_env[start : end + 1],
                                done_env[start : end + 1],
                                agent_index=agent_idx,
                            )
                        start = t + 1
                if start < T:
                    self.replay_buffer.add_sequence(
                        obs_env[start:T],
                        act_env[start:T],
                        rew_env[start:T],
                        next_obs_env[start:T],
                        done_env[start:T],
                        agent_index=agent_idx,
                    )

        metrics = self.agent.update(
            replay_buffer=self.replay_buffer,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            gradient_steps=self.gradient_steps,
        )

        num_episodes = len(completed_winners)
        if num_episodes > 0 and self.team1_indices is not None and self.team2_indices is not None:
            winners_arr = np.asarray(completed_winners, dtype=np.int32)
            win_rate_agent1 = float(np.mean(winners_arr == 1))
            win_rate_agent2 = float(np.mean(winners_arr == 2))
            draw_rate = float(np.mean(winners_arr == 0))
            avg_ep_return_agent1 = float(np.mean(completed_returns_team1))
            avg_ep_return_agent2 = float(np.mean(completed_returns_team2))
        else:
            win_rate_agent1 = 0.0
            win_rate_agent2 = 0.0
            draw_rate = 0.0
            avg_ep_return_agent1 = 0.0
            avg_ep_return_agent2 = 0.0

        if rewards_seq.size > 0 and self.team1_indices is not None:
            team1_rewards = rewards_seq[:, :, self.team1_indices].mean(axis=2).mean(axis=1)
            avg_reward = float(team1_rewards.mean())
        else:
            avg_reward = 0.0

        stats: Dict[str, float] = {
            "episodes_in_batch": float(num_episodes),
            "win_rate_agent1": win_rate_agent1,
            "win_rate_agent2": win_rate_agent2,
            "draw_rate": draw_rate,
            "avg_episode_return_agent1": avg_ep_return_agent1,
            "avg_episode_return_agent2": avg_ep_return_agent2,
            "avg_reward": avg_reward,
        }

        metrics.update(stats)
        return metrics, avg_reward
