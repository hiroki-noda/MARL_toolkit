"""
Self-playトレーナー
"""

import numpy as np
import torch
from typing import List, Optional, Sequence
from ppo_trainer.ppo import PPOAgent
from ppo_trainer.vec_env import VectorizedEnv, MultiAgentVectorizedEnv


class SelfPlayTrainer:
    """Self-playでPPOを学習するトレーナー"""
    
    def __init__(
        self,
        agent: PPOAgent,
        envs: VectorizedEnv,
        steps_per_update: int = 2048,
        num_epochs: int = 10,
        batch_size: int = 64
    ):
        self.agent = agent
        self.envs = envs
        self.steps_per_update = steps_per_update
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_envs = envs.num_envs
        
    def collect_rollouts(self):
        """
        ロールアウトを収集
        1つのモデルを両エージェントに適用してself-playで学習
        """
        obs1, obs2 = self.envs.reset()
        
        # データバッファ
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        # エピソード統計用バッファ
        ep_returns1 = np.zeros(self.num_envs, dtype=np.float32)
        ep_returns2 = np.zeros(self.num_envs, dtype=np.float32)
        completed_returns1 = []
        completed_returns2 = []
        completed_winners = []  # 1: agent1, 2: agent2, 0: draw/none
        
        for step in range(self.steps_per_update):
            # 両エージェントの行動を同じモデルで選択
            actions1 = []
            actions2 = []
            log_probs1_list = []
            log_probs2_list = []
            values1_list = []
            values2_list = []
            
            for i in range(self.num_envs):
                # Agent 1
                obs1_tensor = torch.FloatTensor(obs1[i]).unsqueeze(0).to(self.agent.device)
                action1, log_prob1, _, value1 = self.agent.network.get_action_and_value(obs1_tensor)
                actions1.append(action1.cpu().numpy()[0])
                log_probs1_list.append(log_prob1.item())
                values1_list.append(value1.item())
                
                # Agent 2（同じモデルを使用）
                obs2_tensor = torch.FloatTensor(obs2[i]).unsqueeze(0).to(self.agent.device)
                action2, log_prob2, _, value2 = self.agent.network.get_action_and_value(obs2_tensor)
                actions2.append(action2.cpu().numpy()[0])
                log_probs2_list.append(log_prob2.item())
                values2_list.append(value2.item())
            
            actions1 = np.array(actions1)
            actions2 = np.array(actions2)
            
            # 環境でステップ実行
            (next_obs1, next_obs2), (rewards1, rewards2), dones_arr, infos, _ = self.envs.step(actions1, actions2)

            # エピソードごとの累積報酬を更新
            ep_returns1 += rewards1
            ep_returns2 += rewards2
            
            # Agent 1のデータを保存
            observations.append(obs1)
            actions.append(actions1)
            log_probs.append(np.array(log_probs1_list))
            rewards.append(rewards1)
            dones.append(dones_arr)
            values.append(np.array(values1_list))
            
            # Agent 2のデータを保存（self-playのため同じバッファに追加）
            observations.append(obs2)
            actions.append(actions2)
            log_probs.append(np.array(log_probs2_list))
            rewards.append(rewards2)
            dones.append(dones_arr)
            values.append(np.array(values2_list))
            
            # 終了したエピソードの統計を記録
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

                    # 次のエピソードに備えて累積報酬をリセット
                    ep_returns1[env_idx] = 0.0
                    ep_returns2[env_idx] = 0.0

            obs1 = next_obs1
            obs2 = next_obs2
        
        # 最後の価値を計算
        with torch.no_grad():
            next_values1 = []
            next_values2 = []
            for i in range(self.num_envs):
                obs1_tensor = torch.FloatTensor(obs1[i]).unsqueeze(0).to(self.agent.device)
                _, value1 = self.agent.network(obs1_tensor)
                next_values1.append(value1.item())
                
                obs2_tensor = torch.FloatTensor(obs2[i]).unsqueeze(0).to(self.agent.device)
                _, value2 = self.agent.network(obs2_tensor)
                next_values2.append(value2.item())
        
        # データを整形
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        log_probs = np.concatenate(log_probs, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)
        values = np.concatenate(values, axis=0)
        next_values = np.array(next_values1 + next_values2)

        # エピソード統計の集計
        num_episodes = len(completed_winners)
        if num_episodes > 0:
            completed_winners_arr = np.array(completed_winners)
            win_rate_agent1 = float(np.mean(completed_winners_arr == 1))
            win_rate_agent2 = float(np.mean(completed_winners_arr == 2))
            draw_rate = float(np.mean(completed_winners_arr == 0))
            avg_ep_return_agent1 = float(np.mean(completed_returns1))
            avg_ep_return_agent2 = float(np.mean(completed_returns2))
        else:
            win_rate_agent1 = 0.0
            win_rate_agent2 = 0.0
            draw_rate = 0.0
            avg_ep_return_agent1 = 0.0
            avg_ep_return_agent2 = 0.0

        stats = {
            "episodes_in_batch": num_episodes,
            "win_rate_agent1": win_rate_agent1,
            "win_rate_agent2": win_rate_agent2,
            "draw_rate": draw_rate,
            "avg_episode_return_agent1": avg_ep_return_agent1,
            "avg_episode_return_agent2": avg_ep_return_agent2,
        }
        
        return observations, actions, log_probs, rewards, dones, values, next_values, stats
    
    def train_step(self):
        """1回の学習ステップ"""
        # ロールアウトを収集
        observations, actions, log_probs, rewards, dones, values, next_values, stats = self.collect_rollouts()
        
        # GAEとリターンを計算（簡略化：全体で一度に計算）
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # 各環境ごとにGAEを計算（Agent1とAgent2を交互に処理）
        for env_idx in range(self.num_envs * 2):
            env_start = env_idx
            env_slice = slice(env_start, len(rewards), self.num_envs * 2)
            
            env_rewards = rewards[env_slice]
            env_values = values[env_slice]
            env_dones = dones[env_slice]
            
            last_gae = 0
            for t in reversed(range(len(env_rewards))):
                if t == len(env_rewards) - 1:
                    next_val = next_values[env_idx]
                else:
                    next_val = env_values[t + 1]
                
                delta = env_rewards[t] + self.agent.gamma * next_val * (1 - env_dones[t]) - env_values[t]
                last_gae = delta + self.agent.gamma * self.agent.gae_lambda * (1 - env_dones[t]) * last_gae
                advantages[env_slice][t] = last_gae
        
        returns = advantages + values
        
        # PPOの更新
        metrics = self.agent.update(
            observations,
            actions,
            log_probs,
            advantages,
            returns,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size
        )
        
        # エピソード統計もmetricsに含める
        metrics.update(stats)

        return metrics, rewards.mean()


class SelfPlayMultiAgentTrainer:
    """複数エージェント（num_agents体）を self-play で学習するPPOトレーナー

    1つの PPOAgent（共有パラメータ）を全エージェントに適用し、
    各エージェントごとの経験をまとめて PPO で更新する。

    adversarial_tag_4p 用に 4 エージェント (chaser2体, evader2体) を
    想定しつつ、num_agents 一般に動くように実装している。
    """

    def __init__(
        self,
        agent: PPOAgent,
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

        # 勝率・エピソード報酬をチーム単位で集計したい場合に使用
        self.team1_indices = list(team1_indices) if team1_indices is not None else None
        self.team2_indices = list(team2_indices) if team2_indices is not None else None

    def collect_rollouts(self):
        """ロールアウトを収集（num_agents 体分）"""
        # obs: (num_envs, num_agents, obs_dim)
        obs = self.envs.reset()

        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        # エピソード統計
        ep_returns = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        completed_returns_team1: List[float] = []
        completed_returns_team2: List[float] = []
        completed_winners: List[int] = []  # 1: team1, 2: team2, 0: none/draw

        for _ in range(self.steps_per_update):
            # 全env・全エージェントの行動を同じモデルで選択
            actions_step = np.zeros_like(obs[..., :2], dtype=np.float32)  # (num_envs, num_agents, action_dim)
            logp_step = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
            values_step = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)

            for env_idx in range(self.num_envs):
                for agent_idx in range(self.num_agents):
                    obs_tensor = torch.FloatTensor(obs[env_idx, agent_idx]).unsqueeze(0).to(self.agent.device)
                    action, log_prob, _, value = self.agent.network.get_action_and_value(obs_tensor)
                    actions_step[env_idx, agent_idx] = action.cpu().numpy()[0]
                    logp_step[env_idx, agent_idx] = log_prob.item()
                    values_step[env_idx, agent_idx] = value.item()

            # 環境でステップ実行
            next_obs, rewards_step, dones_env, infos, _ = self.envs.step(actions_step)

            # エピソードごとの累積報酬を更新
            ep_returns += rewards_step

            # データバッファへ格納
            for agent_idx in range(self.num_agents):
                observations.append(obs[:, agent_idx, :])
                actions.append(actions_step[:, agent_idx, :])
                log_probs.append(logp_step[:, agent_idx])
                rewards.append(rewards_step[:, agent_idx])
                dones.append(dones_env)
                values.append(values_step[:, agent_idx])

            # エピソード終了時の統計を記録
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

            obs = next_obs

        # 最後の価値を計算
        with torch.no_grad():
            next_values: List[float] = []
            # データのフラット化順に合わせて、agent軸を先、env軸を後に回す
            for agent_idx in range(self.num_agents):
                for env_idx in range(self.num_envs):
                    obs_tensor = torch.FloatTensor(obs[env_idx, agent_idx]).unsqueeze(0).to(self.agent.device)
                    _, value = self.agent.network(obs_tensor)
                    next_values.append(value.item())
        next_values = np.array(next_values, dtype=np.float32)

        # データを整形（SelfPlayTrainer と同じ並びにする）
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        log_probs = np.concatenate(log_probs, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)
        values = np.concatenate(values, axis=0)

        # エピソード統計
        num_episodes = len(completed_winners)
        if num_episodes > 0 and self.team1_indices is not None and self.team2_indices is not None:
            completed_winners_arr = np.array(completed_winners)
            win_rate_agent1 = float(np.mean(completed_winners_arr == 1))
            win_rate_agent2 = float(np.mean(completed_winners_arr == 2))
            draw_rate = float(np.mean(completed_winners_arr == 0))
            avg_ep_return_agent1 = float(np.mean(completed_returns_team1))
            avg_ep_return_agent2 = float(np.mean(completed_returns_team2))
        else:
            win_rate_agent1 = 0.0
            win_rate_agent2 = 0.0
            draw_rate = 0.0
            avg_ep_return_agent1 = 0.0
            avg_ep_return_agent2 = 0.0

        stats = {
            "episodes_in_batch": num_episodes,
            "win_rate_agent1": win_rate_agent1,
            "win_rate_agent2": win_rate_agent2,
            "draw_rate": draw_rate,
            "avg_episode_return_agent1": avg_ep_return_agent1,
            "avg_episode_return_agent2": avg_ep_return_agent2,
        }

        return observations, actions, log_probs, rewards, dones, values, next_values, stats

    def train_step(self):
        """1回の学習ステップ (num_agents 体)"""
        observations, actions, log_probs, rewards, dones, values, next_values, stats = self.collect_rollouts()

        advantages = np.zeros_like(rewards)

        # env×agent ごとに GAE を計算
        num_env_agents = self.num_envs * self.num_agents
        for env_agent_idx in range(num_env_agents):
            env_start = env_agent_idx
            env_slice = slice(env_start, len(rewards), num_env_agents)

            env_rewards = rewards[env_slice]
            env_values = values[env_slice]
            env_dones = dones[env_slice]

            last_gae = 0.0
            for t in reversed(range(len(env_rewards))):
                if t == len(env_rewards) - 1:
                    next_val = next_values[env_agent_idx]
                else:
                    next_val = env_values[t + 1]

                delta = env_rewards[t] + self.agent.gamma * next_val * (1 - env_dones[t]) - env_values[t]
                last_gae = delta + self.agent.gamma * self.agent.gae_lambda * (1 - env_dones[t]) * last_gae
                advantages[env_slice][t] = last_gae

        returns = advantages + values

        metrics = self.agent.update(
            observations,
            actions,
            log_probs,
            advantages,
            returns,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )

        metrics.update(stats)
        return metrics, rewards.mean()
