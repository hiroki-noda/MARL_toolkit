"""
PPOアルゴリズムの実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ActorCritic(nn.Module):
    """Actor-Criticネットワーク"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 共有の特徴抽出層
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor（方策）ネットワーク
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic（価値）ネットワーク
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向き伝播
        Returns:
            action_mean: 行動の平均
            value: 状態価値
        """
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """
        行動と価値、対数確率を取得
        """
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.actor_logstd)
        
        # 正規分布を作成
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value


class PPOAgent:
    """PPOエージェント

    環境とのやり取りと学習を担当するクラスです。
    主に `select_action` と `update` を通して利用します。
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
        device: str = None
    ):
        """コンストラクタ

        Args:
            obs_dim: 観測ベクトルの次元数。
            action_dim: 連続行動ベクトルの次元数。
            hidden_dim: Actor-Critic ネットワーク内部の隠れ層ユニット数。
            lr: Adam オプティマイザの学習率。
            gamma: 割引率 γ。
            gae_lambda: GAE (Generalized Advantage Estimation) の λ。
            clip_epsilon: PPO のクリッピング係数 (ε)。
            value_coef: 価値損失の重み係数。
            entropy_coef: エントロピー正則化の重み係数。
            max_grad_norm: 勾配クリッピングの上限ノルム。
            device: 使用するデバイス。"cuda" / "cpu"。None の場合は自動判定。
        """
        # デバイスの安全な選択
        if device is None:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception as e:
                print(f"Warning: CUDA check failed ({e}), using CPU")
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # ネットワークの初期化
        self.network = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        """行動を選択"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, _ = self.network(obs_tensor)
            
            if deterministic:
                action = action_mean
            else:
                action_std = torch.exp(self.network.actor_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
        
        return action.cpu().numpy()[0]
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GAE (Generalized Advantage Estimation) を計算
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        num_epochs: int = 10,
        batch_size: int = 64
    ):
        """
        PPOの更新
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # アドバンテージの正規化
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        dataset_size = obs_tensor.shape[0]
        indices = np.arange(dataset_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # 現在の方策での対数確率と価値を計算
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # 方策損失（PPOクリップ）
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 価値損失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # エントロピー損失
                entropy_loss = -entropy.mean()
                
                # 総損失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 勾配更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }
    
    def save(self, path: str):
        """モデルを保存"""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
