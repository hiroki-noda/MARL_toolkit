"""
Self-playで学習するメインスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from sumo_env import AntSumoEnv
from ppo_trainer import (
    PPOAgent,
    SelfPlayTrainer,
    SelfPlayMultiAgentTrainer,
    VectorizedEnv,
    MultiAgentVectorizedEnv,
    SoftActorCriticAgent,
    SACConfig,
    ReplayBuffer,
    TD3Agent,
    TD3Config,
    SelfPlayOffPolicyTrainer,
    SelfPlayOffPolicyRecurrentTrainer,
    SelfPlayOffPolicyMultiAgentTrainer,
    SelfPlayOffPolicyRecurrentMultiAgentTrainer,
    RecurrentPPOAgent,
    SelfPlayRecurrentPPOTrainer,
    SelfPlayRecurrentMultiAgentTrainer,
    SequenceReplayBuffer,
    RecurrentSACAgent,
    RecurrentSACConfig,
    RecurrentTD3Agent,
    RecurrentTD3Config,
    RoleAwareRecurrentTD3Agent,
    RoleAwareRecurrentTD3Config,
    RoleAwareRecurrentSACAgent,
    RoleAwareRecurrentSACConfig,
    RoleAwareRecurrentPPOAgent,
    RoleAwareRecurrentPPOConfig,
)
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime

from env_factory import create_env, make_env_fns, ENV_IDS


def train(
    num_envs: int = 8,
    total_steps: int = 100_000_000,
    steps_per_update: int = 2048,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    save_interval: int = 1,
    log_dir: str = "logs",
    use_gpu: bool = True,
    checkpoint_dir: str = "checkpoints",
    env_id: str = "ant_sumo",
    algo: str = "ppo",
    offpolicy_batch_size: int = 256,
    offpolicy_gradient_steps: int = 1,
    offpolicy_max_seq_len: int = 64,
):
    """学習を実行

    env_id によって使用するマルチエージェント環境を切り替える。
    """
    
    # ディレクトリの作成
    os.makedirs(log_dir, exist_ok=True)
    # チェックポイントはタスク(env_id)とアルゴリズムごとにサブディレクトリを分ける
    if algo == "ppo":
        # 既存との互換性のため、PPOは従来通り env_id 直下
        task_checkpoint_dir = os.path.join(checkpoint_dir, env_id)
    else:
        task_checkpoint_dir = os.path.join(checkpoint_dir, f"{env_id}_{algo}")
    os.makedirs(task_checkpoint_dir, exist_ok=True)
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"{log_dir}/run_{timestamp}")
    
    # デバイスの確認
    try:
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        if device == "cuda":
            print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[WARNING] CUDA check failed: {e}")
        device = "cpu"
    
    print(f"[INFO] Starting training with {num_envs} parallel environments")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Environment ID: {env_id}")
    print(f"[INFO] Algorithm: {algo}")
    
    # 並列環境の作成
    env_fns = make_env_fns(env_id, num_envs)

    # adversarial_tag_4p では4体のエージェントを個別に扱う
    is_four_agent_env = env_id == "adversarial_tag_4p"
    if is_four_agent_env:
        envs = MultiAgentVectorizedEnv(env_fns, num_agents=4)
    else:
        envs = VectorizedEnv(env_fns)
    
    # 1つのサンプル環境を作成して観測・行動空間を取得
    sample_env = create_env(env_id)
    obs_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.shape[0]
    
    print(f"[INFO] Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # アルゴリズムごとのエージェントとトレーナーを作成
    algo = algo.lower()

    if algo == "ppo":
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
        )
        if is_four_agent_env:
            # 4エージェント版タグ環境: agent1,3 が追跡者 / agent2,4 が逃走者
            trainer = SelfPlayMultiAgentTrainer(
                agent=agent,
                envs=envs,
                num_agents=4,
                steps_per_update=steps_per_update,
                num_epochs=num_epochs,
                batch_size=batch_size,
                team1_indices=[0, 2],  # chasers
                team2_indices=[1, 3],  # evaders
            )
        else:
            trainer = SelfPlayTrainer(
                agent=agent,
                envs=envs,
                steps_per_update=steps_per_update,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )

    elif algo == "rppo":
        agent = RecurrentPPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
        )

        if is_four_agent_env:
            # 4エージェント版タグ環境: agent1,3 が追跡者 / agent2,4 が逃走者
            trainer = SelfPlayRecurrentMultiAgentTrainer(
                agent=agent,
                envs=envs,
                num_agents=4,
                steps_per_update=steps_per_update,
                num_epochs=num_epochs,
                batch_size=batch_size,
                team1_indices=[0, 2],  # chasers
                team2_indices=[1, 3],  # evaders
            )
        else:
            trainer = SelfPlayRecurrentPPOTrainer(
                agent=agent,
                envs=envs,
                steps_per_update=steps_per_update,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )

    elif algo == "sac":
        sac_config = SACConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
        )
        agent = SoftActorCriticAgent(sac_config)
        replay_buffer = ReplayBuffer(obs_dim, action_dim, capacity=1_000_000, device=device)
        if is_four_agent_env:
            trainer = SelfPlayOffPolicyMultiAgentTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                gradient_steps=offpolicy_gradient_steps,
                team1_indices=[0, 2],
                team2_indices=[1, 3],
            )
        else:
            trainer = SelfPlayOffPolicyTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                gradient_steps=offpolicy_gradient_steps,
            )

    elif algo == "td3":
        td3_config = TD3Config(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
        )
        agent = TD3Agent(td3_config)
        replay_buffer = ReplayBuffer(obs_dim, action_dim, capacity=1_000_000, device=device)
        if is_four_agent_env:
            trainer = SelfPlayOffPolicyMultiAgentTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                gradient_steps=offpolicy_gradient_steps,
                team1_indices=[0, 2],
                team2_indices=[1, 3],
            )
        else:
            trainer = SelfPlayOffPolicyTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                gradient_steps=offpolicy_gradient_steps,
            )

    elif algo == "rsac":
        rsac_config = RecurrentSACConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
        )
        agent = RecurrentSACAgent(rsac_config)
        replay_buffer = SequenceReplayBuffer(capacity=10_000)
        if is_four_agent_env:
            trainer = SelfPlayOffPolicyRecurrentMultiAgentTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                max_seq_len=offpolicy_max_seq_len,
                gradient_steps=offpolicy_gradient_steps,
                team1_indices=[0, 2],
                team2_indices=[1, 3],
            )
        else:
            trainer = SelfPlayOffPolicyRecurrentTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                max_seq_len=offpolicy_max_seq_len,
                gradient_steps=offpolicy_gradient_steps,
            )

    elif algo == "rsac_role":
        # Role-aware recurrent SAC (adversarial_tag_4p 専用想定)
        rsac_role_config = RoleAwareRecurrentSACConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
        )
        agent = RoleAwareRecurrentSACAgent(rsac_role_config)
        replay_buffer = SequenceReplayBuffer(capacity=10_000)
        if is_four_agent_env:
            trainer = SelfPlayOffPolicyRecurrentMultiAgentTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                max_seq_len=offpolicy_max_seq_len,
                gradient_steps=offpolicy_gradient_steps,
                team1_indices=[0, 2],
                team2_indices=[1, 3],
            )
        else:
            raise ValueError("rsac_role is currently intended only for adversarial_tag_4p (4-agent) environment.")

    elif algo == "rtd3":
        rtd3_config = RecurrentTD3Config(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
        )
        agent = RecurrentTD3Agent(rtd3_config)
        replay_buffer = SequenceReplayBuffer(capacity=10_000)
        if is_four_agent_env:
            trainer = SelfPlayOffPolicyRecurrentMultiAgentTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                max_seq_len=offpolicy_max_seq_len,
                gradient_steps=offpolicy_gradient_steps,
                team1_indices=[0, 2],
                team2_indices=[1, 3],
            )
        else:
            trainer = SelfPlayOffPolicyRecurrentTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                max_seq_len=offpolicy_max_seq_len,
                gradient_steps=offpolicy_gradient_steps,
            )

    elif algo == "rtd3_role":
        # Role-aware recurrent TD3 (adversarial_tag_4p 専用想定)
        role_rtd3_config = RoleAwareRecurrentTD3Config(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
        )
        agent = RoleAwareRecurrentTD3Agent(role_rtd3_config)
        replay_buffer = SequenceReplayBuffer(capacity=10_000)
        if is_four_agent_env:
            trainer = SelfPlayOffPolicyRecurrentMultiAgentTrainer(
                agent=agent,
                envs=envs,
                replay_buffer=replay_buffer,
                steps_per_update=steps_per_update,
                batch_size=offpolicy_batch_size,
                max_seq_len=offpolicy_max_seq_len,
                gradient_steps=offpolicy_gradient_steps,
                team1_indices=[0, 2],
                team2_indices=[1, 3],
            )
        else:
            raise ValueError("rtd3_role is currently intended only for adversarial_tag_4p (4-agent) environment.")

    elif algo == "rppo_role":
        # Role-aware recurrent PPO (adversarial_tag_4p 専用想定)
        rppo_role_config = RoleAwareRecurrentPPOConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=learning_rate,
            device=device,
            num_agents=4,
            num_other_agents=3,
        )
        agent = RoleAwareRecurrentPPOAgent(rppo_role_config)

        if is_four_agent_env:
            trainer = SelfPlayRecurrentMultiAgentTrainer(
                agent=agent,
                envs=envs,
                num_agents=4,
                steps_per_update=steps_per_update,
                num_epochs=num_epochs,
                batch_size=batch_size,
                team1_indices=[0, 2],  # chasers
                team2_indices=[1, 3],  # evaders
            )
        else:
            raise ValueError("rppo_role is currently intended only for adversarial_tag_4p (4-agent) environment.")

    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # 学習ループ
    num_updates = total_steps // (steps_per_update * num_envs)
    
    print(f"[INFO] Total updates: {num_updates}")
    print("[INFO] Starting training...")
    
    for update in range(num_updates):
        # 学習ステップ
        metrics, avg_reward = trainer.train_step()

        # ログ
        global_step = (update + 1) * steps_per_update * num_envs

        if algo == "ppo":
            writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
            writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
            writer.add_scalar("train/entropy", metrics["entropy"], global_step)
        else:
            # Off-policy or recurrent algorithms expose different metrics
            if "q1_loss" in metrics:
                writer.add_scalar("train/q1_loss", metrics["q1_loss"], global_step)
            if "q2_loss" in metrics:
                writer.add_scalar("train/q2_loss", metrics["q2_loss"], global_step)
            if "policy_loss" in metrics:
                writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
            if "critic_loss" in metrics:
                writer.add_scalar("train/critic_loss", metrics["critic_loss"], global_step)
            if "alpha" in metrics:
                writer.add_scalar("train/alpha", metrics["alpha"], global_step)
            # role-aware variants (rtd3_role, rsac_role, rppo_role) optionally report role_loss
            if "role_loss" in metrics:
                writer.add_scalar("train/role_loss", metrics["role_loss"], global_step)

        writer.add_scalar("train/avg_reward", avg_reward, global_step)

        # エピソード統計（勝率・エピソード報酬）をTensorBoardに記録
        episodes_in_batch = metrics.get("episodes_in_batch", 0)
        if episodes_in_batch > 0:
            writer.add_scalar("train/win_rate_agent1", metrics["win_rate_agent1"], global_step)
            writer.add_scalar("train/win_rate_agent2", metrics["win_rate_agent2"], global_step)
            writer.add_scalar("train/draw_rate", metrics["draw_rate"], global_step)
            writer.add_scalar("train/avg_episode_return_agent1", metrics["avg_episode_return_agent1"], global_step)
            writer.add_scalar("train/avg_episode_return_agent2", metrics["avg_episode_return_agent2"], global_step)
            writer.add_scalar("train/episodes_in_batch", episodes_in_batch, global_step)
        
        if (update + 1) % 10 == 0:
            msg = (f"Update {update + 1}/{num_updates} | "
                   f"Steps: {global_step} | "
                   f"Avg Reward: {avg_reward:.6f}")

            if algo == "ppo":
                msg += (f" | Policy Loss: {metrics['policy_loss']:.6f} "
                        f"Value Loss: {metrics['value_loss']:.6f}")
            else:
                if "q1_loss" in metrics and "q2_loss" in metrics:
                    msg += (f" | Q1 Loss: {metrics['q1_loss']:.6f} "
                            f"Q2 Loss: {metrics['q2_loss']:.6f}")
                if "policy_loss" in metrics:
                    msg += f" | Policy Loss: {metrics['policy_loss']:.6f}"
                if "critic_loss" in metrics:
                    msg += f" | Critic Loss: {metrics['critic_loss']:.6f}"
                if "role_loss" in metrics:
                    msg += f" | Role Loss: {metrics['role_loss']:.6f}"

            if episodes_in_batch > 0:
                msg += (f" | Win1: {metrics['win_rate_agent1']:.3f} "
                        f"Win2: {metrics['win_rate_agent2']:.3f} "
                        f"Draw: {metrics['draw_rate']:.3f} "
                        f"EpRet1: {metrics['avg_episode_return_agent1']:.4f} "
                        f"EpRet2: {metrics['avg_episode_return_agent2']:.4f}")

            print(msg)
        
        # モデルの保存（タスクごとのサブディレクトリに保存）
        if (update + 1) % save_interval == 0:
            save_path = os.path.join(task_checkpoint_dir, f"model_step_{global_step}.pt")
            agent.save(save_path)
            print(f"[INFO] Model saved to {save_path}")
    
    # 最終モデルの保存
    final_path = os.path.join(task_checkpoint_dir, "model_final.pt")
    agent.save(final_path)
    print(f"[INFO] Final model saved to {final_path}")
    
    # 環境を閉じる
    envs.close()
    writer.close()
    
    print("[INFO] Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent with self-play on various environments")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--total-steps", type=int, default=100_000_000, help="Total training steps")
    parser.add_argument("--steps-per-update", type=int, default=2048, help="Steps per PPO update")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for PPO update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-interval", type=int, default=10, help="Save model every N updates")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--env-id", type=str, default="ant_sumo",
                        choices=ENV_IDS,
                        help=f"Environment ID to use (default: ant_sumo). Choices: {', '.join(ENV_IDS)}")
    parser.add_argument("--algo", type=str, default="ppo",
            choices=["ppo", "rppo", "sac", "td3", "rsac", "rtd3", "rtd3_role", "rsac_role", "rppo_role"],
            help="RL algorithm to use (default: ppo; rppo/rsac/rtd3/rtd3_role/rsac_role/rppo_role = recurrent variants)")
    parser.add_argument("--offpolicy-batch-size", type=int, default=256,
                        help="Batch size for off-policy updates (SAC/TD3)")
    parser.add_argument("--offpolicy-gradient-steps", type=int, default=1,
                        help="Number of gradient steps per update for off-policy algorithms")
    parser.add_argument("--offpolicy-max-seq-len", type=int, default=64,
                        help="Maximum sequence length for recurrent off-policy algorithms (rsac/rtd3)")
    
    args = parser.parse_args()
    
    train(
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        steps_per_update=args.steps_per_update,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_interval=args.save_interval,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_gpu=args.use_gpu,
        env_id=args.env_id,
        algo=args.algo,
        offpolicy_batch_size=args.offpolicy_batch_size,
        offpolicy_gradient_steps=args.offpolicy_gradient_steps,
        offpolicy_max_seq_len=args.offpolicy_max_seq_len,
    )
