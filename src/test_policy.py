"""
学習済みモデルのテスト・可視化スクリプト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import mujoco
import mujoco.viewer
from sumo_env import AntSumoEnv
from ppo_trainer import (
    PPOAgent,
    SoftActorCriticAgent,
    SACConfig,
    TD3Agent,
    TD3Config,
    RecurrentPPOAgent,
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
from env_factory import create_env, ENV_IDS
import argparse
import time
import glob


def find_latest_checkpoint(checkpoint_dir: str):
    """最新のチェックポイントを見つける"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "model_step_*.pt"))
    if not checkpoints:
        # 最終モデルを探す
        final_model = os.path.join(checkpoint_dir, "model_final.pt")
        if os.path.exists(final_model):
            return final_model
        return None
    
    # ステップ数でソート
    checkpoints.sort(key=lambda x: int(x.split("_step_")[1].split(".pt")[0]))
    return checkpoints[-1]


def test_policy(
    checkpoint_path: str,
    num_episodes: int = 10,
    render: bool = True,
    deterministic: bool = True,
    fps: int = 50,
    env_id: str = "ant_sumo",
    algo: str = "ppo",
):
    """
    学習済みモデルをテスト
    
    Args:
        checkpoint_path: モデルのパス
        num_episodes: テストするエピソード数
        render: レンダリングするかどうか
        deterministic: 決定的な行動を取るか
        fps: FPS（レンダリング時）
    """
    
    print(f"[INFO] Loading model from: {checkpoint_path}")
    
    # 環境の作成
    env = create_env(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # エージェントの作成とモデルのロード
    algo = algo.lower()
    device = "cpu"

    # 4体環境かどうか
    is_four_agent_env = env_id == "adversarial_tag_4p"

    # 非recurrent系は1インスタンスを全エージェントで共有
    recurrent_algos = {"rppo", "rsac", "rtd3", "rtd3_role", "rsac_role", "rppo_role"}

    if algo == "ppo":
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
        agent.load(checkpoint_path)
        agent.network.eval()
        agents = [agent]
    elif algo == "sac":
        sac_config = SACConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        agent = SoftActorCriticAgent(sac_config)
        agent.load(checkpoint_path)
        agents = [agent]
    elif algo == "td3":
        td3_config = TD3Config(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        agent = TD3Agent(td3_config)
        agent.load(checkpoint_path)
        agents = [agent]
    elif algo == "rppo":
        if is_four_agent_env:
            # 4体環境では各物理エージェントごとに独立したhidden stateを持つよう
            # RecurrentPPOAgent インスタンスを4つ生成する（パラメータは共有）
            agents = []
            for _ in range(4):
                a = RecurrentPPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
                a.load(checkpoint_path)
                agents.append(a)
        else:
            agent = RecurrentPPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
            agent.load(checkpoint_path)
            agents = [agent]
    elif algo == "rppo_role":
        rppo_role_config = RoleAwareRecurrentPPOConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        if is_four_agent_env:
            agents = []
            for _ in range(4):
                a = RoleAwareRecurrentPPOAgent(rppo_role_config)
                a.load(checkpoint_path)
                agents.append(a)
        else:
            agent = RoleAwareRecurrentPPOAgent(rppo_role_config)
            agent.load(checkpoint_path)
            agents = [agent]
    elif algo == "rsac":
        rsac_config = RecurrentSACConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        if is_four_agent_env:
            agents = []
            for _ in range(4):
                a = RecurrentSACAgent(rsac_config)
                a.load(checkpoint_path)
                agents.append(a)
        else:
            agent = RecurrentSACAgent(rsac_config)
            agent.load(checkpoint_path)
            agents = [agent]
    elif algo == "rsac_role":
        rsac_role_config = RoleAwareRecurrentSACConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        if is_four_agent_env:
            agents = []
            for _ in range(4):
                a = RoleAwareRecurrentSACAgent(rsac_role_config)
                a.load(checkpoint_path)
                agents.append(a)
        else:
            agent = RoleAwareRecurrentSACAgent(rsac_role_config)
            agent.load(checkpoint_path)
            agents = [agent]
    elif algo == "rtd3":
        rtd3_config = RecurrentTD3Config(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        if is_four_agent_env:
            agents = []
            for _ in range(4):
                a = RecurrentTD3Agent(rtd3_config)
                a.load(checkpoint_path)
                agents.append(a)
        else:
            agent = RecurrentTD3Agent(rtd3_config)
            agent.load(checkpoint_path)
            agents = [agent]
    elif algo == "rtd3_role":
        role_rtd3_config = RoleAwareRecurrentTD3Config(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        if is_four_agent_env:
            agents = []
            for _ in range(4):
                a = RoleAwareRecurrentTD3Agent(role_rtd3_config)
                a.load(checkpoint_path)
                agents.append(a)
        else:
            agent = RoleAwareRecurrentTD3Agent(role_rtd3_config)
            agent.load(checkpoint_path)
            agents = [agent]
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Running {num_episodes} episodes...")
    
    # 統計情報
    episode_rewards_agent1 = []  # team1 (chaser) 平均リターン
    episode_rewards_agent2 = []  # team2 (evader) 平均リターン
    episode_lengths = []
    agent1_wins = 0
    agent2_wins = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()

        # Recurrent agents: 各インスタンスごとにhiddenをリセット
        if algo in recurrent_algos:
            for a in agents:
                if hasattr(a, "reset_eval_hidden"):
                    a.reset_eval_hidden()
        
        # MuJoCoビューアーの初期化（AntSumoEnv かつレンダリング有効時）
        viewer = None
        if render and isinstance(env, AntSumoEnv):
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.distance = 15.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
        
        episode_reward_agent1 = 0.0
        episode_reward_agent2 = 0.0
        step_count = 0
        
        print(f"\n[Episode {episode + 1}/{num_episodes}]")
        
        while True:
            if not is_four_agent_env:
                # 2体環境: 従来通り agent1/agent2 を同じモデルで制御
                base_agent = agents[0]
                obs1 = obs
                action1 = base_agent.select_action(obs1, deterministic=deterministic)

                obs2 = env._get_obs(agent_id=2)
                action2 = base_agent.select_action(obs2, deterministic=deterministic)

                obs, reward1, terminated, truncated, info = env.step({
                    "agent1": action1,
                    "agent2": action2,
                })

                episode_reward_agent1 += reward1
                episode_reward_agent2 += info.get("agent2_reward", 0.0)
            else:
                # 4体環境: agent1,agent3=chasers / agent2,agent4=evaders
                # obs_k をそれぞれ取得し、対応するエージェントインスタンスで行動を選択
                actions = {}
                if algo in recurrent_algos:
                    # recurrent系は4インスタンスを使い分け
                    for idx, agent_id in enumerate(range(1, 5)):
                        o = env._get_obs(agent_id=agent_id)
                        a = agents[idx].select_action(o, deterministic=deterministic)
                        actions[f"agent{agent_id}"] = a

                        # role-aware 系 (rtd3_role / rsac_role / rppo_role) の場合、
                        # 各エージェントの予測を環境に渡す
                        if algo in {"rtd3_role", "rsac_role", "rppo_role"} and hasattr(
                            env, "set_agent_role_prediction"
                        ):
                            ag = agents[idx]
                            if hasattr(ag, "last_role_probs") and ag.last_role_probs is not None:
                                env.set_agent_role_prediction(agent_id, ag.last_role_probs)
                else:
                    base_agent = agents[0]
                    for agent_id in range(1, 5):
                        o = env._get_obs(agent_id=agent_id)
                        a = base_agent.select_action(o, deterministic=deterministic)
                        actions[f"agent{agent_id}"] = a

                # env.step は obs(agent1), reward1, ... を返す
                obs, reward1, terminated, truncated, info = env.step(actions)

                # チームごとの平均報酬を集計
                r1 = info.get("agent1_reward", 0.0)
                r2 = info.get("agent2_reward", 0.0)
                r3 = info.get("agent3_reward", 0.0)
                r4 = info.get("agent4_reward", 0.0)
                team1_step = 0.5 * (r1 + r3)
                team2_step = 0.5 * (r2 + r4)
                episode_reward_agent1 += team1_step
                episode_reward_agent2 += team2_step
            step_count += 1
            
            # レンダリング
            if render:
                if isinstance(env, AntSumoEnv) and viewer is not None:
                    # MuJoCoレンダリング
                    viewer.sync()
                else:
                    # 非MuJoCo環境では env.render() に委譲
                    env.render()
                time.sleep(1.0 / fps)
            
            if terminated or truncated:
                # 勝者の判定
                winner = info.get("winner", "none")
                if winner == "agent1":
                    agent1_wins += 1
                    print(f"  Winner: Agent 1 (Blue)")
                elif winner == "agent2":
                    agent2_wins += 1
                    print(f"  Winner: Agent 2 (Red)")
                else:
                    print(f"  Draw (time limit)")
                
                print(f"  Agent 1 Reward: {episode_reward_agent1:.2f}")
                print(f"  Agent 2 Reward: {episode_reward_agent2:.2f}")
                print(f"  Episode Length: {step_count} steps")
                break
        
        if viewer is not None:
            viewer.close()
        
        episode_rewards_agent1.append(episode_reward_agent1)
        episode_rewards_agent2.append(episode_reward_agent2)
        episode_lengths.append(step_count)
    
    # 統計情報の表示
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Agent 1 Wins: {agent1_wins} ({agent1_wins/num_episodes*100:.1f}%)")
    print(f"Agent 2 Wins: {agent2_wins} ({agent2_wins/num_episodes*100:.1f}%)")
    print(f"Draws: {num_episodes - agent1_wins - agent2_wins}")
    print(f"\nAgent 1 Average Reward: {np.mean(episode_rewards_agent1):.2f} ± {np.std(episode_rewards_agent1):.2f}")
    print(f"Agent 2 Average Reward: {np.mean(episode_rewards_agent2):.2f} ± {np.std(episode_rewards_agent2):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained RL agent on various environments")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to model checkpoint (if not specified, uses latest)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to test")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--fps", type=int, default=50,
                        help="Frames per second for rendering")
    parser.add_argument("--env-id", type=str, default="ant_sumo",
                        choices=ENV_IDS,
                        help=f"Environment ID to use (default: ant_sumo). Choices: {', '.join(ENV_IDS)}")
    parser.add_argument("--algo", type=str, default="ppo",
            choices=["ppo", "rppo", "sac", "td3", "rsac", "rtd3", "rtd3_role", "rsac_role", "rppo_role"],
            help=(
                "RL algorithm used for the checkpoint (default: ppo; "
                "rppo/rsac/rtd3/rtd3_role/rsac_role/rppo_role = recurrent variants)"
            ))
    
    args = parser.parse_args()

    # チェックポイントの決定（タスク・アルゴリズムごとのサブディレクトリを優先）
    if args.checkpoint is None:
        # まずは train.py での規約に合わせて探索
        if args.algo == "ppo":
            task_ckpt_dir = os.path.join(args.checkpoint_dir, args.env_id)
        else:
            task_ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.env_id}_{args.algo}")
        checkpoint_path = find_latest_checkpoint(task_ckpt_dir)

        # 見つからなければ旧形式の checkpoint_dir 直下も試す
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)

        if checkpoint_path is None:
            print(f"[ERROR] No checkpoint found in {task_ckpt_dir} or {args.checkpoint_dir}")
            sys.exit(1)
    else:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    test_policy(
        checkpoint_path=checkpoint_path,
        num_episodes=args.num_episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
        fps=args.fps,
        env_id=args.env_id,
        algo=args.algo,
    )
