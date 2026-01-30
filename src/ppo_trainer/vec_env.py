"""
並列環境のラッパー
"""

import numpy as np
from typing import List, Tuple
import multiprocessing as mp
from multiprocessing import Pipe
import sys


def worker(remote, parent_remote, env_fn):
    """環境を実行するワーカープロセス"""
    parent_remote.close()
    try:
        env = env_fn()
    except Exception as e:
        import traceback
        remote.send(('error', traceback.format_exc()))
        remote.close()
        return
    
    try:
        env = env_fn()
    except Exception as e:
        import traceback
        remote.send(('error', traceback.format_exc()))
        remote.close()
        return
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                # 両エージェントの行動を適用
                action_dict = {
                    "agent1": data[0],
                    "agent2": data[1]
                }
                obs, reward, terminated, truncated, info = env.step(action_dict)
                
                # agent1の観測を返す
                obs1 = obs
                # agent2の観測を取得
                obs2 = env._get_obs(agent_id=2)
                
                done = terminated or truncated
                
                if done:
                    final_obs1 = obs1
                    final_obs2 = obs2
                    obs1, _ = env.reset()
                    obs2 = env._get_obs(agent_id=2)
                    remote.send(((obs1, obs2), (info["agent1_reward"], info["agent2_reward"]), done, info, (final_obs1, final_obs2)))
                else:
                    remote.send(((obs1, obs2), (info["agent1_reward"], info["agent2_reward"]), done, info, None))
                    
            elif cmd == 'reset':
                obs, _ = env.reset()
                obs1 = obs
                obs2 = env._get_obs(agent_id=2)
                remote.send((obs1, obs2))
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
        except Exception as e:
            import traceback
            remote.send(('error', traceback.format_exc()))
            remote.close()
            break


def multi_agent_worker(remote, parent_remote, env_fn, num_agents: int):
    """num_agents 体のマルチエージェント環境を実行するワーカープロセス

    環境側は次のインターフェイスを満たしていることを想定:
    - reset() -> (obs_agent1, info)
    - _get_obs(agent_id=k) で k番目エージェントの観測を取得
    - step({"agent1": a1, ..., "agentN": aN}) ->
        obs_agent1, reward1, terminated, truncated, info
      info には "agent1_reward"〜"agentN_reward" を含む。
    """
    parent_remote.close()
    try:
        env = env_fn()
    except Exception:
        import traceback
        remote.send(("error", traceback.format_exc()))
        remote.close()
        return

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == "step":
                # data: (num_agents, action_dim)
                actions = np.asarray(data, dtype=np.float32)
                action_dict = {f"agent{i+1}": actions[i] for i in range(num_agents)}

                obs1, reward1, terminated, truncated, info = env.step(action_dict)

                # 全エージェントの観測を取得
                obs_list = [env._get_obs(agent_id=i + 1) for i in range(num_agents)]
                obs_arr = np.stack(obs_list, axis=0)

                # 報酬を info から取得（無ければ0）
                rewards = np.array(
                    [info.get(f"agent{i+1}_reward", float(reward1 if i == 0 else 0.0)) for i in range(num_agents)],
                    dtype=np.float32,
                )

                done = terminated or truncated

                if done:
                    final_obs = obs_arr.copy()
                    # リセットして次エピソード開始時の観測を返す
                    _, _ = env.reset()
                    reset_obs_list = [env._get_obs(agent_id=i + 1) for i in range(num_agents)]
                    reset_obs_arr = np.stack(reset_obs_list, axis=0)
                    remote.send((reset_obs_arr, rewards, done, info, final_obs))
                else:
                    remote.send((obs_arr, rewards, done, info, None))

            elif cmd == "reset":
                _, _ = env.reset()
                obs_list = [env._get_obs(agent_id=i + 1) for i in range(num_agents)]
                obs_arr = np.stack(obs_list, axis=0)
                remote.send(obs_arr)

            elif cmd == "close":
                env.close()
                remote.close()
                break
        except Exception:
            import traceback
            remote.send(("error", traceback.format_exc()))
            remote.close()
            break
                
class VectorizedEnv:
    """並列環境のラッパー"""
    
    def __init__(self, env_fns: List):
        """
        Args:
            env_fns: 環境を作成する関数のリスト
        """
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = mp.Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        全環境をリセット
        Returns:
            obs1: Agent1の観測 (num_envs, obs_dim)
            obs2: Agent2の観測 (num_envs, obs_dim)
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = []
        for i, remote in enumerate(self.remotes):
            result = remote.recv()
            # エラーチェック（numpy配列との比較を避ける）
            if isinstance(result, tuple) and len(result) == 2:
                try:
                    if isinstance(result[0], str) and result[0] == 'error':
                        error_msg = f"\n{'='*60}\nError in environment {i} during reset:\n{'='*60}\n{result[1]}\n{'='*60}"
                        raise RuntimeError(error_msg)
                except (ValueError, TypeError):
                    pass  # 比較失敗した場合は正常なデータとして扱う
            results.append(result)
        
        obs1 = np.stack([r[0] for r in results])
        obs2 = np.stack([r[1] for r in results])
        
        return obs1, obs2
    
    def step(self, actions1: np.ndarray, actions2: np.ndarray):
        """
        全環境で1ステップ実行
        Args:
            actions1: Agent1の行動 (num_envs, action_dim)
            actions2: Agent2の行動 (num_envs, action_dim)
        Returns:
            obs: 両エージェントの観測
            rewards: 両エージェントの報酬
            dones: 終了フラグ
            infos: 情報
            final_obs: エピソード終了時の最終観測
        """
        for remote, action1, action2 in zip(self.remotes, actions1, actions2):
            remote.send(('step', (action1, action2)))
        
        results = []
        for i, remote in enumerate(self.remotes):
            result = remote.recv()
            # エラーチェック（numpy配列との比較を避ける）
            if isinstance(result, tuple) and len(result) == 2:
                try:
                    if isinstance(result[0], str) and result[0] == 'error':
                        error_msg = f"\n{'='*60}\nError in environment {i} during step:\n{'='*60}\n{result[1]}\n{'='*60}"
                        raise RuntimeError(error_msg)
                except (ValueError, TypeError):
                    pass  # 比較失敗した場合は正常なデータとして扱う
            results.append(result)
        
        obs1 = np.stack([r[0][0] for r in results])
        obs2 = np.stack([r[0][1] for r in results])
        rewards1 = np.array([r[1][0] for r in results])
        rewards2 = np.array([r[1][1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        
        # エピソード終了時の最終観測を取得
        final_obs1 = []
        final_obs2 = []
        for r in results:
            if r[4] is not None:
                final_obs1.append(r[4][0])
                final_obs2.append(r[4][1])
            else:
                final_obs1.append(None)
                final_obs2.append(None)
        
        return (obs1, obs2), (rewards1, rewards2), dones, infos, (final_obs1, final_obs2)
    
    def close(self):
        """全環境を閉じる"""
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()


class MultiAgentVectorizedEnv:
    """num_agents 体のマルチエージェント環境用 並列ラッパー

    各ワーカー環境は num_agents 個のエージェントを持ち、
    reset/step では (num_envs, num_agents, ...) 形式でやり取りする。
    """

    def __init__(self, env_fns: List, num_agents: int):
        self.num_envs = len(env_fns)
        self.num_agents = num_agents
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])

        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn, num_agents)
            process = mp.Process(target=multi_agent_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def reset(self) -> np.ndarray:
        """全環境をリセット

        Returns:
            obs: 形状 (num_envs, num_agents, obs_dim)
        """
        for remote in self.remotes:
            remote.send(("reset", None))

        results = []
        for i, remote in enumerate(self.remotes):
            result = remote.recv()
            if isinstance(result, tuple) and len(result) == 2:
                try:
                    if isinstance(result[0], str) and result[0] == "error":
                        error_msg = f"\n{'='*60}\nError in environment {i} during reset (multi-agent):\n{'='*60}\n{result[1]}\n{'='*60}"
                        raise RuntimeError(error_msg)
                except (ValueError, TypeError):
                    pass
            results.append(result)

        obs = np.stack(results, axis=0)
        return obs

    def step(self, actions: np.ndarray):
        """全環境で1ステップ実行

        Args:
            actions: 形状 (num_envs, num_agents, action_dim)

        Returns:
            obs: (num_envs, num_agents, obs_dim)
            rewards: (num_envs, num_agents)
            dones: (num_envs,)
            infos: list[dict]
            final_obs: list[np.ndarray | None] 各envごとのエピソード終了時観測
        """
        for remote, act in zip(self.remotes, actions):
            remote.send(("step", act))

        results = []
        for i, remote in enumerate(self.remotes):
            result = remote.recv()
            if isinstance(result, tuple) and len(result) == 2:
                try:
                    if isinstance(result[0], str) and result[0] == "error":
                        error_msg = f"\n{'='*60}\nError in environment {i} during step (multi-agent):\n{'='*60}\n{result[1]}\n{'='*60}"
                        raise RuntimeError(error_msg)
                except (ValueError, TypeError):
                    pass
            results.append(result)

        obs = np.stack([r[0] for r in results], axis=0)
        rewards = np.stack([r[1] for r in results], axis=0)
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        final_obs = []
        for r in results:
            if r[4] is not None:
                final_obs.append(r[4])
            else:
                final_obs.append(None)

        return obs, rewards, dones, infos, final_obs

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
