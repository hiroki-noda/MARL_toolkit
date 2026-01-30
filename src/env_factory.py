"""環境IDから対応するマルチエージェント環境を生成するヘルパ

train.py / test_policy.py 双方から利用して、
- AntSumoEnv (MuJoCo)
- IteratedPrisonersDilemmaEnv
- MatchingPenniesEnv
- ContinuousCoordinationEnv
などを切り替えられるようにする。
"""

from typing import Callable, List

from sumo_env import AntSumoEnv
from ma_envs import (
    IteratedPrisonersDilemmaEnv,
    MatchingPenniesEnv,
    ContinuousCoordinationEnv,
    CooperativeNavigationEnv,
    AdversarialTagEnv,
    AdversarialTagFourAgentEnv,
)


ENV_IDS: List[str] = [
    "ant_sumo",
    "ipd",
    "matching_pennies",
    "coordination",
    "coop_nav",
    "adversarial_tag",
    "adversarial_tag_4p",
]


def create_env(env_id: str):
    """env_id に応じて環境インスタンスを生成する。

    Args:
        env_id: "ant_sumo" | "ipd" | "matching_pennies" | "coordination" | "coop_nav" | "adversarial_tag"
    """
    if env_id == "ant_sumo":
        # 既存の AntSumoEnv と同じデフォルト設定
        return AntSumoEnv(arena_radius=5.0, max_episode_steps=1000)
    if env_id == "ipd":
        return IteratedPrisonersDilemmaEnv(max_episode_steps=10)
    if env_id == "matching_pennies":
        return MatchingPenniesEnv(max_episode_steps=10)
    if env_id == "coordination":
        return ContinuousCoordinationEnv(max_episode_steps=10)
    if env_id == "coop_nav":
        # 2D 協調ナビゲーション環境
        return CooperativeNavigationEnv(max_episode_steps=50)
    if env_id == "adversarial_tag":
        # 2D 対戦型タグ環境
        return AdversarialTagEnv(max_episode_steps=100)
    if env_id == "adversarial_tag_4p":
        # 2D 対戦型タグ環境（追跡者2体 vs 逃走者2体）
        return AdversarialTagFourAgentEnv(max_episode_steps=100)

    raise ValueError(f"Unknown env_id: {env_id}")


def make_env_fns(env_id: str, num_envs: int) -> List[Callable[[], object]]:
    """VectorizedEnv 用に env_fns のリストを作成するヘルパ"""
    return [lambda env_id=env_id: create_env(env_id) for _ in range(num_envs)]
