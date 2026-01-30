"""シンプルなマルチエージェント環境群

現在含まれている環境:
- IteratedPrisonersDilemmaEnv: 連続行動から協調/裏切りに二値化した繰り返し囚人のジレンマ
- MatchingPenniesEnv: 連続行動から表/裏に二値化したマッチングペニーズ
- ContinuousCoordinationEnv: 連続値を揃える協調ゲーム
- CooperativeNavigationEnv: 2D 空間で原点に近づく協調ナビゲーション
- AdversarialTagEnv: 2D 空間で追跡者と逃走者が対戦するタグゲーム
"""

from ma_envs.matrix_games import (
    IteratedPrisonersDilemmaEnv,
    MatchingPenniesEnv,
    ContinuousCoordinationEnv,
)
from ma_envs.coop_navigation_env import CooperativeNavigationEnv
from ma_envs.adversarial_tag_env import AdversarialTagEnv, AdversarialTagFourAgentEnv

__all__ = [
    "IteratedPrisonersDilemmaEnv",
    "MatchingPenniesEnv",
    "ContinuousCoordinationEnv",
    "CooperativeNavigationEnv",
    "AdversarialTagEnv",
    "AdversarialTagFourAgentEnv",
]
