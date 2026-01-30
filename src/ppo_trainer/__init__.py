"""Reinforcement learning package (self-play friendly).

Currently includes:
- PPO (on-policy, self-play trainer)
- SAC (off-policy continuous control)
- TD3 (off-policy continuous control)
"""

from ppo_trainer.ppo import PPOAgent
from ppo_trainer.trainer import SelfPlayTrainer, SelfPlayMultiAgentTrainer
from ppo_trainer.vec_env import VectorizedEnv, MultiAgentVectorizedEnv
from ppo_trainer.sac import SoftActorCriticAgent, SACConfig, ReplayBuffer
from ppo_trainer.td3 import TD3Agent, TD3Config
from ppo_trainer.offpolicy_trainer import (
	SelfPlayOffPolicyTrainer,
	SelfPlayOffPolicyRecurrentTrainer,
	SelfPlayOffPolicyMultiAgentTrainer,
	SelfPlayOffPolicyRecurrentMultiAgentTrainer,
)
from ppo_trainer.recurrent_ppo import (
	RecurrentPPOAgent,
	SelfPlayRecurrentPPOTrainer,
	SelfPlayRecurrentMultiAgentTrainer,
)
from ppo_trainer.sequence_replay import SequenceReplayBuffer
from ppo_trainer.recurrent_sac import RecurrentSACAgent, RecurrentSACConfig
from ppo_trainer.recurrent_td3 import RecurrentTD3Agent, RecurrentTD3Config
from ppo_trainer.role_rtd3 import RoleAwareRecurrentTD3Agent, RoleAwareRecurrentTD3Config
from ppo_trainer.role_rsac import RoleAwareRecurrentSACAgent, RoleAwareRecurrentSACConfig
from ppo_trainer.role_rppo import RoleAwareRecurrentPPOAgent, RoleAwareRecurrentPPOConfig

__version__ = "0.5.0"
__all__ = [
	"PPOAgent",
	"SelfPlayTrainer",
	"SelfPlayMultiAgentTrainer",
	"VectorizedEnv",
	"MultiAgentVectorizedEnv",
	"SoftActorCriticAgent",
	"SACConfig",
	"ReplayBuffer",
	"TD3Agent",
	"TD3Config",
	"SelfPlayOffPolicyTrainer",
	"SelfPlayOffPolicyRecurrentTrainer",
	"SelfPlayOffPolicyMultiAgentTrainer",
	"SelfPlayOffPolicyRecurrentMultiAgentTrainer",
	"RecurrentPPOAgent",
	"SelfPlayRecurrentPPOTrainer",
	"SelfPlayRecurrentMultiAgentTrainer",
	"SequenceReplayBuffer",
	"RecurrentSACAgent",
	"RecurrentSACConfig",
	"RecurrentTD3Agent",
	"RecurrentTD3Config",
	"RoleAwareRecurrentTD3Agent",
	"RoleAwareRecurrentTD3Config",
	"RoleAwareRecurrentSACAgent",
	"RoleAwareRecurrentSACConfig",
	"RoleAwareRecurrentPPOAgent",
	"RoleAwareRecurrentPPOConfig",
]

