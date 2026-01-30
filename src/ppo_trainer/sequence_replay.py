"""Sequence-based replay buffer for recurrent off-policy algorithms.

Each stored sample is a variable-length sequence (episode fragment)
of transitions. Sampling returns a batch of such sequences, and
the recurrent agent is responsible for handling variable lengths
via BPTT.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class SequenceSample:
    obs: np.ndarray          # (T, obs_dim)
    action: np.ndarray       # (T, action_dim)
    reward: np.ndarray       # (T, 1)
    next_obs: np.ndarray     # (T, obs_dim)
    done: np.ndarray         # (T, 1)
    agent_index: Optional[int] = None  # どのエージェントのシーケンスか（multi-agent 用）


class SequenceReplayBuffer:
    """Replay buffer storing variable-length sequences.

    Sequences are stored as numpy arrays. Capacity is expressed in
    number of sequences, not transitions.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[SequenceSample] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add_sequence(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        agent_index: Optional[int] = None,
    ) -> None:
        """Add a sequence to the buffer.

        All inputs must have the same leading dimension T.
        """

        T = obs.shape[0]
        assert action.shape[0] == T
        assert reward.shape[0] == T
        assert next_obs.shape[0] == T
        assert done.shape[0] == T

        # Ensure shapes (T, *) and reward/done as (T, 1)
        if reward.ndim == 1:
            reward = reward.reshape(T, 1)
        if done.ndim == 1:
            done = done.reshape(T, 1)

        sample = SequenceSample(
            obs=obs.astype(np.float32),
            action=action.astype(np.float32),
            reward=reward.astype(np.float32),
            next_obs=next_obs.astype(np.float32),
            done=done.astype(np.float32),
            agent_index=agent_index,
        )

        if len(self.buffer) >= self.capacity:
            # FIFO eviction
            self.buffer.pop(0)
        self.buffer.append(sample)

    def sample(self, batch_size: int, max_seq_len: int) -> List[SequenceSample]:
        """Sample a batch of (sub-)sequences.

        For each sampled stored sequence, we draw a random contiguous
        subsequence up to length max_seq_len.
        """

        if len(self.buffer) == 0:
            return []

        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        results: List[SequenceSample] = []
        for idx in indices:
            seq = self.buffer[idx]
            T = seq.obs.shape[0]
            if T <= max_seq_len:
                # Use full sequence
                results.append(seq)
            else:
                # Random subsequence of length max_seq_len
                start = np.random.randint(0, T - max_seq_len + 1)
                end = start + max_seq_len
                results.append(
                    SequenceSample(
                        obs=seq.obs[start:end],
                        action=seq.action[start:end],
                        reward=seq.reward[start:end],
                        next_obs=seq.next_obs[start:end],
                        done=seq.done[start:end],
                        agent_index=seq.agent_index,
                    )
                )

        return results
