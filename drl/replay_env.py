"""
drl/replay_env.py
Deterministic replay environment for fine-tuning PPO on real experiences.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Optional, Dict

Experience = Tuple[np.ndarray, int, float]
STATE_DIM  = 17


class ExperienceReplayEnv(gym.Env):
    """Replays stored (state, action, reward) experiences in order."""

    def __init__(self, experiences: List[Experience]):
        super().__init__()
        assert len(experiences) > 0, "At least one experience required"
        self.experiences  = experiences
        self._idx         = 0
        self._true_action: Optional[int]   = None
        self._true_reward: Optional[float] = None

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        state, action, reward = self.experiences[self._idx % len(self.experiences)]
        self._idx         += 1
        self._true_action  = action
        self._true_reward  = reward
        return state.astype(np.float32), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = self._true_reward if self._true_reward is not None else 0.0
        state, _, _ = self.experiences[(self._idx - 1) % len(self.experiences)]
        return state.astype(np.float32), reward, True, False, {"replayed": True}
