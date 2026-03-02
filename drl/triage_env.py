"""
drl/triage_env.py
Custom Gymnasium environment for the Synera triage decision problem.
Each episode = one alert event.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

ACTION_ELEVATED  = 0
ACTION_URGENT    = 1
ACTION_IMMEDIATE = 2
ACTION_LABELS    = {0: "ELEVATED", 1: "URGENT", 2: "IMMEDIATE"}

IDX_HR_SIGMA, IDX_SPO2_SIGMA, IDX_TEMP_SIGMA = 0, 1, 2
IDX_BP_SIGMA, IDX_MOTION                      = 3, 4
IDX_HR_FIRST_DERIV, IDX_HR_SECOND_DERIV       = 5, 6
IDX_SPO2_SECOND_DERIV                         = 7
IDX_AGE_NORM                                  = 8
IDX_CARDIAC_RISK, IDX_RESP_RISK               = 9, 10
IDX_DIABETIC_RISK, IDX_SEPSIS_RISK            = 11, 12
IDX_TIME_SIN, IDX_TIME_COS                    = 13, 14
IDX_CONCURRENT_ALERTS, IDX_WARD_OCCUPANCY     = 15, 16
STATE_DIM = 17

REWARD_ATTENDED_5MIN   = +1.0
REWARD_ATTENDED_15MIN  = +0.5
REWARD_DISMISSED       = -0.5
REWARD_MISSED_CRITICAL = -1.0
REWARD_TIMEOUT         = -0.3


class SyneraTriageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._current_state: Optional[np.ndarray] = None
        self._true_severity: Optional[int] = None
        self._episode_count: int = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._true_severity, self._current_state = self._generate_scenario()
        self._episode_count += 1
        return self._current_state.astype(np.float32), {
            "episode": self._episode_count,
            "true_severity": self._true_severity,
            "severity_label": ["LOW", "MEDIUM", "HIGH"][self._true_severity],
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._current_state is not None, "Call reset() first"
        reward = self._compute_reward(action, self._true_severity)
        info = {
            "action": action,
            "action_label": ACTION_LABELS[action],
            "true_severity": self._true_severity,
            "severity_label": ["LOW", "MEDIUM", "HIGH"][self._true_severity],
            "reward": reward,
        }
        if self.render_mode == "human":
            print(
                f"[Triage] Severity={info['severity_label']} | "
                f"Action={info['action_label']} | Reward={reward:+.1f}"
            )
        return self._current_state.astype(np.float32), reward, True, False, info

    def _generate_scenario(self) -> Tuple[int, np.ndarray]:
        rng      = self.np_random
        severity = int(rng.integers(0, 3))
        s        = np.zeros(STATE_DIM, dtype=np.float32)

        if severity == 0:
            s[IDX_HR_SIGMA]          = rng.uniform(1.5, 2.0)
            s[IDX_SPO2_SIGMA]        = rng.uniform(0.0, 0.5)
            s[IDX_TEMP_SIGMA]        = rng.uniform(0.0, 0.3)
            s[IDX_BP_SIGMA]          = rng.uniform(0.0, 0.5)
            s[IDX_MOTION]            = rng.uniform(0.0, 0.2)
            s[IDX_HR_FIRST_DERIV]    = rng.uniform(0.1, 0.3)
            s[IDX_HR_SECOND_DERIV]   = rng.uniform(0.05, 0.15)
            s[IDX_SPO2_SECOND_DERIV] = rng.uniform(-0.05, 0.05)
            s[IDX_AGE_NORM]          = rng.uniform(0.2, 0.4)
            s[IDX_CARDIAC_RISK]      = 0.1
            s[IDX_RESP_RISK]         = 0.1
            s[IDX_DIABETIC_RISK]     = 0.1
            s[IDX_SEPSIS_RISK]       = 0.1
            hour = int(rng.integers(8, 20))
        elif severity == 1:
            s[IDX_HR_SIGMA]          = rng.uniform(2.0, 3.5)
            s[IDX_SPO2_SIGMA]        = rng.uniform(0.5, 1.5)
            s[IDX_TEMP_SIGMA]        = rng.uniform(0.3, 1.0)
            s[IDX_BP_SIGMA]          = rng.uniform(0.5, 1.5)
            s[IDX_MOTION]            = rng.uniform(0.0, 0.2)
            s[IDX_HR_FIRST_DERIV]    = rng.uniform(0.3, 0.6)
            s[IDX_HR_SECOND_DERIV]   = rng.uniform(0.2, 0.4)
            s[IDX_SPO2_SECOND_DERIV] = rng.uniform(0.1, 0.3)
            s[IDX_AGE_NORM]          = rng.uniform(0.4, 0.65)
            s[IDX_CARDIAC_RISK]      = float(rng.choice([0.1, 0.4, 0.7]))
            s[IDX_RESP_RISK]         = float(rng.choice([0.1, 0.4]))
            s[IDX_DIABETIC_RISK]     = float(rng.choice([0.1, 0.4]))
            s[IDX_SEPSIS_RISK]       = float(rng.choice([0.1, 0.4, 0.7]))
            hour = int(rng.integers(0, 24))
        else:
            s[IDX_HR_SIGMA]          = rng.uniform(3.5, 6.0)
            s[IDX_SPO2_SIGMA]        = rng.uniform(2.0, 5.0)
            s[IDX_TEMP_SIGMA]        = rng.uniform(1.0, 3.0)
            s[IDX_BP_SIGMA]          = rng.uniform(1.5, 4.0)
            s[IDX_MOTION]            = rng.uniform(0.0, 0.1)
            s[IDX_HR_FIRST_DERIV]    = rng.uniform(0.6, 1.0)
            s[IDX_HR_SECOND_DERIV]   = rng.uniform(0.5, 1.0)
            s[IDX_SPO2_SECOND_DERIV] = rng.uniform(0.4, 1.0)
            s[IDX_AGE_NORM]          = rng.uniform(0.55, 0.85)
            s[IDX_CARDIAC_RISK]      = float(rng.choice([0.7, 1.0]))
            s[IDX_RESP_RISK]         = float(rng.choice([0.4, 0.7, 1.0]))
            s[IDX_DIABETIC_RISK]     = float(rng.choice([0.4, 0.7]))
            s[IDX_SEPSIS_RISK]       = float(rng.choice([0.7, 1.0]))
            hour = int(rng.integers(0, 8))

        s[IDX_TIME_SIN]          = float(np.sin(2 * np.pi * hour / 24))
        s[IDX_TIME_COS]          = float(np.cos(2 * np.pi * hour / 24))
        s[IDX_CONCURRENT_ALERTS] = float(rng.integers(0, 5)) / 20.0
        s[IDX_WARD_OCCUPANCY]    = float(rng.integers(10, 50)) / 50.0
        return severity, s

    def _compute_reward(self, action: int, true_severity: int) -> float:
        diff = action - true_severity
        if diff == 0:
            return REWARD_ATTENDED_5MIN if true_severity == 2 else REWARD_ATTENDED_15MIN
        elif diff == -1:
            return REWARD_DISMISSED
        elif diff == 1:
            return -0.2
        elif diff <= -2:
            return REWARD_MISSED_CRITICAL
        else:
            return -0.4
