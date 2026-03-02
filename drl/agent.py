"""
drl/agent.py
Synera PPO Triage Agent — wrapper around Stable-Baselines3 PPO.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from drl.triage_env import SyneraTriageEnv, ACTION_LABELS

logger = logging.getLogger("synera.drl.agent")

MODEL_DIR   = Path(__file__).resolve().parent.parent / "models" / "drl"
MODEL_PATH  = MODEL_DIR / "synera_triage_ppo.zip"

PPO_HYPERPARAMS = {
    "policy":          "MlpPolicy",
    "learning_rate":   3e-4,
    "n_steps":         512,
    "batch_size":      64,
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.01,
    "verbose":         0,
}


class SyneraTriageAgent:
    """PPO-based triage priority agent."""

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model: Optional[PPO] = None
        self._env = None
        self._confidence_threshold = 0.6

    def load(self) -> bool:
        """Load pre-trained model weights. Returns True if successful."""
        if MODEL_PATH.exists():
            try:
                self.model = PPO.load(str(MODEL_PATH))
                logger.info("[DRL] Loaded model from %s", MODEL_PATH)
                return True
            except Exception as e:
                logger.error("[DRL] Failed to load model: %s", e)
                self.model = None
                return False
        logger.warning("[DRL] No model at %s. Run pretrain first.", MODEL_PATH)
        return False

    def predict(self, state_vector: np.ndarray) -> Tuple[str, float]:
        """Predict priority tier from state vector. Returns (action_label, confidence)."""
        if self.model is None:
            return self._rule_based_fallback(state_vector)

        try:
            obs = state_vector.reshape(1, -1).astype(np.float32)
            action, _states = self.model.predict(obs, deterministic=True)
            action_int = int(action)

            import torch
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.numpy()[0]
                confidence = float(probs[action_int])

            if confidence < self._confidence_threshold:
                return self._rule_based_fallback(state_vector)

            label = ACTION_LABELS[action_int]
            return label, confidence

        except Exception as e:
            logger.error("[DRL] Prediction error: %s", e)
            return self._rule_based_fallback(state_vector)

    def train(self, total_timesteps: int = 50_000) -> None:
        """Train (or continue training) the PPO agent."""
        env = make_vec_env(SyneraTriageEnv, n_envs=4, seed=42)

        if self.model is None:
            self.model = PPO(env=env, **PPO_HYPERPARAMS)
        else:
            self.model.set_env(env)

        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self.save()

    def fine_tune(self, experiences: List[Tuple[np.ndarray, int, float]]) -> None:
        """Fine-tune on a batch of real experiences."""
        if not experiences or self.model is None:
            logger.warning("[DRL] fine_tune: no experiences or no model")
            return

        from drl.replay_env import ExperienceReplayEnv

        replay_env = make_vec_env(
            lambda: ExperienceReplayEnv(experiences),
            n_envs=1
        )
        self.model.set_env(replay_env)
        fine_tune_steps = max(len(experiences) * 4, 512)
        self.model.learn(total_timesteps=fine_tune_steps, reset_num_timesteps=False)
        self.save()

    def save(self) -> None:
        """Save model weights to disk."""
        if self.model is not None:
            self.model.save(str(MODEL_PATH))

    def _rule_based_fallback(self, state_vector: np.ndarray) -> Tuple[str, float]:
        """Rule-based triage when model not loaded or low confidence."""
        hr_sigma    = float(state_vector[0])
        spo2_sigma  = float(state_vector[1])
        sepsis_risk = float(state_vector[12])
        score = hr_sigma + spo2_sigma + (sepsis_risk * 2)
        if score > 8.0:
            return "IMMEDIATE", 0.0
        elif score > 4.0:
            return "URGENT", 0.0
        return "ELEVATED", 0.0


triage_agent = SyneraTriageAgent()
