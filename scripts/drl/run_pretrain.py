"""
scripts/drl/run_pretrain.py
One-time pre-training. Run before run.py.
Usage: set PYTHONPATH=<repo> && python scripts/drl/run_pretrain.py
"""

import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("synera.pretrain")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drl.agent import SyneraTriageAgent
from drl.triage_env import SyneraTriageEnv, ACTION_LABELS
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
import numpy as np

PRETRAIN_TIMESTEPS = 100_000
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "drl"
MODEL_PATH = MODEL_DIR / "synera_triage_ppo.zip"


def run_pretrain():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("SYNERA DRL PRE-TRAINING")
    logger.info("Timesteps: %s", f"{PRETRAIN_TIMESTEPS:,}")
    logger.info("=" * 60)

    train_env = make_vec_env(SyneraTriageEnv, n_envs=4, seed=42)
    eval_env  = make_vec_env(SyneraTriageEnv, n_envs=1, seed=99)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(MODEL_DIR / "logs"),
        eval_freq=5_000,
        n_eval_episodes=200,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    logger.info("Training started...")
    start = time.time()
    model.learn(total_timesteps=PRETRAIN_TIMESTEPS, callback=eval_callback)
    elapsed = time.time() - start

    model.save(str(MODEL_PATH))
    logger.info("Training complete in %.1fs. Model saved: %s", elapsed, MODEL_PATH)

    env = SyneraTriageEnv()
    results = {0: [], 1: [], 2: []}
    for _ in range(1000):
        obs, info = env.reset()
        true_sev  = info["true_severity"]
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        results[true_sev].append(int(action) == true_sev)

    logger.info("Pre-training accuracy:")
    labels = ["LOW → ELEVATED", "MEDIUM → URGENT", "HIGH → IMMEDIATE"]
    for sev, label in enumerate(labels):
        acc = sum(results[sev]) / len(results[sev]) * 100 if results[sev] else 0
        logger.info("  %s: %.1f%%", label, acc)


if __name__ == "__main__":
    run_pretrain()
