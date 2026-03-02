"""
Synera 2.0 — AutoencoderTrainer
Full training/validation loop for the LSTM autoencoder with:
  - Mini-batch SGD via Adam
  - Cosine LR annealing
  - EarlyStopping + ModelCheckpoint callbacks
  - Anomaly threshold calibration (mean + 3σ over normal validation reconstructions)
  - Training curve plotting
  - --quick-demo CLI flag for fast local testing
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# ── repo-root bootstrap ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]   # ml/tinyml/training → ml/tinyml → ml → ArogyaLink
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from ml.tinyml.model.autoencoder import AutoencoderConfig, VitalAutoencoder
from ml.tinyml.training.callbacks import EarlyStopping, ModelCheckpoint
from ml.tinyml.training.dataset import VitalWindowDataset
from ml.tinyml.training.loss import ReconstructionLoss


class TrainingConfig:
    """All hyperparameters for one training run."""

    def __init__(
        self,
        *,
        epochs:        int   =  80,
        batch_size:    int   =  64,
        lr:            float =  1e-3,
        weight_decay:  float =  1e-5,
        val_fraction:  float =  0.15,
        patience:      int   =  15,
        min_delta:     float =  1e-5,
        threshold_sigma: float = 2.5,  # lowered from 3.0 → ~+2min lead time
        num_workers:   int   =   0,
        save_dir:      str   = "ml/tinyml/checkpoints",
        device:        str   = "auto",
    ):
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.val_fraction    = val_fraction
        self.patience        = patience
        self.min_delta       = min_delta
        self.threshold_sigma = threshold_sigma
        self.num_workers     = num_workers
        self.save_dir        = save_dir

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


class AutoencoderTrainer:
    """
    Manages end-to-end training of VitalAutoencoder.

    Typical usage:
        config  = TrainingConfig(epochs=80, batch_size=64)
        trainer = AutoencoderTrainer(config)
        model   = trainer.train(train_windows, val_windows=None)
        print(f"Threshold: {model.anomaly_threshold:.5f}")
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.cfg     = config or TrainingConfig()
        self.device  = torch.device(self.cfg.device)
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_windows:  np.ndarray,
        train_labels:   Optional[np.ndarray] = None,
        val_windows:    Optional[np.ndarray] = None,
        val_labels:     Optional[np.ndarray] = None,
        autoencoder_cfg: Optional[AutoencoderConfig] = None,
    ) -> VitalAutoencoder:
        """
        Train the VitalAutoencoder on `train_windows`.

        Args:
            train_windows:   (N, 10, 5) normal-only windows for unsupervised training
            train_labels:    (N,) labels (informational only; not used in loss)
            val_windows:     Optional override; if None, splits train set by val_fraction
            val_labels:      Optional
            autoencoder_cfg: Override model architecture

        Returns:
            Trained VitalAutoencoder with anomaly_threshold calibrated.
        """
        # ── Build model ───────────────────────────────────────────────────────
        ae_cfg = autoencoder_cfg or AutoencoderConfig()
        model  = VitalAutoencoder(ae_cfg).to(self.device)
        print(model.architecture_summary())

        # ── Datasets & loaders ────────────────────────────────────────────────
        train_ds, val_ds = self._split(train_windows, train_labels, val_windows, val_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = True,
            num_workers = self.cfg.num_workers,
            pin_memory  = (self.cfg.device == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = self.cfg.batch_size * 2,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
        )
        print(
            f"  Train windows : {len(train_ds):,}  "
            f"Val windows : {len(val_ds):,}  "
            f"Batches/epoch : {len(train_loader)}"
        )

        # ── Optimiser & LR schedule ───────────────────────────────────────────
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr           = self.cfg.lr,
            weight_decay = self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max  = self.cfg.epochs,
            eta_min = self.cfg.lr * 0.01,
        )
        criterion  = ReconstructionLoss(reduction="mean")
        stopper    = EarlyStopping(patience=self.cfg.patience, min_delta=self.cfg.min_delta)
        checkpoint = ModelCheckpoint(
            save_dir = Path(_ROOT) / self.cfg.save_dir,
            filename = "best_autoencoder.pt",
        )

        # ── Training loop ─────────────────────────────────────────────────────
        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        t0 = time.time()
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._run_epoch(model, train_loader, criterion, optimiser)
            val_loss   = self._validate(model, val_loader, criterion)
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            is_best = checkpoint.update(epoch, val_loss, model)
            star    = " ★" if is_best else ""
            if epoch % 5 == 0 or epoch <= 3 or is_best:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:3d}/{self.cfg.epochs} | "
                    f"train {train_loss:.5f} | val {val_loss:.5f} | "
                    f"lr {current_lr:.2e} | {elapsed:.1f}s{star}"
                )

            if stopper(val_loss):
                print(f"\n  Early stop triggered at epoch {epoch} (patience={self.cfg.patience})")
                break

        elapsed_total = time.time() - t0
        print(f"\n  Training complete in {elapsed_total:.1f}s — best val loss: {checkpoint.best_loss:.5f}")

        # ── Reload best weights ───────────────────────────────────────────────
        ckpt = torch.load(checkpoint.checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # ── Calibrate anomaly threshold ───────────────────────────────────────
        threshold = self._calibrate_threshold(model, val_loader)
        model.anomaly_threshold = threshold
        model.config.anomaly_threshold = threshold
        print(f"  Anomaly threshold (μ + {self.cfg.threshold_sigma}σ): {threshold:.5f}")

        # ── Save finalised model ──────────────────────────────────────────────
        model.save(Path(_ROOT) / self.cfg.save_dir / "autoencoder_final.pt")

        return model

    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot train/validation loss curves using matplotlib.

        Args:
            save_path: If provided, saves figure to file instead of showing.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not installed — skipping plot.")
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Synera 2.0 — Autoencoder Training", fontsize=13)

        ax = axes[0]
        ax.plot(epochs, self.history["train_loss"], label="Train MSE", color="#2196F3")
        ax.plot(epochs, self.history["val_loss"],   label="Val MSE",   color="#FF5722")
        ax.set_xlabel("Epoch");  ax.set_ylabel("MSE Loss")
        ax.set_title("Reconstruction Loss"); ax.legend(); ax.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.plot(epochs, self.history["lr"], color="#4CAF50")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate")
        ax2.set_title("LR Schedule (Cosine Annealing)"); ax2.grid(alpha=0.3)
        ax2.set_yscale("log")

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Curve saved → {save_path}")
        else:
            plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, model, loader, criterion, optimiser) -> float:
        model.train()
        total = 0.0
        for x, target, _labels in loader:
            x = x.to(self.device)
            optimiser.zero_grad()
            x_hat, _ = model(x)
            loss   = criterion(x_hat, x)
            loss.backward()
            # Gradient clipping — helps with LSTM instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total += loss.item() * x.size(0)
        return total / len(loader.dataset)

    def _validate(self, model, loader, criterion) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, target, _labels in loader:
                x    = x.to(self.device)
                x_hat, _ = model(x)
                loss   = criterion(x_hat, x)
                total += loss.item() * x.size(0)
        return total / len(loader.dataset)

    def _split(
        self,
        train_windows, train_labels,
        val_windows, val_labels,
    ):
        if val_windows is not None:
            train_ds = VitalWindowDataset(train_windows, train_labels)
            val_ds   = VitalWindowDataset(val_windows,   val_labels)
            return train_ds, val_ds

        full_ds = VitalWindowDataset(train_windows, train_labels)
        n_val   = int(len(full_ds) * self.cfg.val_fraction)
        n_train = len(full_ds) - n_val
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
        return train_ds, val_ds

    def _calibrate_threshold(self, model, val_loader) -> float:
        """
        Compute anomaly threshold = mean + threshold_sigma * std
        using validation set reconstruction errors.
        """
        errors = []
        model.eval()
        criterion = ReconstructionLoss(reduction="none")
        with torch.no_grad():
            for x, target, _labels in val_loader:
                x         = x.to(self.device)
                x_hat, _  = model(x)
                err       = criterion.per_sample(x_hat, x)
                errors.append(err.cpu().numpy())
        errors = np.concatenate(errors)
        mu     = float(np.mean(errors))
        sigma  = float(np.std(errors))
        return mu + self.cfg.threshold_sigma * sigma


# ─────────────────────────────────────────────────────────────────────────────
# CLI — python -m ml.tinyml.training.trainer --quick-demo
# ─────────────────────────────────────────────────────────────────────────────
def _quick_demo():
    """10-epoch sanity-check run using synthetic data."""
    print("\n" + "=" * 60)
    print("  Synera 2.0 — Autoencoder Quick-Demo Training")
    print("=" * 60)

    sys.path.insert(0, str(_ROOT))
    from scripts.data_gen.scenario_generator import prepare_training_data

    print("  Generating synthetic training windows …")
    X, labels = prepare_training_data(n_patients=10, minutes_per_patient=10)
    print(f"  Windows: {X.shape}  Labels: {labels.shape}")

    cfg     = TrainingConfig(epochs=10, batch_size=32, patience=5)
    trainer = AutoencoderTrainer(cfg)
    model   = trainer.train(X, labels)

    # Quick sanity inference
    sample = torch.from_numpy(X[:4].astype("float32"))
    errors, alerts = model.predict(sample)
    print(f"\n  Sample reconstruction errors : {errors.detach().numpy().round(5)}")
    print(f"  Pre-alerts                   : {alerts.detach().numpy()}")
    print(f"  Threshold                    : {model.anomaly_threshold:.5f}")
    print("\n  Quick-demo PASSED ✓\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Synera autoencoder")
    parser.add_argument("--quick-demo", action="store_true", help="10-epoch sanity check")
    parser.add_argument("--epochs",     type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.quick_demo:
        _quick_demo()
    else:
        from scripts.data_gen.scenario_generator import prepare_training_data
        X, labels  = prepare_training_data(n_patients=50, minutes_per_patient=60)
        cfg        = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
        trainer    = AutoencoderTrainer(cfg)
        model      = trainer.train(X, labels)
        trainer.plot_training_curves(save_path="ml/tinyml/checkpoints/training_curves.png")
