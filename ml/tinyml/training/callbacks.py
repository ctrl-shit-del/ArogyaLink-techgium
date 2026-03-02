"""
Synera 2.0 — Training Callbacks
EarlyStopping and ModelCheckpoint utilities for the autoencoder trainer.
"""
from __future__ import annotations

import shutil
from pathlib import Path


class EarlyStopping:
    """
    Stops training when validation loss has not improved for `patience` epochs.

    Usage:
        stopper = EarlyStopping(patience=15, min_delta=1e-5)
        for epoch in range(max_epochs):
            val_loss = trainer.validate()
            if stopper(val_loss):
                print(f"Early stop at epoch {epoch}")
                break
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self):
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    @property
    def remaining(self) -> int:
        return self.patience - self.counter


class ModelCheckpoint:
    """
    Saves the best model checkpoint to disk whenever validation loss improves.

    Usage:
        ckpt = ModelCheckpoint(save_dir="ml/tinyml/checkpoints")
        ckpt.update(epoch, val_loss, model)
    """

    def __init__(self, save_dir: str | Path, filename: str = "best_autoencoder.pt"):
        self.save_dir  = Path(save_dir)
        self.filename  = filename
        self.best_loss = float("inf")
        self.best_epoch: int | None = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        return self.save_dir / self.filename

    def update(self, epoch: int, val_loss: float, model) -> bool:
        """
        Save model if val_loss improved.

        Returns:
            True if a new best checkpoint was saved, else False.
        """
        import torch
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            # Save full model state
            torch.save({
                "epoch":      epoch,
                "val_loss":   val_loss,
                "state_dict": model.state_dict(),
            }, self.checkpoint_path)
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"ModelCheckpoint(path={self.checkpoint_path}, "
            f"best_epoch={self.best_epoch}, best_loss={self.best_loss:.6f})"
        )
