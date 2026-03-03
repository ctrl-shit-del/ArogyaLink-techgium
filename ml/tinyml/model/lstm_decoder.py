"""
Synera 2.0 — LSTM Decoder
Reconstructs a (batch, 10, 5) vital-sign sequence from a (batch, 8) latent vector.

Architecture:
  Linear(8→16)    : Expands latent back to hidden dimension
  LSTM(16→16)     : Processes the tiled latent over seq_len timesteps
  Linear(16→5)    : Projects each hidden state back to feature space

The decoder tiles the latent vector seq_len times as input to the LSTM.
This is the standard sequence-to-sequence autoencoder pattern.

Parameter count:
  Linear(8→16):   8×16 + 16              = 144
  LSTM(16→16):    4 × (16×(16+16) + 16) = 4 × 528 = 2112
  Linear(16→5):   16×5 + 5              = 85
  Total decoder:  2341 parameters
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):

    def __init__(
        self,
        latent_size: int = 8,
        hidden_size: int = 16,
        output_size: int = 5,
        seq_len:     int = 10,
        num_layers:  int = 1,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len     = seq_len
        self.num_layers  = num_layers

        # Expand latent back to hidden size
        self.fc_expand = nn.Linear(latent_size, hidden_size)

        # Recurrent decoder — input is the expanded latent tiled seq_len times
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection: hidden → feature space
        self.fc_out = nn.Linear(hidden_size, output_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc_expand.weight)
        nn.init.zeros_(self.fc_expand.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch, latent_size=8) — compressed representation

        Returns:
            reconstructed: (batch, seq_len=10, output_size=5) — in [0,1] via sigmoid
        """
        batch = latent.shape[0]

        # Expand latent to hidden dim
        expanded = self.fc_expand(latent)          # (batch, 16)

        # Tile across time dimension so LSTM has seq_len inputs
        # Each timestep receives the same latent-derived vector; the LSTM
        # learns the temporal reconstruction pattern from recurrent state.
        decoder_input = expanded.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, 10, 16)

        lstm_out, _ = self.lstm(decoder_input)     # (batch, 10, 16)

        reconstructed = self.fc_out(lstm_out)      # (batch, 10,  5)

        # Sigmoid keeps reconstruction in [0,1] to match normalised input
        return torch.sigmoid(reconstructed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
