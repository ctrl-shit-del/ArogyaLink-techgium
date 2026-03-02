"""
Synera 2.0 — LSTM Encoder
Compresses a (batch, 10, 5) vital-sign window to a (batch, 8) latent vector.

Architecture rationale (every decision is hardware-justified):
  hidden_size=16 : Minimum capacity to capture physiological correlations
                   between HR, SpO2, and BP.  Going smaller loses sensitivity.
  latent_size=8  : Bottleneck forces the model to learn essential trajectory
                   patterns; larger latent = less anomaly discrimination.
  num_layers=1   : Each additional LSTM layer doubles inference RAM. One layer
                   fits within the 50KB RAM budget on the ESP32-S3.
  causal only    : No bidirectional LSTM — device inference is strictly online,
                   one reading at a time.  Can't look ahead.

Parameter count:
  LSTM(5→16):  4 × (16×(5+16) + 16) = 4 × 352 = 1408
  Linear(16→8): 16×8 + 8             =         136
  Total encoder: 1544 parameters
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):

    def __init__(
        self,
        input_size:  int = 5,
        hidden_size: int = 16,
        latent_size: int = 8,
        num_layers:  int = 1,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers  = num_layers

        # Single-layer LSTM — processes (batch, seq=10, features=5) sequentially
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,     # input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Latent projection: the final hidden state is compressed to 8 dims
        self.fc_latent = nn.Linear(hidden_size, latent_size)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for the linear layer; default init for LSTM is already good."""
        nn.init.xavier_uniform_(self.fc_latent.weight)
        nn.init.zeros_(self.fc_latent.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len=10, features=5) — normalised vital-sign window

        Returns:
            latent:      (batch, latent_size=8)
            lstm_state:  (h_n, c_n) — can be passed to the decoder
        """
        # lstm_out: (batch, seq, hidden=16)
        # h_n:      (num_layers=1, batch, hidden=16)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the hidden state from the LAST timestep — it has seen the full sequence
        # h_n[-1]: (batch, hidden=16)
        last_hidden = h_n[-1]

        latent = self.fc_latent(last_hidden)   # (batch, 8)
        return latent, (h_n, c_n)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
