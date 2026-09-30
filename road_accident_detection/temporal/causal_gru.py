"""
Strictly Causal Gated Recurrent Unit (Causal GRU) for Accident Anomaly Detection.
Processes temporal sliding windows without future-frame leakage.
"""

from __future__ import annotations
from typing import Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    if torch is None:
        raise ImportError("torch is None")
    TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False
    torch = type("torch", (), {"Tensor": Any, "device": Any})()

    class _DummyModule:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def eval(self):
            return self
        def to(self, *args, **kwargs):
            return self

    class _DummyNN:
        Module = _DummyModule
        def Sequential(self, *args, **kwargs): return _DummyModule()
        def Linear(self, *args, **kwargs): return _DummyModule()
        def LayerNorm(self, *args, **kwargs): return _DummyModule()
        def ReLU(self, *args, **kwargs): return _DummyModule()
        def GRU(self, *args, **kwargs): return _DummyModule()
        def Dropout(self, *args, **kwargs): return _DummyModule()
        def Sigmoid(self, *args, **kwargs): return _DummyModule()

    nn = _DummyNN()


class CausalGRUModel(nn.Module):
    """
    Causal GRU neural network for sequential risk detection.
    Processes feature sequences forward in time.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input normalization / projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Strictly causal forward GRU (bidirectional=False ensures zero future leakage)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output anomaly classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over sequence.
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional initial hidden state (num_layers, batch_size, hidden_dim)
        Returns:
            anomaly_score: Tensor of shape (batch_size, 1) representing latest frame anomaly score
            hidden: Updated hidden state of shape (num_layers, batch_size, hidden_dim)
        """
        proj = self.input_layer(x)
        out, hidden = self.gru(proj, hidden)
        # Take the last time step output to predict current instantaneous sustained risk
        last_step = out[:, -1, :]
        anomaly_score = self.classifier(last_step)
        return anomaly_score, hidden


class TemporalInferenceEngine:
    """
    Inference interface wrapping CausalGRUModel.
    Supports PyTorch evaluation mode and step-by-step rolling prediction.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        device: str = "cpu",
    ):
        self.device = torch.device(device) if TORCH_AVAILABLE else "cpu"
        self.model = None
        if TORCH_AVAILABLE:
            self.model = CausalGRUModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            ).to(self.device)

            if model_path:
                self.load_weights(model_path)

            self.model.eval()

    def load_weights(self, path: str) -> None:
        """Load trained PyTorch state dict."""
        if not TORCH_AVAILABLE or self.model is None:
            return
        try:
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)
        except Exception:
            pass

    def evaluate_sequence(self, feature_matrix: np.ndarray) -> float:
        """
        Evaluate temporal anomaly score on a single object's feature matrix.
        Args:
            feature_matrix: np.ndarray of shape (seq_len, input_dim)
        Returns:
            anomaly_score: float in range [0.0, 1.0]
        """
        if feature_matrix is None or feature_matrix.shape[0] == 0:
            return 0.0

        if not TORCH_AVAILABLE or self.model is None:
            # Deterministic heuristic fallback when PyTorch is not available
            if feature_matrix.shape[1] > 2:
                recent_risk = float(np.mean(np.abs(feature_matrix[-3:, 2])))
                return float(np.clip(recent_risk / 10.0, 0.0, 1.0))
            return 0.0

        with torch.no_grad():
            tensor_x = torch.tensor(feature_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
            score_tensor, _ = self.model(tensor_x)
            return float(score_tensor.squeeze().item())
