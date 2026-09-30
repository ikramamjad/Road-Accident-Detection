"""
Sequence Dataset & Synthetic Trajectory Generator.
Provides training and evaluation sequences for Causal GRU and Fusion layers,
compatible with Car Crash Dataset (CCD) format and synthetic benchmarks.
"""

from __future__ import annotations
from typing import Any, List, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    if torch is None:
        raise ImportError("torch is None")
    TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False
    torch = type("torch", (), {"Tensor": Any})()
    Dataset = object


class SequenceDataset(Dataset):
    """PyTorch Dataset for temporal feature sequences."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: np.ndarray of shape (N, seq_len, feature_dim)
            labels: np.ndarray of shape (N, 1) or (N,)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class SyntheticSequenceGenerator:
    """Generates synthetic crash, near-miss, and normal driving temporal sequences."""

    @staticmethod
    def generate_normal_driving(seq_len: int = 16, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth vehicle trajectories without anomalies."""
        samples = []
        for _ in range(num_samples):
            seq = np.zeros((seq_len, 12), dtype=np.float32)
            base_x = np.random.uniform(0.2, 0.8)
            base_y = np.random.uniform(0.3, 0.7)
            speed = np.random.uniform(20.0, 40.0) / 500.0
            heading = np.random.uniform(-0.2, 0.2)

            for t in range(seq_len):
                cx = base_x + speed * np.cos(heading) * t
                cy = base_y + speed * np.sin(heading) * t
                w, h = 0.08, 0.12
                vx = speed * np.cos(heading)
                vy = speed * np.sin(heading)
                ax = np.random.normal(0, 0.005)
                ay = np.random.normal(0, 0.005)
                yaw_rate = np.random.normal(0, 0.02)
                pose_score = 0.0
                kin_risk = 0.05
                seq[t] = [cx, cy, w, h, vx, vy, ax, ay, yaw_rate, pose_score, kin_risk, speed]
            samples.append(seq)

        X = np.array(samples, dtype=np.float32)
        y = np.zeros((num_samples, 1), dtype=np.float32)
        return X, y

    @staticmethod
    def generate_collision(seq_len: int = 16, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate collision trajectories with sudden shock, deceleration, and post-impact stop."""
        samples = []
        for _ in range(num_samples):
            seq = np.zeros((seq_len, 12), dtype=np.float32)
            impact_frame = np.random.randint(seq_len // 2, seq_len - 2)
            speed = np.random.uniform(30.0, 60.0) / 500.0

            for t in range(seq_len):
                if t < impact_frame:
                    cx = 0.4 + speed * t
                    cy = 0.5
                    vx, vy = speed, 0.0
                    ax, ay = 0.0, 0.0
                    yaw_rate = 0.0
                    pose_score = 0.0
                    kin_risk = 0.2 + 0.6 * (t / impact_frame)
                else:
                    # Post-impact: abrupt speed drop, high angular spin
                    cx = 0.4 + speed * impact_frame + np.random.normal(0, 0.01)
                    cy = 0.5 + np.random.normal(0, 0.02)
                    vx, vy = 0.02 * speed, 0.0
                    ax, ay = -0.5, 0.2
                    yaw_rate = 0.8
                    pose_score = 0.85 if np.random.rand() > 0.5 else 0.2
                    kin_risk = 0.95

                seq[t] = [cx, cy, 0.08, 0.12, vx, vy, ax, ay, yaw_rate, pose_score, kin_risk, speed]
            samples.append(seq)

        X = np.array(samples, dtype=np.float32)
        y = np.ones((num_samples, 1), dtype=np.float32)
        return X, y

    @staticmethod
    def generate_near_miss(seq_len: int = 16, num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Generate near-miss trajectories with hard braking, swerve, then recovery."""
        samples = []
        for _ in range(num_samples):
            seq = np.zeros((seq_len, 12), dtype=np.float32)
            evasion_frame = seq_len // 2

            for t in range(seq_len):
                if t < evasion_frame:
                    vx, vy = 0.06, 0.0
                    ax, ay = 0.0, 0.0
                    kin_risk = 0.3 + 0.4 * (t / evasion_frame)
                    yaw_rate = 0.0
                elif t < evasion_frame + 3:
                    # Hard evasion
                    vx, vy = 0.02, 0.04
                    ax, ay = -0.3, 0.2
                    kin_risk = 0.75  # High kinematic warning
                    yaw_rate = 0.6
                else:
                    # Recovery
                    vx, vy = 0.04, 0.01
                    ax, ay = 0.05, 0.0
                    kin_risk = 0.25
                    yaw_rate = 0.1

                seq[t] = [0.5, 0.5, 0.08, 0.12, vx, vy, ax, ay, yaw_rate, 0.0, kin_risk, 0.05]
            samples.append(seq)

        X = np.array(samples, dtype=np.float32)
        # Labeled as negative for accident trigger (0.0) but used for hard negative tuning
        y = np.zeros((num_samples, 1), dtype=np.float32)
        return X, y
