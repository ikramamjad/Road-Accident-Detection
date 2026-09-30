"""
Training Script for Causal GRU and Multi-Channel Fusion Layer.
Trains strictly causal sequential models on synthetic/CCD trajectory features.
"""

import argparse
import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from road_accident_detection.temporal.causal_gru import CausalGRUModel
from road_accident_detection.temporal.dataset import SequenceDataset, SyntheticSequenceGenerator
from road_accident_detection.fusion.multi_channel_fusion import LearnedFusionMLP


_ROOT = Path(__file__).resolve().parent.parent


def train_causal_gru(
    output_dir: Optional[str] = None,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    seq_len: int = 16,
):
    if output_dir is None or "d:/" in str(output_dir).lower() or "d:\\" in str(output_dir).lower():
        output_dir = str(_ROOT / "data" / "weights")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using device: {device}")

    # Generate synthetic training & validation sequences
    print("[Dataset] Generating training trajectories (Normal, Collision, Near-Miss)...")
    X_norm, y_norm = SyntheticSequenceGenerator.generate_normal_driving(seq_len=seq_len, num_samples=300)
    X_crash, y_crash = SyntheticSequenceGenerator.generate_collision(seq_len=seq_len, num_samples=300)
    X_near, y_near = SyntheticSequenceGenerator.generate_near_miss(seq_len=seq_len, num_samples=150)

    X = np.concatenate([X_norm, X_crash, X_near], axis=0)
    y = np.concatenate([y_norm, y_crash, y_near], axis=0)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    split = int(0.8 * len(X))
    train_dataset = SequenceDataset(X[:split], y[:split])
    val_dataset = SequenceDataset(X[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CausalGRUModel(input_dim=12, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[Causal GRU] Starting training for {epochs} epochs...")
    best_val_loss = float("inf")
    save_path = Path(output_dir) / "causal_gru_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds, _ = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)
                pred_binary = (preds >= 0.5).float()
                correct += (pred_binary == batch_y).sum().item()
                total += len(batch_y)

        val_loss /= len(val_dataset)
        val_acc = correct / max(total, 1)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(save_path))

    print(f"[Causal GRU] Training complete! Best weights saved to: {save_path}")


def train_fusion_mlp(
    output_dir: Optional[str] = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    if output_dir is None or "d:/" in str(output_dir).lower() or "d:\\" in str(output_dir).lower():
        output_dir = str(_ROOT / "data" / "weights")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Fusion MLP] Training Learned Cross-Channel Fusion MLP...")

    # Generate synthetic channel score tuples: [det, pose, kin, temp]
    N = 1000
    X = np.random.uniform(0.0, 0.4, (N, 4)).astype(np.float32)
    y = np.zeros((N, 1), dtype=np.float32)

    # Crash samples: elevated kinematic risk + pose anomaly or temporal sequence
    crash_idx = np.random.choice(N, size=400, replace=False)
    for idx in crash_idx:
        X[idx, 0] = np.random.uniform(0.7, 0.95)  # det
        X[idx, 1] = np.random.uniform(0.6, 0.98)  # pose
        X[idx, 2] = np.random.uniform(0.75, 0.99) # kin
        X[idx, 3] = np.random.uniform(0.6, 0.95)  # temp
        y[idx, 0] = 1.0

    # Near-miss hard negative: high kinematic risk, but pose is 0.0 and temporal resolves
    near_idx = np.random.choice([i for i in range(N) if i not in crash_idx], size=150, replace=False)
    for idx in near_idx:
        X[idx, 0] = np.random.uniform(0.7, 0.9)
        X[idx, 1] = 0.05
        X[idx, 2] = np.random.uniform(0.65, 0.85)
        X[idx, 3] = np.random.uniform(0.2, 0.45)
        y[idx, 0] = 0.0

    split = int(0.8 * N)
    train_dataset = SequenceDataset(X[:split], y[:split])
    val_dataset = SequenceDataset(X[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LearnedFusionMLP(in_features=4, hidden_dim=16).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.squeeze(1).to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    save_path = Path(output_dir) / "fusion_mlp_best.pt"
    torch.save(model.state_dict(), str(save_path))
    print(f"[Fusion MLP] Model weights saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Causal GRU and Fusion Network")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=str(_ROOT / "data" / "weights"))
    args = parser.parse_args()

    train_causal_gru(output_dir=args.output_dir, epochs=args.epochs)
    train_fusion_mlp(output_dir=args.output_dir, epochs=args.epochs)
