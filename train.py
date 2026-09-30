"""
train.py
=============================================================================
Memory-Efficient Synthetic/Semi-Synthetic Sepsis Model Training Pipeline.
=============================================================================
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import WaveformDataLoader
from preprocess import PhysiologicalPreprocessor
from model import SepsisDetectionNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Train) %(message)s"
)


def generate_training_data(num_samples_per_class: int = 100, window_len: int = 1250):
    """
    Synthesizes healthy vs septic 10-second window pairs.
    """
    logging.info(f"Generating synthetic training batch ({num_samples_per_class * 2} windows)...")
    prep = PhysiologicalPreprocessor()
    t = np.linspace(0, 10, window_len, dtype=np.float32)

    windows, labels = [], []

    for _ in range(num_samples_per_class):
        # Healthy variation (HR 60-80 bpm)
        hr_h = np.random.uniform(1.0, 1.3)
        ecg_h = np.sin(2 * np.pi * hr_h * t) + 0.5 * np.sin(4 * np.pi * hr_h * t)
        ppg_h = np.sin(2 * np.pi * hr_h * t - 0.2) + 0.3 * np.sin(4 * np.pi * hr_h * t)
        ecg_h = prep.process_ecg(ecg_h, 125)
        ppg_h = prep.process_ppg(ppg_h, 125)

        windows.append(np.stack([ecg_h, ppg_h], axis=0))
        labels.append(0.0)

        # Sepsis variation (HR 100-140 bpm, low perfusion, noise)
        ecg_s, ppg_s = prep.simulate_sepsis_morphology(ecg_h, ppg_h, 125)
        windows.append(np.stack([ecg_s, ppg_s], axis=0))
        labels.append(1.0)

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)
    return torch.tensor(X), torch.tensor(y)


def train_model(epochs: int = 12, batch_size: int = 16, save_path: str = "artifacts/sepsis_detector.pt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logging.info("Starting Sepsis Deep Learning Model Training...")

    X, y = generate_training_data(num_samples_per_class=120)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on device: {device}")

    model = SepsisDetectionNet(in_channels=2, seq_len=1250).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            predicted_classes = (preds >= 0.5).float()
            correct += (predicted_classes == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        logging.info(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved successfully to '{save_path}'.")
    return model


if __name__ == "__main__":
    train_model()
