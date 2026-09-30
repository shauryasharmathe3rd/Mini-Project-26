"""
model.py
=============================================================================
Deep Learning Neural Architecture: 1D-CNN + Bidirectional LSTM.
Designed to capture both local QRS/PPG pulse morphology (CNN) and
long-term autonomic rhythm heart-rate variability (LSTM).
=============================================================================
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Model) %(message)s"
)


class SepsisDetectionNet(nn.Module):
    def __init__(self, in_channels: int = 2, seq_len: int = 1250):
        super(SepsisDetectionNet, self).__init__()
        logging.info("Constructing SepsisDetectionNet (1D-CNN Feature Extractor + Bi-LSTM)...")

        # 3-Layer 1D Convolutional Neural Network
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )

        # 2-Layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass:
        x shape: [Batch_Size, Channels (2), Window_Length (1250)]
        """
        feats = self.conv_block(x)
        feats = feats.permute(0, 2, 1)  # [B, 64, 128]
        lstm_out, (h_n, _) = self.lstm(feats)

        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat((h_forward, h_backward), dim=1)

        prob = self.classifier(h_combined)
        return prob
