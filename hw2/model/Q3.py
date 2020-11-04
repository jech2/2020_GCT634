import numpy as np
import os
import torch
import torch.nn as nn

class Baseline(nn.Module):
  def __init__(self, num_mels, genres):
    super(Baseline, self).__init__()

    self.conv0 = nn.Sequential(
      nn.Conv1d(num_mels, out_channels=32, kernel_size=7, stride=1, padding=3),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=7, stride=7)
    )

    self.conv1 = nn.Sequential(
      nn.Conv1d(32, out_channels=32, kernel_size=7, stride=1, padding=3),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=7, stride=7)
    )

    self.conv2 = nn.Sequential(
      nn.Conv1d(32, out_channels=32, kernel_size=7, stride=1, padding=3),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=7, stride=7)
    )

    # Aggregate features over temporal dimension.
    self.final_pool = nn.AdaptiveAvgPool1d(1)

    # Predict genres using the aggregated features.
    self.linear = nn.Linear(32, len(genres))

  def forward(self, x):
    x = self.conv0(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.final_pool(x)
    x = self.linear(x.squeeze(-1))
    return x