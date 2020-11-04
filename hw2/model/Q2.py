import numpy as np
import os
import torch
import torch.nn as nn

# TODO: Question 2
class Q2(nn.Module):
  def __init__(self, embed_size=753, genres=8, hidden_size=32):
    super(Q2, self).__init__()
    
    self.hidden_size = hidden_size
    
    self.main = nn.Sequential(
        nn.Linear(embed_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, len(genres))
    )
    
  def forward(self, x):
    x = self.main(x)
    return x