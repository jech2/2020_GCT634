import numpy as np
import os
import torch
import torch.nn as nn
from model.Q1 import Q1
from model.Q2 import Q2

class SpecAndEmbed(nn.Module):
  def __init__(self, num_mels, genres):
    super(SpecAndEmbed, self).__init__()

    self.spec_conv = Q1(num_mels=num_mels, genres=genres)
    self.embed_mlp = Q2(genres=genres)
    self.BatchNorm = nn.BatchNorm1d(len(genres))
    self.ReLU = nn.ReLU()

    self.linear = nn.Linear(16, len(genres))

  def forward(self, spec, embed):
    spec = self.spec_conv(spec)
    #spec = self.BatchNorm(spec)
    spec = self.ReLU(spec)
    embed = self.embed_mlp(embed)
    #embed = self.BatchNorm(embed)
    embed = self.ReLU(embed)
    out = torch.cat([spec, embed], dim=1)
    out = self.linear(out)
    return out