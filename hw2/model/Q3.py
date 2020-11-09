import numpy as np
import os
import torch
import torch.nn as nn
# from model.Q1 import Q1
# from model.Q2 import Q2
import torchvision.models as models

# input : mel spec + embedding features
# Model that use spectral features and embedding features
class SpecAndEmbed(nn.Module):
  def __init__(self, num_mels, genres, model=None):
    super(SpecAndEmbed, self).__init__()
    self.model = model
    # Model settings of spec_cov
    if self.model == "Q1":
      self.spec_conv = Q1(num_mels=num_mels, genres=genres)
    elif self.model == "Base2DCNN":
      self.spec_conv = Base2DCNN(genres)
    elif self.model == "resnet34":
      self.spec_conv = models.resnet34(pretrained=True)
      fc_in_features = self.spec_conv.fc.in_features
      self.spec_conv.fc = nn.Linear(in_features=fc_in_features, out_features=len(genres))
    # Model settings of embed_mlp
    self.embed_mlp = Q2(genres=genres)
    self.BatchNorm = nn.BatchNorm1d(len(genres))
    self.ReLU = nn.ReLU()

    self.linear = nn.Linear(16, len(genres))

  def forward(self, spec, embed):
    spec = self.spec_conv(spec)
    spec = self.BatchNorm(spec)
    spec = self.ReLU(spec)
    embed = self.embed_mlp(embed)
    #embed = self.BatchNorm(embed) # not use this
    embed = self.ReLU(embed)
    out = torch.cat([spec, embed], dim=1)
    out = self.linear(out)
    return out

# input : mel spec
class Base2DCNN(nn.Module):
  def __init__(self, genres):
    super(Base2DCNN, self).__init__()
    self.conv0 = nn.Sequential(
      nn.Conv2d(1, out_channels=16, kernel_size=7, stride=1, padding=3),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5, stride=5)
    )

    self.conv1 = nn.Sequential(
      nn.Conv2d(16, out_channels=32, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=3)
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(32, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=3)
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
    # Aggregate features over temporal dimension.
    self.final_pool = nn.AdaptiveAvgPool2d(1)

    # Predict genres using the aggregated features.
    self.linear = nn.Linear(128, len(genres))
   

  def forward(self, x):
    x = self.conv0(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.final_pool(x)
    x = self.linear(torch.squeeze(x))
    return x

if __name__ == "__main__":
    from torchsummary import summary
    genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    net = Base2DCNN(genres).to(device)
    summary(net, (1, 96, 938))
    #summary(net, (1, 96, 125))