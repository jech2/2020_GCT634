import numpy as np
import os
import librosa
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader

# Mel-spectrogram setup.
SR = 16000
FFT_HOP = 512
FFT_SIZE = 1024
NUM_MELS = 96

genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
genre_dict = {g: i for i, g in enumerate(genres)}

class SpecDataset(Dataset):
  def __init__(self, paths,  mean=0, std=1, time_dim_size=None):
    self.paths = paths
    self.mean = mean
    self.std = std
    self.time_dim_size = time_dim_size

  def __getitem__(self, i):
    # Get i-th path.
    path = self.paths[i]
    # Get i-th spectrogram path.
    path = 'gtzan/spec/' + path.replace('.wav', '.npy')

    # Extract the genre from its path.
    genre = path.split('/')[-2]
    # Trun the genre into index number.
    label = genre_dict[genre]

    # Load the mel-spectrogram.
    spec = np.load(path)
    if self.time_dim_size is not None:
      # Slice the temporal dimension with a fixed length so that they have
      # the same temporal dimensionality in mini-batches.
      spec = spec[:, :self.time_dim_size]
    # Perform standard normalization using pre-computed mean and std.
    spec = (spec - self.mean) / self.std

    return spec, label
  
  def __len__(self):
    return len(self.paths)


def load_split(path):
    with open(path) as f:
        paths = [line.rstrip('\n') for line in f]
    return paths

def extract_melspec(train_path, test_path):
    print('extracting melspec...')
    # Make directories to save mel-spectrograms.
    for genre in genres:
        os.makedirs('gtzan/spec/' + genre, exist_ok=True)
    
    for path_in in tqdm(train_path + test_path):
        # The spectrograms will be saved under `gtzan/spec/` with an file extension of `.npy`
        path_out = 'gtzan/spec/' + path_in.replace('.wav', '.npy')

        # Skip if the spectrogram already exists
        if os.path.isfile(path_out):
            continue
            
        # Load the audio signal with the desired sampling rate (SR).
        sig, _ = librosa.load(f'gtzan/wav/{path_in}', sr=SR, res_type='kaiser_fast')
        # Compute power mel-spectrogram.
        melspec = librosa.feature.melspectrogram(sig, sr=SR, n_fft=FFT_SIZE, hop_length=FFT_HOP, n_mels=NUM_MELS)
        # Transform the power mel-spectrogram into the log compressed mel-spectrogram.
        melspec = librosa.power_to_db(melspec)
        # "float64" uses too much memory! "float32" has enough precision for spectrograms.
        melspec = melspec.astype('float32')

        # Save the spectrogram.
        np.save(path_out, melspec)