import numpy as np
import random
import os
import librosa
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
import soundfile as sf
import shutil
from torch.utils.data import Dataset, DataLoader

# Mel-spectrogram setup.
SR = 16000
FFT_HOP = 512
FFT_SIZE = 1024
NUM_MELS = 96

genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
splits = ['train', 'val', 'test']
genre_dict = {g: i for i, g in enumerate(genres)}

class SpecDataset(Dataset):
    def __init__(self, paths, mean=0, std=1, time_dim_size=None, model=None):
        self.paths = paths
        self.mean = mean
        self.std = std
        self.time_dim_size = time_dim_size
        self.model = model

    def __getitem__(self, i):
        # Get i-th path.
        path = self.paths[i]
        # Get i-th spectrogram path.
        #path = 'gtzan/seg_spec/' + path.replace('.wav', '.npy')

        # Extract the genre from its path.
        #genre = path.split('/')[-2]
        genre = (path.split('/')[-1]).split('.')[0]
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

        if self.model == 'Base2DCNN':     
            spec = np.expand_dims(spec, axis=0)
        elif self.model == 'vgg':
            spec = np.expand_dims(spec, axis=0)
            spec = np.repeat(spec,3,axis=0)
        return spec, label
    
    def __len__(self):
        return len(self.paths)

class EmbedDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __getitem__(self, i):
        # Get i-th path.
        path = self.paths[i]
        # Get i-th embeddding path.
        path = 'gtzan/embed/' + path.replace('.wav', '.npy')

        # Extract the genre from its path.
        genre = path.split('/')[-2]
        # Trun the genre into index number.
        label = genre_dict[genre]

        # Load the mel-spectrogram.
        embed = np.load(path)

        return embed, label
    
    def __len__(self):
        return len(self.paths)

class SpecEmbedDataset(Dataset):
    def __init__(self, paths,  mean=0, std=1, time_dim_size=None):
        self.paths = paths
        self.mean = mean
        self.std = std
        self.time_dim_size = time_dim_size

    def __getitem__(self, i):
        # Get i-th path.
        path = self.paths[i]
        # Get i-th spectrogram path.
        spec_path = 'gtzan/spec/' + path.replace('.wav', '.npy')
        embed_path = 'gtzan/msd_embed/' + path.replace('.wav', '.npy')

        # Extract the genre from its path.
        genre = path.split('/')[-2]
        # Trun the genre into index number.
        label = genre_dict[genre]

        # Load the mel-spectrogram.
        spec = np.load(spec_path)
        if self.time_dim_size is not None:
            # Slice the temporal dimension with a fixed length so that they have
            # the same temporal dimensionality in mini-batches.
            spec = spec[:, :self.time_dim_size]
        # Perform standard normalization using pre-computed mean and std.
        spec = (spec - self.mean) / self.std

        # Load the mel-spectrogram.
        embed = np.load(embed_path)

        return [spec, embed], label
    
    def __len__(self):
        return len(self.paths)

def load_split(path):
    with open(path) as f:
        paths = [line.rstrip('\n') for line in f]
    return paths

def extract_melspec(path, doSeg):
    print('extracting melspec segment...')

    segment_size = 4 * SR // FFT_HOP # we will make each song into segments
    print(segment_size)
    if doSeg:
        out_base = 'gtzan/seg_spec/'
    else:
        out_base = 'gtzan/ori_spec/'
    # Make directories to save mel-spectrograms.
    for split in splits:
        out_dir = out_base + split
        os.makedirs(out_dir, exist_ok=True)
    
        for path_in in tqdm(glob(os.path.join(path,split,'*.wav'))):
            # The spectrograms will be saved under `gtzan/spec/` with an file extension of `.npy`
            filename = path_in.split('/')[-1]
            path_out = os.path.join(out_dir, filename)
            
            # Skip if the spectrogram already exists
            if os.path.isfile(path_out):
                continue
            
            # Load the audio signal with the desired sampling rate (SR).
            sig, _ = librosa.load(path_in, sr=SR, res_type='kaiser_fast')

            # Compute power mel-spectrogram.
            melspec = librosa.feature.melspectrogram(sig, sr=SR, n_fft=FFT_SIZE, hop_length=FFT_HOP, n_mels=NUM_MELS)
            # Transform the power mel-spectrogram into the log compressed mel-spectrogram.
            melspec = librosa.power_to_db(melspec)
            # "float64" uses too much memory! "float32" has enough precision for spectrograms.
            melspec = melspec.astype('float32')
            
            if doSeg:
                current_seg = 0
                while current_seg + segment_size <= melspec.shape[1]:
                    idx = current_seg // segment_size
                    path_out_seg = path_out.replace('.wav', f'_{idx}.npy')
                    # Save the spectrogram.
                    np.save(path_out_seg, melspec[:,current_seg:current_seg + segment_size])
                    current_seg += segment_size
            else:
                # Save the spectrogram.
                np.save(path_out, melspec)

# after augmentation, aug_wav folder created and total data x 2 
def save_augmentation(path):
    print('do augmentation')
    # Make directories
    for split in splits:
        out_dir = 'gtzan/aug_wav/' + split
        os.makedirs(out_dir, exist_ok=True)
    
        for path_in in tqdm(glob(os.path.join(path,split,'*.wav'))):
            filename = path_in.split('/')[-1]
            path_out = os.path.join(out_dir, filename)
            path_out_aug = os.path.join(out_dir, filename.replace('.wav', '_a.wav'))
            
            # Skip if the spectrogram already exists
            if os.path.isfile(path_out) and os.path.isfile(path_out_aug):
                continue
            
            # Save original file
            shutil.copy(path_in, out_dir)

            if split is 'test':
                continue

            # Load the audio signal with the desired sampling rate (SR).
            sig, _ = librosa.load(path_in, sr=SR, res_type='kaiser_fast')

            sig = augmentation(sig)
            
            # Save augmented file
            sf.write(path_out_aug, sig, samplerate=SR)

# do augmentation
def augmentation(y):
    # white noise
    wn = np.random.randn(len(y))
    y = y + 0.0025*wn

    # time shift
    shft = np.random.randint(6000) # 0 ~ 8000 sample shift(0.5sec)
    y = np.roll(y, shft)

    # time stretch
    stretch = float(np.random.randint(8, 12))
    stretch = stretch / 10.0 # stretch ratio 0.8 ~ 1.2
    y = librosa.effects.time_stretch(y, stretch)

    y = y.astype('float32')
    
    return y