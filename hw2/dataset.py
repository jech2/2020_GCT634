import numpy as np
import random
import os
from tqdm import tqdm
from glob import glob
import soundfile as sf
import shutil

import librosa
from musicnn.extractor import extractor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Mel-spectrogram setup.
SR = 16000
FFT_HOP = 512
FFT_SIZE = 1024
NUM_MELS = 96
MFCC_DIM = 30
SEGMENT_SIZE = 4 * SR // FFT_HOP # we will make each song into segments : 4 seconds

genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
splits = ['train', 'val', 'test']
genre_dict = {g: i for i, g in enumerate(genres)}

# Dataset when using mel spec as input
class SpecDataset(Dataset):
    def __init__(self, data_dir, split, mean=0, std=1, time_dim_size=None, model=None):
        self.data_dir = data_dir
        self.split = split
        self.mean = mean
        self.std = std
        self.time_dim_size = time_dim_size
        self.model = model
        
        self.paths = sorted(glob(os.path.join(data_dir, split, '*.npy')))

    def __getitem__(self, i):
        # Get i-th spectrogram path.
        path = self.paths[i]
        # Extract the genre from its path.
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

        # whether model is 1D, 2D or not set shape
        if self.model in ['Base2DCNN', 'Segmented2DCNN']:     
            spec = np.expand_dims(spec, axis=0)
        elif (self.model in ['vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101']) :
            spec = np.expand_dims(spec, axis=0)
            spec = np.repeat(spec, 3, axis=0)
        return spec, label
    
    def __len__(self):
        return len(self.paths)

# Dataset only for Q2
class EmbedDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.paths = sorted(glob(os.path.join(data_dir, split, '*.npy')))

    def __getitem__(self, i):
        # Get i-th path.
        path = self.paths[i]
        # Extract the genre from its path.
        genre = (path.split('/')[-1]).split('.')[0]
        # Trun the genre into index number.
        label = genre_dict[genre]

        # Load the mel-spectrogram.
        embed = np.load(path)

        return embed, label

    def __len__(self):
        return len(self.paths)

# Dataset for SpecAndEmbed model
class SpecEmbedDataset(Dataset):
    def __init__(self, spec_dir, embed_dir, split, mean=0, std=1, time_dim_size=None, model=None):
        self.spec_dir = spec_dir
        self.embed_dir = embed_dir
        self.split = split
        self.mean = mean
        self.std = std
        self.time_dim_size = time_dim_size
        self.model = model
        self.spec_paths = sorted(glob(os.path.join(spec_dir, split, '*.npy')))
        
        self.embed_paths = sorted(glob(os.path.join(embed_dir, split, '*.npy')))

    def __getitem__(self, i):
        # Get i-th path.
        spec_path = self.spec_paths[i]
        embed_path = self.embed_paths[i]

        # Extract the genre from its path.
        genre = (spec_path.split('/')[-1]).split('.')[0]
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

        # whether model is 1D, 2D or not set shape
        if self.model == 'Base2DCNN':     
            spec = np.expand_dims(spec, axis=0)
        elif (self.model in ['vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101']) :
            spec = np.expand_dims(spec, axis=0)
            spec = np.repeat(spec, 3, axis=0)
        spec = spec.astype('float32')

        # Load the mel-spectrogram.
        embed = np.load(embed_path)
        return [spec, embed], label
        
    def __len__(self):
        return len(self.spec_paths)

def load_split(path):
    with open(path) as f:
        paths = [line.rstrip('\n') for line in f]
    return paths

# Extract mel-spectograms
def extract_melspec(in_base, out_base, doSeg):
    print('extracting melspec')
    # Make directories to save mel-spectrograms.
    for split in splits:
        out_dir = out_base + split
        os.makedirs(out_dir, exist_ok=True)
    
        for path_in in tqdm(glob(os.path.join(in_base,split,'*.wav'))):
            # The spectrograms will be saved under `gtzan/spec/` with an file extension of `.npy`
            filename = path_in.split('/')[-1]
            path_out = os.path.join(out_dir, filename.replace('.wav','.npy'))
            
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
            
            # Do segmentation
            if doSeg:
                current_seg = 0
                while current_seg + SEGMENT_SIZE <= melspec.shape[1]:
                    idx = current_seg // SEGMENT_SIZE
                    path_out_seg = path_out.replace('.npy', f'_{idx}.npy')
                    # Save the spectrogram.
                    np.save(path_out_seg, melspec[:,current_seg:current_seg + SEGMENT_SIZE])
                    current_seg += SEGMENT_SIZE
            else:
                # Save the spectrogram.
                np.save(path_out, melspec)

# Save augmented files
# After augmentation, aug_wav folder created and total data x 2 
def save_augmentation(in_base, out_base):
    print('do augmentation')
    # Make directories
    for split in splits:
        out_dir = out_base + split
        os.makedirs(out_dir, exist_ok=True)
    
        for path_in in tqdm(glob(os.path.join(in_base,split,'*.wav'))):
            filename = path_in.split('/')[-1]
            path_out = os.path.join(out_dir, filename)
            path_out_aug = os.path.join(out_dir, filename.replace('.wav', '_a.wav'))
            
            # Skip if the augmented file already exists
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

# Do augmentation
def augmentation(y):
    # White noise
    wn = np.random.randn(len(y))
    y = y + 0.0025*wn

    # Time shift
    shft = np.random.randint(8000) # 0 ~ 8000 sample shift(0.5sec)
    y = np.roll(y, shft)

    # Time stretch
    stretch = float(np.random.randint(8, 12))
    stretch = stretch / 10.0 # stretch ratio 0.8 ~ 1.2
    y = librosa.effects.time_stretch(y, stretch)

    y = y.astype('float32')
    
    return y

# (Beta version) Extract other features rather than using only mel spec
# Here, MFCC, dMFCC, ddMFCC and spectral centroid is used
def extract_features(in_base, out_base, doSeg):
    print('extracting features...')
    # Make directories to save audio features
    for split in splits:
        out_dir = out_base + split
        os.makedirs(out_dir, exist_ok=True)
    
        for path_in in tqdm(glob(os.path.join(in_base,split,'*.wav'))):
            filename = path_in.split('/')[-1]
            path_out = os.path.join(out_dir, filename.replace('.wav','.npy'))
            
            # Skip if the file already exists
            if os.path.isfile(path_out):
                continue
            
            # Load the audio signal with the desired sampling rate (SR).
            y, _ = librosa.load(path_in, sr=SR, res_type='kaiser_fast')
            # STFT
            D = np.abs(librosa.core.stft(y, hop_length=FFT_HOP, n_fft=FFT_SIZE, win_length=FFT_SIZE))
            # mfcc
            mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=MFCC_DIM)
            # delta mfcc and double delta
            delta_mfcc = librosa.feature.delta(mfcc)
            ddelta_mfcc = librosa.feature.delta(mfcc, order=2)
            # spectral centroid
            spec_centroid = librosa.feature.spectral_centroid(S=D)
            
            # concatenate all features
            features = np.concatenate([mfcc, delta_mfcc, ddelta_mfcc, spec_centroid], axis=0)
            
            if doSeg:
                current_seg = 0
                while current_seg + SEGMENT_SIZE <= features.shape[1]:
                    idx = current_seg // SEGMENT_SIZE
                    path_out_seg = path_out.replace('.npy', f'_{idx}.npy')
                    # Save the spectrogram.
                    np.save(path_out_seg, features[:,current_seg:current_seg + SEGMENT_SIZE])
                    current_seg += SEGMENT_SIZE
            else:
                # Save the spectrogram.
                np.save(path_out, features)

# Extract embeddings from musicnn
def extract_embeddings(in_base, out_base, model='MSD_musicnn'):
    # Make directories to save embeddings.
    for split in splits:
        out_dir = out_base + split
        os.makedirs(out_dir, exist_ok=True)
    
        for path_in in tqdm(glob(os.path.join(in_base,split,'*.wav'))):
            filename = path_in.split('/')[-1]
            path_out = os.path.join(out_dir, filename.replace('.wav','.npy'))
            
            # Skip if the embeddings file already exists
            if os.path.isfile(path_out):
                continue

            # Extract the embedding using the pre-trained model.
            _, _, embeds = extractor(path_in, model=model, extract_features=True)
            # Average the embeddings over temporal dimension.
            embed = embeds['max_pool'].mean(axis=0)

            # Save the embedding.
            np.save(path_out, embed)

# Based on the seg_spec file name, copy embed file from gtzan/embed/ to gtzan/seg_embed
# copy_embeddings_for_segment('gtzan/seg_spec/', 'gtzan/embed/', 'gtzan/seg_embed/')
def copy_embeddings_for_segment(pair_base, in_base, out_base):
    print('copy embeddings for segment...')
    # Make directories to save embeddings
    for split in splits:
        out_dir = out_base + split
        os.makedirs(out_dir, exist_ok=True)
    
        for path_in in tqdm(glob(os.path.join(in_base,split,'*.npy'))):
            filename = path_in.split('/')[-1]
            pair_file = os.path.join(pair_base, split, filename)
            path_out = os.path.join(out_dir, filename)
            
            # Skip if embeddings file already exists
            if os.path.isfile(path_out):
                continue
            # Load the embedding file and copy it
            embed = np.load(path_in)
            current_seg = 0
            for idx in range(10):
                pair_seg = pair_file.replace('.npy', f'_{idx}.npy')
                path_out_seg = path_out.replace('.npy', f'_{idx}.npy')
                if os.path.isfile(pair_seg):
                    np.save(path_out_seg, embed)
                else:
                    break


if __name__ == "__main__":
    # Save augmented wav file
    # save_augmentation(in_base='gtzan/wav/', out_base='gtzan/aug_wav/')
    
    # Extract GTT and MSD embeddings from original and augmented data
    # extract_embeddings(in_base='gtzan/wav/', out_base='gtzan/embed/', model='MTT_musicnn')
    # extract_embeddings(in_base='gtzan/wav/', out_base='gtzan/msd_embed/', model='MSD_musicnn')
    # extract_embeddings(in_base='gtzan/aug_wav/', out_base='gtzan/aug_embed/', model='MTT_musicnn')
    # extract_embeddings(in_base='gtzan/aug_wav/', out_base='gtzan/aug_msd_embed/', model='MSD_musicnn')
    
    # Extract mel spec
    extract_melspec(in_base='gtzan/wav/', out_base='gtzan/spec/', doSeg=False)   
    # extract_melspec(in_base='gtzan/wav/', out_base='gtzan/seg_spec/', doSeg=True)   
    # extract_melspec(in_base='gtzan/aug_wav/', out_base='gtzan/aug_spec/', doSeg=False)   
    # extract_melspec(in_base='gtzan/aug_wav/', out_base='gtzan/seg_aug_spec/', doSeg=True)   
    
    # extract hand made features
    # extract_features(in_base='gtzan/wav/', out_base='gtzan/features/', doSeg=False)
    # extract_features(in_base='gtzan/wav/', out_base='gtzan/seg_features/', doSeg=True)
    # extract_features(in_base='gtzan/aug_wav/', out_base='gtzan/aug_features/', doSeg=False)
    # extract_features(in_base='gtzan/aug_wav/', out_base='gtzan/seg_aug_features/', doSeg=True)
    
    # Copy embeddings for segment condition
    # copy_embeddings_for_segment(pair_base='gtzan/seg_spec/', in_base='gtzan/embed/', out_base='gtzan/seg_embed/')
    # copy_embeddings_for_segment(pair_base='gtzan/seg_spec/', in_base='gtzan/msd_embed/', out_base='gtzan/seg_msd_embed/')
    # copy_embeddings_for_segment(pair_base='gtzan/seg_aug_spec/',in_base='gtzan/aug_embed/', out_base='gtzan/seg_aug_embed/')
    # copy_embeddings_for_segment(pair_base='gtzan/seg_aug_spec/',in_base='gtzan/aug_msd_embed/', out_base='gtzan/seg_aug_msd_embed/')