# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import sys
import os
import numpy as np
import librosa
from sklearn.cluster import KMeans

data_path = './dataset/'
spec_path = './spec/'
codebook_path = './codebook/'

MFCC_DIM = 30

def extract_spec(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        # print file_path
        y0, sr = librosa.load(file_path, sr=22050)
        # we use first 1 second
        half = len(y0)/4
        y = y0[:round(half)]
        # mfcc
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)
        # delta mfcc and double delta
        delta_mfcc = librosa.feature.delta(mfcc)
        ddelta_mfcc = librosa.feature.delta(mfcc, order=2)

        # STFT
        D = np.abs(librosa.core.stft(y, hop_length=512, n_fft=1024, win_length=1024))
        D_dB = librosa.amplitude_to_db(D, ref=np.max)

        # mel spectrogram
        mel_S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(mel_S, ref=np.max) #log compression
            
        # spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(S=D)
        
        # concatenate all features
        features = np.concatenate([mfcc, delta_mfcc, ddelta_mfcc, spec_centroid], axis=0)
        
        # save mfcc as a file
        file_name = file_name.replace('.wav','.npy')
        save_file = spec_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, features)

    f.close();

def extract_codebook(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')
    i = 0
    for file_name in f:
        i = i + 1
        if not (i % 10):
            print(i)
        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        # #print file_path
        y0, sr = librosa.load(file_path, sr=22050)
        # we use first 1 second
        half = len(y0)/4
        y = y0[:round(half)]
        # STFT
        S_full, phase = librosa.magphase(librosa.stft(y, n_fft=1024, window='hann', hop_length=256, win_length=1024))
        n = len(y)

        # Check the shape of matrix: row must corresponds to the example index !!!
        X = S_full.T
        
        # codebook by using K-Means Clustering
        K = 20
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
        features_kmeans = np.zeros(X.shape[0])
        # for each sample, summarize feature!!!
        codebook = np.zeros(K)
        for sample in range(X.shape[0]):
            features_kmeans[sample] = kmeans.labels_[sample]
        
        # codebook histogram!
        unique, counts = np.unique(features_kmeans, return_counts=True)

        for u in unique:
            u = int(u)
            codebook[u] = counts[u]
        # save mfcc as a file
        file_name = file_name.replace('.wav','.npy')
        save_file = codebook_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, codebook)

    f.close();

if __name__ == '__main__':
    extract_spec(dataset='train')                 
    extract_spec(dataset='valid')                                  
    extract_spec(dataset='test')
    extract_codebook(dataset='train')
    extract_codebook(dataset='valid')
    extract_codebook(dataset='test')


