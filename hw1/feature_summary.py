# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import mixture

data_path = './dataset/'
spec_path = './spec/'
codebook_path = './codebook/'

FEATURE_DIM = 91
K = 20
def sum_feature(dataset='train'):
    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        #feature_mat = np.zeros(shape=(3*FEATURE_DIM, 1000))
        feature_mat = np.zeros(shape=(3*FEATURE_DIM+K, 1000))
        #feature_mat = np.zeros(shape=(FEATURE_DIM, 1000))
    else:
        #feature_mat = np.zeros(shape=(3*FEATURE_DIM, 200))
        feature_mat = np.zeros(shape=(3*FEATURE_DIM+K, 200))
        #feature_mat = np.zeros(shape=(FEATURE_DIM, 200))

    i = 0
    for file_name in f:

        # load spec file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        spec_file = spec_path + file_name
        spec = np.load(spec_file)
        # mean, max, min pooling
        feature_mat[:FEATURE_DIM,i]= np.mean(spec, axis=1)
        feature_mat[FEATURE_DIM:2*FEATURE_DIM,i] = np.max(spec, axis=1)
        feature_mat[2*FEATURE_DIM:3*FEATURE_DIM,i] = np.min(spec, axis=1)
        
        # import codebook
        codebook_file = codebook_path + file_name
        codebook = np.load(codebook_file)
        feature_mat[3*FEATURE_DIM:3*FEATURE_DIM+K,i] = codebook
        i = i + 1

    f.close()

    return feature_mat

# when import codebook only
def import_codebook(dataset='train'):
    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        code_mat = np.zeros(shape=(K, 1000))
    else:
        code_mat = np.zeros(shape=(K, 200))

    i = 0
    for file_name in f:
        # load codebook file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        
        codebook_file = codebook_path + file_name
        codebook = np.load(codebook_file)
        code_mat[:K,i] = codebook
        i = i + 1

    f.close()

    return code_mat


if __name__ == '__main__':
    train_data = sum_feature('train')
    valid_data = sum_feature('valid')
    test_data = sum_feature('test')

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,3)
    plt.imshow(test_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.show()








