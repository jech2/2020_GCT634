# randomly select 5 songs for each class from train dataset and split them as validation set (40 / 580) 
import pandas as pd
import numpy as np
import os
import shutil

from sklearn.model_selection import StratifiedShuffleSplit

from tqdm import tqdm
from dataset import load_split
# Stratified ShuffleSplit cross-validator 
# provides train/test indices to split data in train/test sets.

genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
genre_dict = {g: i for i, g in enumerate(genres)}
# Split validation set from train set
def split_val():
    train_list = load_split('gtzan/split/train.txt')
    #print(train_list)
    n_songs_for_val = 5 # use 5 songs for each class as validation
    n_val = n_songs_for_val * len(genres)
    labels = []
    for train_item in train_list:
        # Extract the genre from its path.
        genre = train_item.split('/')[-2]
        # Trun the genre into index number.
        labels.append(genre_dict[genre])

    train_list = np.array(train_list)
    labels = np.array(labels)
    split = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=1004)

    for train_idx, val_idx in split.split(train_list, labels):
        X_train = train_list[train_idx]
        X_val = train_list[val_idx]
        Y_train = labels[train_idx]
        Y_val = labels[val_idx]
    
    print('Y train ', np.unique(Y_train, return_counts=True))
    print('Y val ', np.unique(Y_val, return_counts=True))

    new_train_path = 'gtzan/split/splited_train.txt'
    new_val_path = 'gtzan/split/splited_val.txt'

    with open(new_train_path, 'w') as f:
        X_train = X_train.tolist()
        for item in X_train:
            f.write("%s\n" % item)
    with open(new_val_path, 'w') as f:
        X_val = X_val.tolist()
        for item in X_val:
            f.write("%s\n" % item)

def make_segmented_list():
    train_list = load_split('gtzan/split/splited_train.txt')
    val_list = load_split('gtzan/split/splited_val.txt')
    test_list = load_split('gtzan/split/test.txt')
    
    new_train_path = 'gtzan/split/splited_segmented_train.txt'
    new_val_path = 'gtzan/split/splited_segmented_val.txt'
    new_test_path = 'gtzan/split/segmented_test.txt'

    with open(new_train_path, 'w') as f:
        for path in tqdm(train_list):
            for idx in range(7):
                segmented_path = path.replace('.wav', f'_{idx}.wav')
                f.write("%s\n" % segmented_path)

    with open(new_val_path, 'w') as f:
        for path in tqdm(val_list):
            for idx in range(7):
                segmented_path = path.replace('.wav', f'_{idx}.wav')
                f.write("%s\n" % segmented_path)

    with open(new_test_path, 'w') as f:
        for path in tqdm(test_list):
            for idx in range(7):
                segmented_path = path.replace('.wav', f'_{idx}.wav')
                f.write("%s\n" % segmented_path)

# split files in txt into separate folder
def split_wavs():
    train_list = load_split('gtzan/split/splited_train.txt')
    val_list = load_split('gtzan/split/splited_val.txt')
    test_list = load_split('gtzan/split/test.txt')

    train_out_dir = 'gtzan/wav/train/'
    val_out_dir = 'gtzan/wav/val/'
    test_out_dir = 'gtzan/wav/test/'
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(val_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)

    for path_in in tqdm(train_list):
        src = f'gtzan/wav/{path_in}'
        shutil.copy(src, train_out_dir)
    for path_in in tqdm(val_list):
        src = f'gtzan/wav/{path_in}'
        shutil.copy(src, val_out_dir)
    for path_in in tqdm(test_list):
        src = f'gtzan/wav/{path_in}'
        shutil.copy(src, test_out_dir)

if __name__ == '__main__':
    split_val()
    #make_segmented_list()
    split_wavs()