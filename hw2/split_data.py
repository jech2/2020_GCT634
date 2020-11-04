# randomly select 5 songs for each class from train dataset and split them as validation set (40 / 580) 
import pandas as pd
import numpy as np
from dataset import load_split
# Stratified ShuffleSplit cross-validator 
# provides train/test indices to split data in train/test sets.
from sklearn.model_selection import StratifiedShuffleSplit


genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']
genre_dict = {g: i for i, g in enumerate(genres)}

if __name__ == '__main__':
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