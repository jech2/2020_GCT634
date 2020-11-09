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
from feature_summary import *

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score

def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1):

    # Choose a classifier (here, linear SVM)
    clf = SVC(kernel=hyper_param1, random_state=1234, C=0.0005)
    # clf = KNeighborsClassifier(n_neighbors=10, algorithm=hyper_param1)
    #clf = MLPClassifier(random_state=1, alpha=0.0005, learning_rate_init=hyper_param1, activation='logistic', max_iter=1000)
    
    # train
    clf.fit(train_X, train_Y)

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/200.0*100.0
    print('lr : ', hyper_param1, ' validation accuracy = ',str(accuracy) ,' %')
    
    return clf, accuracy

def train_with_merged(train_X, train_Y, valid_X, valid_Y):
    clf = MLPClassifier(random_state=1, alpha=0.0005, learning_rate_init=0.1, activation='logistic', max_iter=1000)

    print('train on merged train dataset')
    X = np.concatenate([train_X, valid_X], axis=0)
    Y = np.concatenate([train_Y, valid_Y], axis=0)

    clf.fit(X, Y)
    return clf


if __name__ == '__main__':

    # load data
    train_X = sum_feature('train')
    valid_X = sum_feature('valid')
    test_X = sum_feature('test')
    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 100)
    valid_Y = np.repeat(cls, 20)
    test_Y = np.repeat(cls, 20)

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
    
    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)
    
    # training model
    alphas = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50]
    #alphas = ['auto', 'ball_tree', 'kd_tree', 'brute']
    #alphas = ['linear', 'poly', 'rbf', 'sigmoid']
    #alphas = ['identity', 'logistic', 'tanh', 'relu']
    
    model = []
    valid_acc = []
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a)
        model.append(clf)
        valid_acc.append(acc)
    
    # choose the model that achieve the best validation accuracy
    # final_model = model[np.argmax(valid_acc)]
    print('test using best model : lr 0.1')
    final_model = train_with_merged(train_X, train_Y, valid_X, valid_Y)
    # now, evaluate the model with the test set
    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X/(train_X_std + 1e-5)
    test_Y_hat = final_model.predict(test_X)

    accuracy = np.sum((test_Y_hat == test_Y))/200.0*100.0
    print('test accuracy = ',str(accuracy),' %')

    # draw confusion matrix
    classes = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string','vocal']
    test_Y_c = [classes[i-1] for i in test_Y]
    test_Y_hat_c = [classes[i-1] for i in test_Y_hat]
    print(confusion_matrix(test_Y_c, test_Y_hat_c, labels=classes))
