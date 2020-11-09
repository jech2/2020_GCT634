# Music Genre Classification
Music genre classification is an important task that can be used in many musical applications such as music search or recommender systems. 

Abstract:
* Implementation of CNN based models for music genre classification
* 1D conv, 2D conv, Extracted embeddings with 2-layer MLP, VGG and ResNet with different number of layers  
* Offline data augmentation ( white noise, time shift, time stretch )
* Segmentation of audio samples into 4 secs segments
* Finally, get 86% accuracy when using ResNet34 Finetuning and MSD musicnn together!

## Dataset
We use the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset which has been the most widely used in the music genre classification task. 
The dataset contains 30-second audio files including 10 different genres including reggae, classical, country, jazz, metal, pop, disco, hiphop, rock and blues. 
For this homework, we are going to use a subset of GTZAN with only 8 genres. You can download the subset from [this link](https://drive.google.com/file/d/1J1DM0QzuRgjzqVWosvPZ1k7MnBRG-IxS/view?usp=sharing).

Once you downloaded the dataset, unzip and move the dataset to your home folder. After you have done this, you should have the following content in the dataset folder.  
```
$ tar zxvf gtzan.tar.gz
$ cd gtzan
$ ls *
split:
test.txt  train.txt

wav:
classical  country   disco     hiphop    jazz      metal     pop       reggae
$ cd ..      # go back to your home folder for next steps
```

### Dependency
* Python 3.7
* Numpy 1.16.6
* Librosa 0.8.0
* Pytorch 1.6.0
* Scikit Learn 0.23.2
* musicnn
* tqdm

## Training CNNs from Scratch
The baseline model extracts mel-spectrogram and has a simple set of CNN model 
that includes convolutional layer, batch normalization, maxpooling and dense layer.


## Improving Algorithms 
* The first thing to do is to segment audio clips and generate more data. The baseline code utilizes the whole mel-spectrogram as an input to the network (e.g. 128x1287 dimensions). Try to make the network input between 3-5 seconds segment and average the predictions of the segmentations for an audio clip.

* You can try training a model using both mel-spectrograms and features extracted using the pre-trained models. The baseline code is using a pre-trained model trained on 19k songs, but `musicnn` also has models trained on 200k songs! Try using the model giving `model='MSD_musicnn'` option on feature extraction.

* You can try 1D CNN or 2D CNN models and choose different model parameters:
    * Filter size
    * Pooling size
    * Stride size 
    * Number of filters
    * Model depth
    * Regularization: L2/L1 and Dropout

* You should try different hyperparameters to train the model and optimizers:
    * Learning rate
    * Model depth
    * Optimizers: SGD (with Nesterov momentum), Adam, RMSProp, ...

* You can try different parameters (e.g. hop and window size) to extract mel-spectrogram or different features as input to the network (e.g. MFCC, chroma features ...). 

* You can also use ResNet or other CNNs with skip connections. 

* Furthermore, you can augment data using digital audio effects.
