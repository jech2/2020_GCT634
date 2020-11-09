# Music Genre Classification
Music genre classification is an important task that can be used in many musical applications such as music search or recommender systems.

## Abstract
* Implementation of CNN based models for music genre classification
* 1D conv, 2D conv, Extracted embeddings with 2-layer MLP, VGG and ResNet with different number of layers  
* Offline data augmentation ( white noise, time shift, time stretch )
* Segmentation of audio samples into 4 secs segments
* Finally, get 86% accuracy when using ResNet34 Finetuning and MSD musicnn together with augmentation and segmentation!

## Dataset
[GTZAN](http://marsyas.info/downloads/datasets.html) dataset has been the most widely used in the music genre classification task. 
The dataset contains 30-second audio files including 10 different genres including reggae, classical, country, jazz, metal, pop, disco, hiphop, rock and blues. 
In this project, a subset of GTZAN with only 8 genres is used. You can download the subset from [this link](https://drive.google.com/file/d/1J1DM0QzuRgjzqVWosvPZ1k7MnBRG-IxS/view?usp=sharing).

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

## Dependency
* Python 3.7
* Numpy 1.16.6
* [Librosa](https://librosa.org/) 0.8.0
* Pytorch 1.6.0
* Scikit Learn 0.23.2
* [musicnn](https://github.com/jordipons/musicnn)
* tqdm

## Avaliable Models
Basically, 1D CNN, 2D CNN, 2-layer MLP are implemented and you can check it in model directory.

For VGG and ResNet, pretrained VGG and ResNet weight was downloaded from [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html) and finetuned by changing the final FC layer.

* 1D CNN : Baseline, Q1
* 2D CNN : Base2DCNN
* 2-layer MLP : Q2
* VGG : vgg13, vgg16, vgg19
* ResNet : resnet18, resnet34, resnet50, resnet101

## Training CNNs from Scratch
The baseline model extracts mel-spectrogram and has a simple set of CNN model 
that includes convolutional layer, batch normalization, maxpooling and dense layer.

### Split Data
Once you downloaded the dataset, run split_data.py
```
$ python split_data.py
```
which will automatically splits 5 samples of each class of train set into validation set, and split wav files as a directory of gtzan/wav/train, gtzan/wav/val, gtzan/wav/test


### Feature Extraction
To extract mel spectrogram, which will be basic input of our CNNs, run dataset.py 
```
$ python dataset.py
```

### Train
To train a Baseline classifier, run main.py, which will show you Baseline Classifier training for 100 epochs. 
```
$ python main.py
```

To test, use --do_test command, and early stopping is available manually by ctrl + C.

For more detail of training options, please refer to the argparse of main.py


### (Optional) Augmentation, Segmentation
For offline data augmentation, adding white noise, time shift, time stretch is used at wav level. I'm planning to update more data augmentation methods such as pitch shifting.

Segmentation is done in mel-spectrogram level considering time stretch augmentation. By default, segment size is 4 seconds, making 7 samples for each audio clip in average.

To augment or segment data, see __main__() function of dataset.py and remove the comments that you need.