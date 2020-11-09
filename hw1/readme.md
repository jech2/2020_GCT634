# Musical Instrument Recognition
Musical instrument recognition is a fundamental task in understanding music by computers. 

## Abstract
* Implementation of machine learning approach for musical instrument recognition
* For extracting features, MFCC, delta MFCC, double delta MFCC, Spectral Centroid are used.
* For summarizing features, mean, max, min pooling and codebook are used
* After parameter search, best model got 89% accuracy on test set; merged the training set and validation set as training set

## Dataset
[NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) is a large collection of musical instrument tones from the Google Magenta project. In this project, a subset with 10 classes of different musical instruments, including bass, brass, flute, guitar, keyboard, mallet, organ, reed, string and vocal was used. For experiment, it is split into training, validation and test sets. For each class, the training set has 100 audio samples and both validation and test sets have 20 audio samples. You can download the subset [here](https://drive.google.com/drive/folders/1uewIV8Mm4xXCYnkj9nglg5TFsbpnpgDE?usp=sharing). 

Once you downloaded the dataset, make sure that you have the following files and folders.  

```
$ ls 
test test_list.txt train train_list.txt valid valid_list.txt
$ cd ..      # go back to your home folder for next steps
```

## Getting Started
Main algorithm consists of three parts: Feature Extraction, Feature Summary, Train and Test.

* feature_extraction.py: loads audio files, extracts MFCC features using Librosa and stores them in the "mfcc" folder
* feature_summary.py: contains functions to summarize the extracted MFCC features
* train_test.py: train models and test it 


### Feature Extraction

```
$ python feature_extraction.py
```
If the run is successful (it takes some time), you will see that the "spec" and "codebook" folder is generated and it contains the extracted features:

### Feature summary



### Train and Test
Finally, run the traing and test code
```
$ python train_test.py
```

If the run is successful, it will display the validation and test accuracy values.  