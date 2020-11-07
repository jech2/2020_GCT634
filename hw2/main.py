import argparse
import numpy as np
import time
import os
import random
import librosa
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from exp_utils import create_exp_dir, set_seed
from dataset import SpecDataset, EmbedDataset, SpecEmbedDataset, load_split, extract_melspec, save_augmentation
from model.Baseline import Baseline, SegmentedBaseline
from model.Q1 import Q1
from model.Q2 import Q2
from model.Q3 import SpecAndEmbed, Base2DCNN
import torchvision.models as models
from sklearn.metrics import confusion_matrix

# Mel-spectrogram setup.
SR = 16000
FFT_HOP = 512
FFT_SIZE = 1024
NUM_MELS = 96

# Data processing setup.
BATCH_SIZE = 32
TEST_BATCH_SIZE = 7

genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

class Trainer(object):
    def __init__(self, args=None):
        self.args = args
        #self.best_val_loss = 987654321.0;
        self.best_val_acc = 0.0;
        # Load train and test data
        if args.use_segment:
            # self.train_path = load_split('gtzan/split/splited_segmented_train.txt')
            # self.val_path = load_split('gtzan/split/splited_segmented_val.txt')
            # self.test_path = load_split('gtzan/split/segmented_test.txt')
            self.train_path = glob('gtzan/seg_spec/train/*.npy')
            self.val_path = glob('gtzan/seg_spec/val/*.npy')
            self.test_path = sorted(glob('gtzan/seg_spec/test/*.npy'))
        else:
            # self.train_path = load_split('gtzan/split/splited_train.txt')
            # self.val_path = load_split('gtzan/split/splited_val.txt')
            # self.test_path = load_split('gtzan/split/test.txt')
            self.train_path = glob('gtzan/ori_features/train/*.npy')
            self.val_path = glob('gtzan/ori_features/val/*.npy')
            self.test_path = sorted(glob('gtzan/ori_features/test/*.npy'))
            #self.train_path = glob('gtzan/split/train/*.wav')
            # self.val_path = glob('gtzan/split/val/*.wav')
            # self.test_path = sorted(glob('gtzan/split/test/*.wav'))
            # self.train_path = glob('gtzan/ori_spec/train/*.npy')
            # self.val_path = glob('gtzan/ori_spec/val/*.npy')
            # self.test_path = sorted(glob('gtzan/ori_spec/test/*.npy'))

        # Each entry of the lists look like this:
        print('train set :', len(self.train_path), '| val set :', len(self.val_path), '| test set :', len(self.test_path))
        #save_augmentation('gtzan/split/')
        #extract_melspec('gtzan/aug_wav/', doSeg=self.args.use_segment)
        #segmentation(self.train_path, self.val_path, self.test_path)       

        if args.model == 'Q2':
            self.dataset_train = EmbedDataset(self.train_path)
            self.dataset_val = EmbedDataset(self.val_path)
            self.dataset_test = EmbedDataset(self.test_path)
        elif args.model == 'SpecAndEmbed':
            # Make directories to save mel-spectrograms.
            #extract_melspec(self.train_path, self.val_path, self.test_path)
            # Load all spectrograms.
            dataset_train = SpecEmbedDataset(self.train_path)
            specs = [s[0] for s, _ in dataset_train]
            # Compute the minimum temporal dimension size.
            time_dims = [s.shape[1] for s in specs]
            min_time_dim_size = min(time_dims)
            # Stack the spectrograms
            specs = [s[:, :min_time_dim_size] for s in specs]
            specs = np.stack(specs)
            # Compute mean and standard deviation for standard normalization.
            mean = specs.mean()
            std = specs.std()

            print('min_time_dim_size :', min_time_dim_size, '| mean :', mean, '| std :', std)

            # based on that, make spec dataset
            self.dataset_train = SpecEmbedDataset(self.train_path, mean, std, min_time_dim_size)
            self.dataset_val = SpecEmbedDataset(self.val_path, mean, std, min_time_dim_size)
            self.dataset_test = SpecEmbedDataset(self.test_path, mean, std, min_time_dim_size)
        else:
            # Make directories to save mel-spectrograms.
            #extract_melspec(self.train_path, self.val_path, self.test_path)
            #extract_melspec('gtzan/split/', doSeg=self.args.use_segment)
            # Load all spectrograms.
            dataset_train = SpecDataset(self.train_path)
            specs = [s for s, _ in dataset_train]
            # Compute the minimum temporal dimension size.
            time_dims = [s.shape[1] for s in specs]
            min_time_dim_size = min(time_dims)
            # Stack the spectrograms
            specs = [s[:, :min_time_dim_size] for s in specs]
            specs = np.stack(specs)
            # Compute mean and standard deviation for standard normalization.
            mean = specs.mean()
            std = specs.std()

            print('min_time_dim_size :', min_time_dim_size, '| mean :', mean, '| std :', std)

            # based on that, make spec dataset
            self.dataset_train = SpecDataset(self.train_path, mean, std, min_time_dim_size, self.args.model)
            self.dataset_val = SpecDataset(self.val_path, mean, std, min_time_dim_size, self.args.model)
            self.dataset_test = SpecDataset(self.test_path, mean, std, min_time_dim_size, self.args.model)

        num_workers = os.cpu_count()
        # the drop_last argument drops the last non-full batch of each worker’s dataset replica.
        self.loader_train = DataLoader(self.dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
        self.loader_val = DataLoader(self.dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
        self.loader_test = DataLoader(self.dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=num_workers, drop_last=False)

        if args.model == 'Q1':
            self.model = Q1(num_mels=NUM_MELS, genres=genres)
        elif args.model == 'Q2':
            self.embed_size = self.dataset_train[0][0].shape[0]
            self.model = Q2(embed_size=self.embed_size, genres=genres)
        elif args.model == 'SpecAndEmbed':
            self.model = SpecAndEmbed(num_mels=NUM_MELS, genres=genres)
        elif args.model == 'SegmentedBaseline':
            self.model = SegmentedBaseline(NUM_MELS, genres)
        elif args.model == 'Base2DCNN':
            self.model = Base2DCNN(genres)
        elif args.model == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=fc_in_features, out_features=len(genres))
        elif args.model == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=fc_in_features, out_features=len(genres))
        elif args.model == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=fc_in_features, out_features=len(genres))
        elif args.model == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=fc_in_features, out_features=len(genres))
        elif args.model == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=fc_in_features, out_features=len(genres))
        else:
            self.model = Baseline(NUM_MELS, genres)
        
        # Define a loss function, which is cross entropy here.
        self.criterion = torch.nn.CrossEntropyLoss()
        # Setup an optimizer. Here, we use Stochastic gradient descent (SGD) with a nesterov mementum.
        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, nesterov=True, weight_decay=self.args.weight_decay)
        # Choose a device. We will use GPU if it's available, otherwise CPU.
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Move variables to the desired device.
        self.model.to(device)
        self.criterion.to(device)
        self.device = device

    # Util function for computing accuracy.
    def accuracy(self, source, target):
        source = source.max(1)[1].long().cpu()
        target = target.cpu()
        correct = (source == target).sum().item()
        return correct / float(source.shape[0])

    # Util function for computing accuracy.
    def segment_accuracy(self, source, target):
        source = source.max(1)[1].long().cpu()
        target = target.cpu()
        voted_label = source[torch.argmax(source)]
        label = target[0]
        correct = (voted_label == label).sum().item()
        #print(source, target)
        #print(voted_label, label, correct)
        return correct

    def validate(self):
        # Set the status of the model as evaluation.
        self.model.eval()
        # `torch.no_grad()` disables computing gradients. The gradients are still 
        # computed even though you use `model.eval()`. You should use `torch.no_grad()` 
        # if you don't want your memory is overflowed because of unnecesary gradients.
        with torch.no_grad():
            epoch_loss = 0
            epoch_acc = 0
            #pbar = tqdm(self.loader_val, desc=f'Validation')  # progress bar
            #for x, y in pbar:
            for _, batch in enumerate(self.loader_val):
                # Move mini-batch to the desired device.
                if self.args.model == 'SpecAndEmbed':
                    x = batch[0][0].to(self.device)
                    x_embed = batch[0][1].to(self.device)
                    y = batch[1].to(self.device)
                    prediction = self.model(x, x_embed)

                else :
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)
                    
                    # Feed forward the model.
                    prediction = self.model(x)
                # Compute the loss.
                loss = self.criterion(prediction, y)
                # Compute the accuracy.
                acc = self.accuracy(prediction, y)

                # Log training metrics.
                batch_size = len(x)
                epoch_loss += batch_size * loss.item()
                epoch_acc += batch_size * acc
                # Update the progress bar.
                #pbar.set_postfix({'loss': epoch_loss / len(self.dataset_val), 'acc': epoch_acc / len(self.dataset_val)})

            # Compute the evaluation scores.
            val_loss = epoch_loss / len(self.dataset_val)
            val_acc = epoch_acc / len(self.dataset_val)

            if (val_acc > self.best_val_acc):
                #print('best validation loss!')
                self.best_val_acc = val_acc
                with open(os.path.join(self.args.work_dir, 'model.pt'), 'wb') as f:
                    torch.save(self.model, f)
                with open(os.path.join(self.args.work_dir, 'optimizer.pt'), 'wb') as f:
                    torch.save(self.optimizer.state_dict(), f)
            log_str = f'| val_loss {val_loss:5.3f} | val_acc {val_acc:5.3f}'
            return log_str           


    def train(self):
        # Iterate over epochs.
        for epoch in range(self.args.n_epochs):
            # Set the status of the model as training.
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0
            #pbar = tqdm(self.loader_train, desc=f'Epoch {epoch:02}')  # progress bar
            # for x, y in pbar
            for _, batch in enumerate(self.loader_train):
                # Move mini-batch to the desired device.
                if self.args.model == 'SpecAndEmbed':
                    x = batch[0][0].to(self.device)
                    x_embed = batch[0][1].to(self.device)
                    y = batch[1].to(self.device)
                    prediction = self.model(x, x_embed)

                else :
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)
                    # Feed forward the model.
                    prediction = self.model(x)
                # Compute the loss.
                loss = self.criterion(prediction, y)
                # Compute the accuracy.
                acc = self.accuracy(prediction, y)
                # Perform backward propagation to compute gradients.
                loss.backward()
                # Update the parameters.
                self.optimizer.step()
                # Reset the computed gradients.
                self.optimizer.zero_grad()

                # Log training metrics.
                batch_size = len(x)
                epoch_loss += batch_size * loss.item()
                epoch_acc += batch_size * acc
                # # Update the progress bar.
                # pbar.set_postfix({'loss': epoch_loss / len(self.dataset_train), 
                #                 'acc': epoch_acc / len(self.dataset_train)})
            lr = self.optimizer.param_groups[0]['lr']
            train_loss = epoch_loss / len(self.dataset_train)
            train_acc = epoch_acc / len(self.dataset_train)
            log_str = f'| epoch {epoch:3d} | lr {lr:.3g} | train_loss {train_loss:5.2f} | train_acc {train_acc:5.2f} '
                       
            log_str += self.validate()
            self.args.logging(log_str)
    
    def test(self):
        # Load the best saved model.
        print('load the best saved model')
        with open(os.path.join(self.args.work_dir, 'model.pt'), 'rb') as f:
            self.model = torch.load(f)
        self.model = self.model.to(self.device)
        
        # Set the status of the model as evaluation.
        self.model.eval()
        test_Y = None
        test_Y_hat = None
        # `torch.no_grad()` disables computing gradients. The gradients are still 
        # computed even though you use `model.eval()`. You should use `torch.no_grad()` 
        # if you don't want your memory is overflowed because of unnecesary gradients.
        with torch.no_grad():
            epoch_loss = 0
            epoch_acc = 0
            # pbar = tqdm(self.loader_test, desc=f'Test')  # progress bar
            #for x, y in pbar:
            for _, batch in enumerate(self.loader_test):
                # Move mini-batch to the desired device.
                if self.args.model == 'SpecAndEmbed':
                    x = batch[0][0].to(self.device)
                    x_embed = batch[0][1].to(self.device)
                    y = batch[1].to(self.device)
                    prediction = self.model(x, x_embed)
                else :
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)                    
                    # Feed forward the model.
                    prediction = self.model(x)
                    
                if test_Y is None and test_Y_hat is None:
                    test_Y_hat = prediction.max(1)[1].long().cpu()
                    test_Y = y.cpu()
                else:
                    test_Y_hat = torch.cat((test_Y_hat, prediction.max(1)[1].long().cpu()))
                    test_Y = torch.cat((test_Y, y.cpu()))
                
                # Compute the loss.
                loss = self.criterion(prediction, y)
                # Compute the accuracy.
                if self.args.use_segment:
                    acc = self.segment_accuracy(prediction, y)
                else:
                    acc = self.accuracy(prediction, y)
                    
                # Log training metrics.
                batch_size = len(x)
                epoch_loss += batch_size * loss.item()
                epoch_acc += batch_size * acc

                
                # Update the progress bar.
                # pbar.set_postfix({'loss': epoch_loss / len(self.dataset_test), 'acc': epoch_acc / len(self.dataset_test)})

            # Compute the evaluation scores.
            
            test_loss = epoch_loss / len(self.dataset_test)
            test_acc = epoch_acc / len(self.dataset_test)
            self.args.logging('=' * 100)
            log_str = f'| test_loss {test_loss:5.2f} | test_acc {test_acc:5.2f}'
            self.args.logging(log_str)

            test_Y_c = [genres[i] for i in test_Y]
            test_Y_hat_c = [genres[i] for i in test_Y_hat]
            print(confusion_matrix(test_Y_c, test_Y_hat_c, labels=genres))
            

            return test_acc

def main():
    parser = argparse.ArgumentParser(description="HW2 Training")
    parser.add_argument('--debug', action='store_true',
                    help='Debug mode')
    parser.add_argument('--model', type=str, default='Baseline',
                        choices=['Baseline', 'Q1', 'Q2', 'SpecAndEmbed', 'SegmentedBaseline','Base2DCNN', 'vgg16', 'resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='backbone model (default is Baseline)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam'],
                        help='optimizer')
    parser.add_argument('--work_dir', default='experiment', type=str,
                    help='experiment directory.')
    parser.add_argument('--n_test', type=int, default=1,
                    help='number of tests with different seeds')
    parser.add_argument('--n_epochs', type=int, default=10,
                    help='number of epochs')
    parser.add_argument('--use_segment', action='store_true',
                    help='Use segmented data')
    parser.add_argument('--use_augment', action='store_true',
                    help='Use augmented data')
    parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay')
    
    args = parser.parse_args()
    print('using model: ', args.model)
    if not torch.cuda.is_available():
        raise SystemError('GPU device not found!')
    print(f'Found GPU at: {torch.cuda.get_device_name()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Librosa version: {librosa.__version__}')
    
    args.work_dir = '{}-{}'.format(args.work_dir, args.model)
    args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
    args.logging = create_exp_dir(args.work_dir, debug=args.debug)

    # args information logging!
    args.logging('=' * 100)
    for k, v in args.__dict__.items():
        args.logging('    - {} : {}'.format(k, v))
    args.logging('=' * 100)

    test_accs = np.zeros(args.n_test)
    test_acc_log = "test accs : "
    for i in range(args.n_test):
        manualSeed = random.randint(1, 10000) # use if you want new results
        print("Seed: ", manualSeed)
        set_seed(manualSeed)
        args.seed = manualSeed

        trainer = Trainer(args)
        trainer.train()
        test_accs[i] = trainer.test()
        print(f'{i}th test')
        test_acc_log += (f'{test_accs[i]:5.2f},')
    args.logging(test_acc_log)
    test_avg_acc = np.mean(test_accs)
    test_avg_log = f'avg test accs for {args.n_test} trials is {test_avg_acc:5.2f}'
    args.logging(test_avg_log)

if __name__ == '__main__':
    main()