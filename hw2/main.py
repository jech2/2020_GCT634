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
from exp_utils import create_exp_dir
from dataset import SpecDataset,EmbedDataset, load_split, extract_melspec
from model.Baseline import Baseline
from model.Q1 import Q1
from model.Q2 import Q2
#from model.Q3 import Q3

# Mel-spectrogram setup.
SR = 16000
FFT_HOP = 512
FFT_SIZE = 1024
NUM_MELS = 96

# Data processing setup.
BATCH_SIZE = 4

genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

class Trainer(object):
    def __init__(self, args=None):
        self.args = args
        self.best_val_loss = 987654321.0;

        # Load train and test data
        self.train_path = load_split('gtzan/split/splited_train.txt')
        self.val_path = load_split('gtzan/split/splited_val.txt')
        self.test_path = load_split('gtzan/split/test.txt')

        # Each entry of the lists look like this:
        print('train set :', len(self.train_path), '| val set :', len(self.val_path), '| test set :', len(self.test_path))
        
        if args.model == 'Q2':
            self.dataset_train = EmbedDataset(self.train_path)
            self.dataset_val = EmbedDataset(self.val_path)
            self.dataset_test = EmbedDataset(self.test_path)
        else:
            # Make directories to save mel-spectrograms.
            extract_melspec(self.train_path, self.test_path)

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
            self.dataset_train = SpecDataset(self.train_path, mean, std, min_time_dim_size)
            self.dataset_val = SpecDataset(self.val_path, mean, std, min_time_dim_size)
            self.dataset_test = SpecDataset(self.test_path, mean, std, min_time_dim_size)

        num_workers = os.cpu_count()
        # the drop_last argument drops the last non-full batch of each worker’s dataset replica.
        self.loader_train = DataLoader(self.dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
        self.loader_val = DataLoader(self.dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
        self.loader_test = DataLoader(self.dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, drop_last=False)

        # Training setup.
        self.lr = 0.0006  # learning rate
        self.momentum = 0.9
        self.num_epochs = 50
        self.weight_decay = 0.0  # L2 regularization weight

        if args.model == 'Q1':
            self.model = Q1(NUM_MELS, genres)
        elif args.model == 'Q2':
            self.embed_size = self.dataset_train[0][0].shape[0]
            self.model = Q2(self.embed_size, genres)
        elif args.model == 'Q3':
            self.model = Q3(NUM_MELS, genres)
        else:
            self.model = Baseline(NUM_MELS, genres)
        
        # Define a loss function, which is cross entropy here.
        self.criterion = torch.nn.CrossEntropyLoss()
        # Setup an optimizer. Here, we use Stochastic gradient descent (SGD) with a nesterov mementum.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=True, weight_decay=self.weight_decay)
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

    def validate(self):
        # Set the status of the model as evaluation.
        self.model.eval()
        # `torch.no_grad()` disables computing gradients. The gradients are still 
        # computed even though you use `model.eval()`. You should use `torch.no_grad()` 
        # if you don't want your memory is overflowed because of unnecesary gradients.
        with torch.no_grad():
            epoch_loss = 0
            epoch_acc = 0
            pbar = tqdm(self.loader_val, desc=f'Validation')  # progress bar
            for x, y in pbar:
                # Move mini-batch to the desired device.
                x = x.to(self.device)
                y = y.to(self.device)

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
                pbar.set_postfix({'loss': epoch_loss / len(self.dataset_val), 'acc': epoch_acc / len(self.dataset_val)})

            # Compute the evaluation scores.
            val_loss = epoch_loss / len(self.dataset_val)
            val_acc = epoch_acc / len(self.dataset_val)

            if (val_loss < self.best_val_loss):
                print('best validation loss!')
                self.best_val_loss = val_loss
                with open(os.path.join(self.args.work_dir, 'model.pt'), 'wb') as f:
                    torch.save(self.model, f)
                with open(os.path.join(self.args.work_dir, 'optimizer.pt'), 'wb') as f:
                    torch.save(self.optimizer.state_dict(), f)
            log_str = f'| val_loss | {val_loss:.5f}, val_acc | {val_acc * 100:.2f}%'
            self.args.logging(log_str)           


    def train(self):
        # Iterate over epochs.
        for epoch in range(self.num_epochs):
            # Set the status of the model as training.
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0
            pbar = tqdm(self.loader_train, desc=f'Epoch {epoch:02}')  # progress bar
            for x, y in pbar:
                # Move mini-batch to the desired device.
                x = x.to(self.device)
                y = y.to(self.device)

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
                # Update the progress bar.
                pbar.set_postfix({'loss': epoch_loss / len(self.dataset_train), 
                                'acc': epoch_acc / len(self.dataset_train)})

            log_str = '| epoch {:3d} | lr {:.3g} | loss {:5.2f} | accruacy {:5.2f}'.format(epoch, self.optimizer.param_groups[0]['lr'], epoch_loss / len(self.dataset_train), epoch_acc / len(self.dataset_train))
            self.args.logging(log_str)
            
            self.validate()
    
    def test(self):
        # Load the best saved model.
        print('load the best saved model')
        with open(os.path.join(self.args.work_dir, 'model.pt'), 'rb') as f:
            self.model = torch.load(f)
        self.model = self.model.to(self.device)
        
        # Set the status of the model as evaluation.
        self.model.eval()

        # `torch.no_grad()` disables computing gradients. The gradients are still 
        # computed even though you use `model.eval()`. You should use `torch.no_grad()` 
        # if you don't want your memory is overflowed because of unnecesary gradients.
        with torch.no_grad():
            epoch_loss = 0
            epoch_acc = 0
            pbar = tqdm(self.loader_test, desc=f'Test')  # progress bar
            for x, y in pbar:
                # Move mini-batch to the desired device.
                x = x.to(self.device)
                y = y.to(self.device)

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
                pbar.set_postfix({'loss': epoch_loss / len(self.dataset_test), 'acc': epoch_acc / len(self.dataset_test)})

            # Compute the evaluation scores.
            test_loss = epoch_loss / len(self.dataset_test)
            test_acc = epoch_acc / len(self.dataset_test)
            self.args.logging('=' * 100)
            log_str = f'| test_loss {test_loss:.5f} | test_acc {test_acc * 100:.2f}%'
            self.args.logging(log_str)
            

def main():
    parser = argparse.ArgumentParser(description="HW2 Training")
    parser.add_argument('--model', type=str, default='Baseline',
                        choices=['Baseline', 'Q1', 'Q2', 'Q3'],
                        help='backbone model (default is Baseline)')
    parser.add_argument('--work_dir', default='experiment', type=str,
                    help='experiment directory.')
    args = parser.parse_args()
    print('using model: ', args.model)
    if not torch.cuda.is_available():
        raise SystemError('GPU device not found!')
    print(f'Found GPU at: {torch.cuda.get_device_name()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Librosa version: {librosa.__version__}')
    manualSeed = random.randint(1, 10000) # use if you want new results
    #manualSeed = 333
    print("Seed: ", manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args.seed = manualSeed
    args.work_dir = '{}-{}'.format(args.work_dir, args.model)
    args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
    args.logging = create_exp_dir(args.work_dir)

    # args information logging!
    args.logging('=' * 100)
    for k, v in args.__dict__.items():
        args.logging('    - {} : {}'.format(k, v))
    args.logging('=' * 100)

    trainer = Trainer(args)
    trainer.train()
    
    trainer.test()

if __name__ == '__main__':
    main()