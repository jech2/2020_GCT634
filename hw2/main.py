import argparse
import numpy as np
import time
import os
import random
import librosa
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import confusion_matrix

from exp_utils import create_exp_dir, set_seed
from dataset import SpecDataset, EmbedDataset, SpecEmbedDataset, load_split, extract_melspec, save_augmentation
from model.Baseline import Baseline
from model.Q1 import Q1
from model.Q2 import Q2
from model.Q3 import SpecAndEmbed, Base2DCNN


genres = genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

class Trainer(object):
    def __init__(self, args=None):
        self.args = args
        #self.best_val_loss = 987654321.0
        self.best_val_acc = 0.0
        self.best_train_acc = 0.0
        
        # Dataset setting based on model selection
        if args.model == 'Q2':
            self.dataset_train = EmbedDataset(self.args.data_dir, 'train')
            self.dataset_val = EmbedDataset(self.args.data_dir, 'val')
            self.dataset_test = EmbedDataset(self.args.data_dir, 'test')
        else:
            # Load all spectrograms.
            if args.model == 'SpecAndEmbed':
                dataset_train = SpecEmbedDataset(self.args.data_dir, self.args.embed_dir, 'train')
                specs = [s[0] for s, _ in dataset_train]
            else:
                dataset_train = SpecDataset(self.args.data_dir, 'train')
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
            log_data_stat = f'min_time_dim_size : {min_time_dim_size} | mean : {mean} | std : {std}'
            self.args.logging(log_data_stat)
            
            # based on that, make spec dataset
            if args.model == 'SpecAndEmbed':
                self.dataset_train = SpecEmbedDataset(self.args.data_dir, self.args.embed_dir, 
                                    'train', mean, std, min_time_dim_size, self.args.model2)
                self.dataset_val = SpecEmbedDataset(self.args.data_dir, self.args.embed_dir, 
                                    'val', mean, std, min_time_dim_size, self.args.model2)
                self.dataset_test = SpecEmbedDataset(self.args.data_dir, self.args.embed_dir, 
                                    'test', mean, std, min_time_dim_size, self.args.model2)
            else:
                self.dataset_train = SpecDataset(self.args.data_dir, 'train', mean, std, min_time_dim_size, self.args.model)
                self.dataset_val = SpecDataset(self.args.data_dir, 'val', mean, std, min_time_dim_size, self.args.model)
                self.dataset_test = SpecDataset(self.args.data_dir, 'test', mean, std, min_time_dim_size, self.args.model)

        # Each entry of the lists look like this
        log_data_len = f'train set : {self.dataset_train.__len__()} | val set : {self.dataset_val.__len__()} | test set : {self.dataset_test.__len__()}'
        self.args.logging(log_data_len)

        # Dataloader setting
        num_workers = os.cpu_count()

        # the drop_last argument drops the last non-full batch of each worker’s dataset replica.
        self.loader_train = DataLoader(self.dataset_train, batch_size=self.args.train_bsz, shuffle=True, num_workers=num_workers, drop_last=True)
        self.loader_val = DataLoader(self.dataset_val, batch_size=self.args.val_bsz, shuffle=True, num_workers=num_workers, drop_last=True)
        self.loader_test = DataLoader(self.dataset_test, batch_size=self.args.test_bsz, shuffle=False, num_workers=num_workers, drop_last=False)

        # Model selection
        if args.model == 'Q1':
            self.model = Q1(num_mels=self.args.num_mels, genres=genres)
        elif args.model == 'Q2':
            self.embed_size = self.dataset_train[0][0].shape[0]
            self.model = Q2(embed_size=self.embed_size, genres=genres)
        elif args.model == 'SpecAndEmbed':
            self.model = SpecAndEmbed(num_mels=self.args.num_mels, genres=genres, model=self.args.model2)
        elif args.model == 'Base2DCNN':
            self.model = Base2DCNN(genres)
        elif args.model == 'vgg13':
            self.model = models.vgg13(pretrained=True)
            fc_in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features=fc_in_features, out_features=len(genres))
        elif args.model == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            fc_in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features=fc_in_features, out_features=len(genres))
        elif args.model == 'vgg19':
            self.model = models.vgg19(pretrained=True)
            fc_in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features=fc_in_features, out_features=len(genres))
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
            self.model = Baseline(self.args.num_mels, genres)
        
        # Define a loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        # Setup an optimizer. for SGD, we use Stochastic gradient descent (SGD) with a nesterov mementum.
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

    # Util function for computing accuracy for segmented input(only for test) : max pooling.
    def segment_accuracy(self, source, target):
        if torch.unique(target).shape[0] is not 1 :
            self.args.logging('check test set is mistakenly shuffled or not')
            exit()
        source = source.max(1)[1].long().cpu()
        target = target.cpu()
        counts = torch.zeros(len(genres)) # counts : 0 ~ 7
        for i in range(self.args.test_bsz): # source : 0 ~ 6 
            counts[source[i]] += 1
        voted_label = torch.argmax(counts)
        label = target[0]
        correct = (voted_label == label).sum().item()
        return voted_label, label, correct

    def print_confusion_matrix(self, test_Y_c, test_Y_hat_c, labels=genres):
        self.args.logging(f'| confusion matrix for {genres}')
        cm = confusion_matrix(test_Y_c, test_Y_hat_c, labels=genres)
        self.args.logging(f'{cm}')

    # Validation
    def validate(self,i_train):
        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_acc = 0           
            for _, batch in enumerate(self.loader_val):
                # Data feeding is different when using SpecAndEmbed model 
                if self.args.model == 'SpecAndEmbed':
                    x = batch[0][0].to(self.device)
                    x_embed = batch[0][1].to(self.device)
                    y = batch[1].to(self.device)
                    prediction = self.model(x, x_embed)
                else :
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)
                    prediction = self.model(x)
                # Calculate loss and accuracy 
                loss = self.criterion(prediction, y)
                acc = self.accuracy(prediction, y)

                # Log training metrics.
                batch_size = len(x)
                epoch_loss += batch_size * loss.item()
                epoch_acc += batch_size * acc

            # Compute the evaluation scores.
            val_loss = epoch_loss / len(self.dataset_val)
            val_acc = epoch_acc / len(self.dataset_val)
            isBest = False
            # Based on validation accuracy or validation loss, best model is saved.
            if (val_acc > self.best_val_acc and self.args.debug is False):
                #print('best validation loss!')
                isBest = True
                self.best_val_acc = val_acc
                with open(os.path.join(self.args.work_dir, f'model_{i_train}.pt'), 'wb') as f:
                    torch.save(self.model, f)
                with open(os.path.join(self.args.work_dir, f'optimizer_{i_train}.pt'), 'wb') as f:
                    torch.save(self.optimizer.state_dict(), f)
            log_str = f'| val_loss {val_loss:5.2f} | val_acc {val_acc*100:5.2f}%'
            return log_str, isBest           

    # Train
    def train(self, i_train):
        # Iterate over epochs.
        for epoch in range(self.args.n_epochs):
            # Set the status of the model as training.
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0
            for _, batch in enumerate(self.loader_train):
                # Data feeding is different when using SpecAndEmbed model 
                if self.args.model == 'SpecAndEmbed':
                    x = batch[0][0].to(self.device)
                    x_embed = batch[0][1].to(self.device)
                    y = batch[1].to(self.device)
                    prediction = self.model(x, x_embed)
                else :
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)
                    prediction = self.model(x)
                # Compute the loss and accuracy.
                loss = self.criterion(prediction, y)
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

            # Log training metrics into logging.
            lr = self.optimizer.param_groups[0]['lr']
            train_loss = epoch_loss / len(self.dataset_train)
            train_acc = epoch_acc / len(self.dataset_train)
            log_str = f'| epoch {epoch:3d} | lr {lr:.3g} | train_loss {train_loss:5.2f} | train_acc {train_acc*100:5.2f}% '
            # Do validation and log update.          
            log_str_v, isBest = self.validate(i_train)
            log_str += log_str_v
            if isBest:
                self.best_train_acc = train_acc
            self.args.logging(log_str)
        self.args.logging(f'| best train accuracy : {self.best_train_acc*100:5.2f}%')
        self.args.logging(f'| best validation accuracy : {self.best_val_acc*100:5.2f}%')
        return self.best_train_acc, self.best_val_acc
    
    # Test : we only use this for final model
    def test(self,i_train):
        # Load the best saved model.
        if self.args.debug is False:
            self.args.logging('load the best saved model')
            with open(os.path.join(self.args.work_dir, f'model_{i_train}.pt'), 'rb') as f:
                self.model = torch.load(f)
            self.model = self.model.to(self.device)
        
        # Set the status of the model as evaluation.
        self.model.eval()
        # Variables to calculate the confusion matrix
        test_Y = None
        test_Y_hat = None
        with torch.no_grad():
            epoch_loss = 0
            epoch_acc = 0
            for _, batch in enumerate(self.loader_test):
                # Data feeding is different when using SpecAndEmbed model 
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
                
                # Compute the loss
                loss = self.criterion(prediction, y)
                # When using segmented input, use max pooling
                if self.args.use_segment:
                    pooled_Y_hat, Y, acc = self.segment_accuracy(prediction, y)
                    # Variables are appended to calculate the confusion matrix
                    if test_Y is None and test_Y_hat is None:
                        test_Y_hat = torch.Tensor([[pooled_Y_hat.item()]])
                        test_Y = torch.Tensor([[Y.item()]])
                    else:
                        pooled_Y_hat = torch.Tensor([[pooled_Y_hat.item()]])
                        Y = torch.Tensor([[Y.item()]])
                        test_Y_hat = torch.cat((test_Y_hat, pooled_Y_hat))
                        test_Y = torch.cat((test_Y, Y))
                else:
                    acc = self.accuracy(prediction, y)                
                    # Variables are appended to calculate the confusion matrix
                    if test_Y is None and test_Y_hat is None:
                        test_Y_hat = prediction.max(1)[1].long().cpu()
                        test_Y = y.cpu()
                    else:
                        test_Y_hat = torch.cat((test_Y_hat, prediction.max(1)[1].long().cpu()))
                        test_Y = torch.cat((test_Y, y.cpu()))

                # Log training metrics.
                batch_size = len(x)
                epoch_loss += batch_size * loss.item()
                epoch_acc += batch_size * acc

            # Compute the evaluation scores.            
            test_loss = epoch_loss / len(self.dataset_test)
            test_acc = epoch_acc / len(self.dataset_test)
            self.args.logging('=' * 100)
            log_str = f'| test_loss {test_loss:5.4f} | test_acc {test_acc*100:5.2f}%'
            self.args.logging(log_str)

            # Compute the confusion matrix
            test_Y_c = [genres[int(i)] for i in test_Y]
            test_Y_hat_c = [genres[int(i)] for i in test_Y_hat]
            
            # Print Confusion matrix
            self.print_confusion_matrix(test_Y_c, test_Y_hat_c, labels=genres)
            
            return test_acc

def main():
    parser = argparse.ArgumentParser(description="HW2 Training")
    parser.add_argument('--title', type=str, default="", help='experiment description')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    # Directory Settings
    parser.add_argument('--work_dir', default='experiment', type=str, help='experiment directory')
    parser.add_argument('--data_dir', default='gtzan/spec/', type=str, help='data directory')
    parser.add_argument('--embed_dir', default='gtzan/embed/', type=str, 
                        help='data directory for embedding when using both mel and embeddings')
    # Model 
    parser.add_argument('--model', type=str, default='Baseline',
                        choices=['Baseline', 'Q1', 'Q2', 'SpecAndEmbed','Base2DCNN', 
                        'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='backbone model (default is Baseline)')
    parser.add_argument('--model2', type=str, default='Base2DCNN', choices=['Q1','Base2DCNN', 'resnet34'],
                        help='backbone model for embed model')                        
    parser.add_argument('--num_mels', type=int, default=96, help='number of mel bins')
    # Optimizer 
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    # Training information
    parser.add_argument('--do_test', action='store_true', help='do test')
    parser.add_argument('--n_train', type=int, default=1, help='number of training with different seeds')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--use_segment', action='store_true', help='Use segmented data')
    # Batch size settings
    parser.add_argument('--train_bsz', type=int, default=32, help='train batch size')
    parser.add_argument('--val_bsz', type=int, default=16, help='train batch size')
    parser.add_argument('--test_bsz', type=int, default=7, help='train batch size')

    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemError('GPU device not found!')
    print(f'Found GPU at: {torch.cuda.get_device_name()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Librosa version: {librosa.__version__}')
    
    # Create Experiment Directory : if args.debug=False, not logging
    args.work_dir = f'{args.work_dir}/{args.model}'
    args.title = time.strftime('%Y%m%d-%H%M%S-') + args.title
    args.work_dir = os.path.join(args.work_dir, args.title)
    args.logging = create_exp_dir(args.work_dir, debug=args.debug)

    # Args information logging!
    args.logging('=' * 100)
    for k, v in args.__dict__.items():
        args.logging('    - {} : {}'.format(k, v))
    args.logging('=' * 100)

    train_accs = np.zeros(args.n_train)
    train_acc_log = "| train accs : "

    val_accs = np.zeros(args.n_train)
    val_acc_log = "| val accs : "

    test_accs = np.zeros(args.n_train)
    test_acc_log = "| test accs : "
    
    # total n_train number of independent training
    for i in range(args.n_train):
        try:
            args.logging(f'{i}th independent training')
            # Seed settings
            manualSeed = random.randint(1, 10000) # use if you want new results
            args.logging(f'Seed: {manualSeed}')
            set_seed(manualSeed)
            args.seed = manualSeed
            # Train
            trainer = Trainer(args)
            train_accs[i], val_accs[i] = trainer.train(i)
            train_acc_log += f'{train_accs[i]*100:5.2f}%, '
            val_acc_log += f'{val_accs[i]*100:5.2f}%, '
        except KeyboardInterrupt: # ctrl + C early stopping
            args.logging('-' * 100)
            args.logging('Exiting from training early')

        if args.do_test:
            test_accs[i] = trainer.test(i)
            test_acc_log += f'{test_accs[i]*100:5.2f}%, '
        args.logging('=' * 100)

    # Calculate average train, validation, test accuracy
    args.logging(train_acc_log)
    args.logging(val_acc_log)
    train_avg_acc = np.mean(train_accs)
    val_avg_acc = np.mean(val_accs)
    train_avg_log = f'| avg train acc for {args.n_train} trials is {train_avg_acc*100:5.2f}%'
    val_avg_log = f'| avg val acc for {args.n_train} trials is {val_avg_acc*100:5.2f}%'
    args.logging(train_avg_log)
    args.logging(val_avg_log)

    if args.do_test:
        args.logging(test_acc_log)
        test_avg_acc = np.mean(test_accs)
        test_avg_log = f'| avg test acc for {args.n_train} trials is {test_avg_acc*100:5.2f}%'
        args.logging(test_avg_log)
    args.logging('=' * 100)

if __name__ == '__main__':
    main()