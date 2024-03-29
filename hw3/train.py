import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transcriber, Transcriber_CRNN, Transcriber_ONF, Transcriber_RNN, Transcriber_udRNN
from dataset import MAESTRO_small, allocate_batch
from evaluate import evaluate
from constants import HOP_SIZE


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def train(model_type, logdir, batch_size, iterations, validation_interval, sequence_length, learning_rate, weight_decay, cnn_unit, fc_unit, debug=False, save_midi=False, n_train=1):
    # Set the log directory
    if logdir is None:
        logdir = Path('runs') / ('exp_' + datetime.now().strftime('%y%m%d-%H%M%S')+'_'+model_type)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    

    # Make sequence length as the multiples of HOP_SIZE -> why?
    if sequence_length % HOP_SIZE != 0:
        adj_length = sequence_length // HOP_SIZE * HOP_SIZE
        print(f'sequence_length: {sequence_length} is not divide by {HOP_SIZE}.\n \
                adjusted into : {adj_length}')
        sequence_length = adj_length
    
    # Dataset setting
    if debug:
        dataset = MAESTRO_small(groups=['debug'], sequence_length=sequence_length, hop_size=HOP_SIZE, random_sample=True)
        valid_dataset = dataset
        iterations = 100
        validation_interval = 10
    else:
        dataset = MAESTRO_small(groups=['train'], sequence_length=sequence_length, hop_size=HOP_SIZE, random_sample=True)
        valid_dataset = MAESTRO_small(groups=['validation'], sequence_length=sequence_length, hop_size=HOP_SIZE, random_sample=False)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    # Device setting
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    print(th.cuda.device_count(), th.cuda.current_device())
    # Model setting
    if model_type == 'baseline':
        model = Transcriber(cnn_unit=cnn_unit, fc_unit=fc_unit)
    elif model_type == 'rnn':
        model = Transcriber_RNN(cnn_unit=cnn_unit, fc_unit=fc_unit)
    elif model_type == 'crnn':
        model = Transcriber_CRNN(cnn_unit=cnn_unit, fc_unit=fc_unit)
    elif model_type == 'ONF':
        model = Transcriber_ONF(cnn_unit=cnn_unit, fc_unit=fc_unit)
    elif model_type == 'udrnn':
        model = Transcriber_udRNN(cnn_unit=cnn_unit, fc_unit=fc_unit)
    optimizer = th.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.98)
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)

    # Training : why not using batch enumerate and using custom cycle function
    loop = tqdm(range(1, iterations+1))
    
    try:
        for step, batch in zip(loop, cycle(loader)):
            optimizer.zero_grad()
            batch = allocate_batch(batch, device) # oh this is useful

            # Feed the input to model(audio -> frame and onset logit : just a classification)
            frame_logit, onset_logit = model(batch['audio'])
            frame_loss = criterion(frame_logit, batch['frame'])
            onset_loss = criterion(onset_logit, batch['onset'])
            loss = onset_loss + frame_loss

            loss.mean().backward()

            # What clip_grad_norm does?
            for parameter in model.parameters():
                clip_grad_norm_([parameter], 3.0)

            optimizer.step()
            scheduler.step()
            loop.set_postfix_str("loss: {:.3e}".format(loss.mean()))

            if step % validation_interval == 0:
                model.eval()
                with th.no_grad():
                    loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                    metrics = defaultdict(list)
                    for batch in loader:
                        batch_results = evaluate(model, batch, device)
                        
                        for key, value in batch_results.items():
                            metrics[key].extend(value)
                print('')
                with open(Path(logdir) / 'results.txt', 'a+') as f:
                    for key, value in metrics.items():
                        if key[-2:] == 'f1' or 'loss' in key:
                            eval_string = f'{key:27} : {np.mean(value):.4f}'
                            print(eval_string)
                            f.write(eval_string + '\n')
                    f.write('\n')
                model.train()
    except KeyboardInterrupt: # ctrl + C early stopping
        with open(Path(logdir) / 'results.txt', 'a+') as f:
            dashes = '-' * 100
            print(dashes)
            f.write(dashes+'\n')
            early_log = 'Exiting from training early'
            print(early_log)
            f.write(early_log+'\n')

    # Save the results and delete dataset
    th.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step' : step,
            'cnn_unit' : cnn_unit,
            'fc_unit' : fc_unit
            },
            Path(logdir) / f'model-{step}.pt')
    del dataset, valid_dataset 
    
    test_dataset = MAESTRO_small(groups=['test'], hop_size=HOP_SIZE, random_sample=False)
    model.eval()
    with th.no_grad():
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        metrics = defaultdict(list)
        for batch in loader:
            batch_results = evaluate(model, batch, device, save=save_midi, save_path=logdir)
            for key, value in batch_results.items():
                metrics[key].extend(value)
    print('')
    for key, value in metrics.items():
        if key[-2:] == 'f1' or 'loss' in key:
            print(f'{key} : {np.mean(value)}')

    with open(Path(logdir) / 'results.txt', 'a+') as f:
        for key, values in metrics.items():
            _, category, name = key.split('/')
            metric_string = f'{category:>32} {name:26}: {np.mean(values):.3f} +- {np.std(values):.3f}'
            print(metric_string)
            f.write(metric_string + '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='baseline', type=str)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('-v', '--sequence_length', default=102400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=6e-4, type=float)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-i', '--iterations', default=10000, type=int)
    parser.add_argument('-vi', '--validation_interval', default=1000, type=int)
    parser.add_argument('-wd', '--weight_decay', default=0)
    parser.add_argument('-cnn', '--cnn_unit', default=48, type=int)
    parser.add_argument('-fc', '--fc_unit', default=256, type=int)
    parser.add_argument('--save_midi', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_train', default=1, type=int, help='number of independent trains') 
    args = parser.parse_args()

    for i in range(0, args.n_train):
        train(**vars(parser.parse_args()))  