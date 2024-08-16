#!/usr/bin/env python3

import os
import json
import yaml
import argparse

import wandb
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def evaluate(config, validate, model):

    # model evaluation hyperparameters
    batch_size = config['batch_size']

    # set to eval mode
    model.eval()

    # validation
    validate_dataset = TensorDataset(validate['x'], validate['y'])
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=256)
    validation_criterion = nn.BCELoss()
 
    # set device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    validation_criterion.to(device)

    metrics = {}
    with torch.no_grad():
        loss = 0.0
        for sequences, target in validate_loader:
            sequences = sequences.to(device)
            target = target.to(device).float()
            outputs = model(sequences)
            loss += validation_criterion(outputs.squeeze(), target)

            # compute confusion matrix
            tp, fp, tn, fn = 0, 0, 0, 0
            predicted = (outputs > 0.5).float().squeeze()
            for i in range(len(predicted)):
                if predicted[i] == target[i] == True:
                    tp += 1 # true positive
                elif predicted[i] == True and target[i] == False:
                    fp += 1 # false positive
                elif predicted[i] == target[i] == False:
                    tn += 1 # true negative
                elif predicted[i] == False and target[i] == True:
                    fn += 1 # false negative

            # metrics
            epsilon = 1e-10 # prevents ZeroDivisionError
            metrics['precision'] = tp / (tp + fp + epsilon)
            metrics['recall'] = tp / (tp + fn + epsilon)
            metrics['f-measure'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + epsilon)


        loss /= len(validate_loader.dataset)
        metrics['validation_loss'] = loss.item()


    return metrics

if __name__ == '__main__':

    # retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_pathname', type=str)
    parser.add_argument('tensor_pathname', type=str)
    parser.add_argument('model_pathname', type=str)
    parser.add_argument('epoch', type=int)
    parser.add_argument('earlystop_marker_pathname', type=str)
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # create output directory
    output_dir = os.path.join(script_dir, 'output')
    os.mkdir(output_dir)
    print(f'created: {output_dir}')

    # load in wandb info
    with open(os.path.join(script_dir, args.config_pathname), 'r') as file:
        config = yaml.safe_load(file)

    # login to wandb
    os.environ['WANDB_API_KEY'] = config['wandb']['api_key']
    os.environ['WANDB_ENTITY'] = config['wandb']['entity']
    os.environ['WANDB_PROJECT'] = config['wandb']['project']
    os.environ['WANDB_RUN_ID'] = config['wandb']['run_id']
    wandb.login()

    # load in tensor from HDF5 file
    with h5py.File(args.tensor_pathname, 'r') as h5f:
        dataset = h5f['validate'][:]
        x = torch.as_tensor(dataset['timeseries'].copy())
        y = torch.as_tensor(dataset['label'].copy())

    # load in model object to train
    model = torch.jit.load(args.model_pathname)

    # resume run in wandb
    with wandb.init(resume='must') as run:
        metrics = evaluate(run.config, {'x':x, 'y':y}, model)
        run.log({'validation_loss': metrics['validation_loss'],
                 'precision': metrics['precision'],
                 'recall': metrics['recall'],
                 'f-measure': metrics['f-measure'], 
                 'epoch': args.epoch}) # report to wandb
        print(f'evaluation epoch={args.epoch} node logged to run: {run.id}')

        # early stopping condition
        history = wandb.Api().run(f"{config['wandb']['entity']}/{config['wandb']['project']}/{config['wandb']['run_id']}").history()
        counter = 0
        delta = 0
        for i, h in enumerate(history[np.max([0, args.epoch - 5]) : (args.epoch + 1)]): # previous 5 runs
            # checks for monontically increases
            vl = metrics['validation_loss'] if i == args.epoch else h['validation_loss']
            new_delta = vl - h['training_loss']
            if new_delta > delta:
                counter += 1
            else:
                counter = 0
            delta = new_delta

            # create file marker if threshold is reached
            if counter >= config['earlystop_threshold']:
                with open(args.earlystop_marker_pathname, 'w'):
                    pass

