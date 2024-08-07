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

    with torch.no_grad():
        validate_loss = 0.0
        for sequences, target in validate_loader:
            sequences = sequences.to(device)
            target = target.to(device).float()
            outputs = model(sequences)
            validate_loss += validation_criterion(outputs.squeeze(), target)
        validate_loss /= len(validate_loader.dataset)
        print(f'Validate loss: {validate_loss}\n')

    return validate_loss

if __name__ == '__main__':

    # retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_pathname', type=str)
    parser.add_argument('tensor_pathname', type=str)
    parser.add_argument('model_pathname', type=str)
    parser.add_argument('epoch', type=int)
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

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
        validate_loss = evaluate(run.config, {'x':x, 'y':y}, model)
        run.log({'epoch': args.epoch, 'validate_loss': validate_loss}) # report to wandb
        print(f'logged to run: {run.id}')

        # save best model if last epoch
        if args.epoch == run.config['max_epoch'] - 1: # zero based
            torch.jit.save(model, f'{sweep_id}_{run_id}-bestmodel.pt')
