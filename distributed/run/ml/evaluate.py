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
    with torch.no_grad():
        validate_loss = 0.0
        for sequences, target in validate_loader:
            sequences = sequences.to(device)
            target = target.to(device).float()
            outputs = model(sequences)
            validate_loss += validation_criterion(outputs.squeeze(), target)
        validate_loss /= len(validate_loader.dataset)
        print(f'Validate Loss: {validate_loss}\n')

    return validate_loss

if __name__ == '__main__':

    # retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config file', type=str, help='Path to file that specifies sweep id and run id.')
    parser.add_argument('epoch', type=int help='Specifies which epoch will be evaluated')
    wandb_pathname, epoch = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # load in wandb info
    with open(os.path.join(script_dir, wandb_pathname), 'r') as file:
        config = yaml.safe_load(file)
    os.environ['WANDB_API_KEY'] = config['api_key']
    entity = config['wandb']['entity']
    project = config['wandb']['project']
    sweep_id = config['wandb']['sweep_id']
    run_id = config['wandb']['run_id']
    wandb.login()

    # load in io info
    tensor_pathname = config['tensor_pathname']
    model_pathname = config['model_pathname']

    # load in tensor from HDF5 file
    with h5py.File(tensor_pathname, 'r') as h5f:
        dataset = h5f['validate'][:]
        x = torch.as_tensor(dataset['timeseries'].copy())
        y = torch.as_tensor(dataset['label'].copy())

    # load in model object to train
    with open(model_pathname, 'r') as model_f
        model = torch.load(model_f)

    # resume run in wandb
    with wandb.init(entity=entity, project=project, id=run_id, resume='must') as run:
        validate_loss = evaluate(run.config, {'x':x, 'y':y}, model)
        wandb.log({'epoch': epoch, 'validate_loss': validate_loss}) # report to wandb
        
    print(f'epoch: {epoch}, validate_loss: {validate_loss}')
