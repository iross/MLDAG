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


def train(config, model):
    """Returns modified model"""
    
    # training hyperparameters
    num_epochs = config['epochs']           # Number of epochs
    batch_size = config['batch_size']       # Batch size
    learning_rate = config['learning_rate'] # Learning rate

    # create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # initialize the model, loss function, and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # set device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # train the model
    for epoch in range(num_epochs):
        training_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()

            # forward + optimization
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # post epoch procedures
        training_loss /= len(train_loader.dataset) # normalize loss
        if early_stopping(training_loss, model): break
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {training_loss}')
    
    return model


def sweep_model(config=None):
    """Trains the model with wandb integration"""
    with wandb.init(config=config) as run:
        config = run.config
        checkpoint_pathname = f'checkpoints/{run.project}_{run.sweep_id}_{run.id}.pt'
        train(config, checkpoint_pathname)


if __name__ == '__main__':

    # retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('wandb file', type=str, help='Path to file that specifies sweep id and run id.')
    parser.add_argument('io file', type=str, help='Path to file that specifies naming conventions for io operations.')
    wandb_pathname, io_pathname = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # load in wandb info
    with open(os.path.join(script_dir, wandb_pathname), 'r') as file:
        wandb_info = yaml.safe_load(file)
    os.environ['WANDB_API_KEY'] = wandb_info['api_key']
    project = wandb_info['project']
    entity = wandb_info['entity']
    sweep_id = wandb_info['sweep_id']
    run_id = wandb_info['run_id']
    wandb.login()

    # load in io info
    with open(os.path.join(script_dir, io_pathname), 'r') as file:
        io_info = yaml.safe_load(file)
    hyperparameters_pathname = io_info['hyperparameters_pathname']
    tensor_pathname = io_info['tensor_pathname']
    model_pathname = io_info['model_pathname']
    model_out_pathname = io_info['model_out_pathname']

    # load in train tensor from HDF5 file
    with h5py.File(tensor_pathname, 'r') as h5f:
        train_dataset = h5f['train'][:]
        x_train = torch.as_tensor(train_dataset['timeseries'])
        y_train = torch.as_tensor(train_dataset['label'])

    # load in model object to train
    with open(model_pathname, 'r') as model_f
        model = torch.load(model_f)

    # resume run in wandb
    with wandb.init(project=project, entity=entity, id=run_id, resume='must') as run:
        model = train(run.config, model)
        
    # save new model
    torch.save(model, model_out_pathname)
    print(f'model saved at: {model_out_pathname}')
