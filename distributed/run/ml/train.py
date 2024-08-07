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
    model.to(device)
    criterion.to(device)

    # train the model for a single epoch
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
    print(f'Training loss: {training_loss}')

    return model


if __name__ == '__main__':

    # retrieve arguments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('config_pathname', type=str)
    parser.add_argument('tensor_pathname', type=str)
    parser.add_argument('model_pathname', type=str)
    parser.add_argument('output_model_pathname', type=str)
    args = parser.parse_args()

    # load config
    with open(os.path.join(script_dir, args.config_pathname), 'r') as file:
        config = yaml.safe_load(file)

    # login to wandb
    os.environ['WANDB_API_KEY'] = config['wandb']['api_key']
    os.environ['WANDB_ENTITY'] = config['wandb']['entity']
    os.environ['WANDB_PROJECT'] = config['wandb']['project']
    os.environ['WANDB_RUN_ID'] = config['wandb']['run_id']
    wandb.login()

    # load in train tensor from HDF5 file
    with h5py.File(args.tensor_pathname, 'r') as h5f:
        train_dataset = h5f['train'][:]
        x_train = torch.as_tensor(train_dataset['timeseries'].copy())
        y_train = torch.as_tensor(train_dataset['label'].copy())

    # load in model object to train
    model = torch.jit.load(args.model_pathname)

    # resume run in wandb
    with wandb.init(resume='must') as run:
        model = train(run.config, model)

    # save new model
    torch.jit.save(model, args.output_model_pathname)
    print(f'model saved at: {args.output_model_pathname}')
