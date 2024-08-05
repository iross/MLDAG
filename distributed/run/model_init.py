#!/usr/bin/env python3

import os
import json
import yaml

import htcondor
import wandb
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim


class LSTMNet(nn.Module):
    """Define the LSTM network"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Get the last time step output
        out = self.sigmoid(out)

        return out


if __name__ == '__main__':
    """Responsible for model creation with specified hyperparameters from wandb."""
    script_dir = os.path.dirname(os.path.abspth(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    config_pathname = parser.parse_args()
    with open(os.path.join(script_dir, config_pathname), 'r') as file:
        config = yaml.safe_load(file)

    ## get hyperparameters from run to create LSTM model
    with wandb.init(entity=config['wandb']['entity'], 
                    project=config['wandb']['project'], 
                    id=config['wandb']['run_id'], 
                    resume='must') as run:
        # input size is slice which does not contain the time axis. (e*j)
        input_size = len(htcondor.JobEventType.names) * config['wandb']['run_config']['j']
        model = LSTMNet(input_size,
                        config['wandb']['run_config']['hidden_size'],
                        config['wandb']['run_config']['num_layers'],
                        config['wandb']['run_config']['num_classes']
        )

        # save model for the run
        torch.save(model, config['io']['model_name'])
    

