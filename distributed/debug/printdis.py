#!/usr/bin/env python3

import os
import json
import yaml
import argparse
import shutil

import wandb
import numpy as np
import h5py


if __name__ == '__main__':

    # retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_pathname', type=str)
    parser.add_argument('tensor_pathname', type=str)
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

        print(len(h5f['train']['label']))
        print(len(h5f['validate']['label']))
        print(len(h5f['test']['label']))

