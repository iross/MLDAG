#!/usr/bin/env python3

import os
import yaml
import argparse

import wandb


def run_init():
    with wandb.init() as run:
        config['wandb']['run_id'] = run.id
        

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspth(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('config file', type=str)
    config_pathname = parser.parse_args()
    with open(os.path.join(script_dir, config_pathname), 'r') as file:
        config = yaml.safe_load(file)

    # wandb login
    os.environ['WANDB_API_KEY'] = config['api_key']
    wandb.login()

    # initialize run
    wandb.agent(config['wandb']['sweep_id'], run_init)

    # write back updated config
    with open(os.path.join(script_dir, config_pathname), 'w') as file:
        yaml.dump(config, file)
