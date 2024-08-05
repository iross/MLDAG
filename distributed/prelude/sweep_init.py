#!/usr/bin/env python3

import os
import yaml
import argparse

import wandb


def main():
    script_dir = os.path.dirname(os.path.abspth(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('config file', type=str)
    config_pathname = parser.parse_args()
    with open(os.path.join(script_dir, config_pathname), 'r') as file:
        config = yaml.safe_load(file)
    
    # login login
    os.environ['WANDB_API_KEY'] = config['api_key']
    wandb.login()

    # create sweep
    sweep_config = {
        **config['wandb']['sweep'],
        'parameters': {**config['preprocessing']['parameters'], **config['training']['parameters']}
    } 
    config['wandb']['sweep_id'] = wandb.sweep(sweep_config, project=config['project'])
    
    # write back updated config
    with open(os.path.join(script_dir, config_pathname), 'w') as file:
        yaml.dump(config, file)

if __name__ == '__main__': 
    main()
