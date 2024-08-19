#!/usr/bin/env python3

import os
import yaml
import argparse

import wandb


def run_init():
    with wandb.init(resume='never') as run:
        config['wandb']['run_id'] = run.id
        run.config.update({
            'run_prefix': args.run_prefix # important for final node to cross-reference given only sweep_id and run_id when fetching best run
        })

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('config_pathname', type=str)
    parser.add_argument('run_prefix', type=str)
    parser.add_argument('output_config_pathname', type=str)
    args = parser.parse_args()
    with open(os.path.join(script_dir, args.config_pathname), 'r') as file:
        config = yaml.safe_load(file)

    # wandb login
    os.environ['WANDB_API_KEY'] = config['wandb']['api_key']
    os.environ['WANDB_ENTITY'] = config['wandb']['entity']
    os.environ['WANDB_PROJECT'] = config['wandb']['project']
    wandb.login()

    # initialize run
    wandb.agent(config['wandb']['sweep_id'], run_init, count=1)

    # write back updated config
    with open(os.path.join(script_dir, args.output_config_pathname), 'w') as file:
        yaml.dump(config, file)
