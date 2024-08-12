#!/usr/bin/env python3

import os
import shutil
import re
import yaml
import argparse



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('config_pathname', type=str)
    args = parser.parse_args()
    with open(os.path.join(script_dir, args.config_pathname), 'r') as file:
        config = yaml.safe_load(file)

    # create directory for sweep
    sweep_dir_pathname = os.path.join(script_dir, config['wandb']['sweep_id'])
    os.mkdir(sweep_dir_pathname)

    # move all related sweep files to the directory
    files = [f for f in os.listdir(script_dir) if os.path.isfile(os.path.join(script_dir, f))]
    pattern = re.compile(r'pipeline\..*|sweep.yaml|.*-config.yaml|.*\.h5|.*\.pt')
    for file in files:
        if pattern.match(file):
            shutil.move(file, sweep_dir_pathname)

    # move the logs directory
    logs_dir_pathname = os.path.join(script_dir, 'logs')
    shutil.move(logs_dir_pathname, sweep_dir_pathname)


if __name__ == '__main__':
    main()
