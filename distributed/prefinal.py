#!/usr/bin/env python3

import os
import json
import yaml
import argparse
import shutil
import glob


if __name__ == '__main__':

    # retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_pathname', type=str)
    parser.add_argument('output_pathname', type=str, help='pathname of directory for input files of final node')
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # load in wandb info
    with open(os.path.join(script_dir, args.config_pathname), 'r') as file:
        config = yaml.safe_load(file)
    
    # create input directory of FINAL NODE
    if not os.path.exists(args.output_pathname):
        os.mkdir(args.output_pathname)

    # copy HDF5 file to input directory of FINAL NODE
    file_pattern = os.path.join(script_dir, f"*.h5")
    src_files = glob.glob(file_pattern)
    for src_file in src_files:
        shutil.move(src_file, os.path.join(args.output_pathname, os.path.basename(src_file)))
    
    # copy last epoch .pt file to input directory of FINAL NODE
    src_files = glob.glob(os.path.join(script_dir, f"*.pt"))
    for src_file in src_files:
        shutil.move(src_file, os.path.join(args.output_pathname, os.path.basename(src_file)))

