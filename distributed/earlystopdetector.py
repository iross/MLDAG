#!/usr/bin/env python3

import argparse
import os


def main(args):
    """ If file marker is found in script directory, then it will return -1 as exit code,
        else returns 0 to indicate no early termination. """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, args.marker_filename)):
        return -1 # early termination
    return 0 # normal termination

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('marker_filename', str=type)
    args = parser.parse()

    main(args)
