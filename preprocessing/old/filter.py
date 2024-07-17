#!/usr/bin/env python3
# -*- mode: python -*-

import argparse
import csv
import io
import os
import re
import json
import sys
import htcondor 


def job_filter(jobs):
    """Given the geld as a dictionary, iterate through each job and filter accordingly."""
    
    filtered = {}
    for job_id, job_info in jobs.items():
        held = False
        for event in job_info['events']:
            if event['EventTypeNumber'] == htcondor.JobEventType.JOB_HELD:
                held = True
            elif event['EventTypeNumber'] == htcondor.JobEventType.JOB_RELEASED:
                if held: filtered[job_id] = job_info

    return filtered


def parse_args():
    parser = argparse.ArgumentParser(description='Geld filtering. Only include jobs that went to hold and idle state.')
    parser.add_argument('geld', type=str, help='Geld relative path.')
    parser.add_argument('out', type=str, help='Output filename.')
    return parser.parse_args()


def main():
    args = parse_args()

    # open geld file
    with open(args.geld, 'r') as f:
        jobs = json.load(f)

    # get filtered jobs and write to json
    filtered = job_filter(jobs)
    with open(args.out, 'w') as out:
        json.dump(filtered, out)
    

if __name__ == "__main__":
    main()
