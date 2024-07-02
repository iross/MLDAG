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


def add_label(jobs):
    """
    Iterate through jobs and labels it as transient or non-transient. If it cannot label,
    then delete it from jobs. 
    """

    t = 'transient'
    nt = 'non-transient'
    keys_to_delete = []

    # labeling
    for job_id, job_info in jobs.items():
        event = job_info['events'][-1]
            
        label = None
        if event['EventTypeNumber'] == htcondor.JobEventType.JOB_TERMINATED:
            label = (t if event['TerminatedNormally'] else nt)
        elif event['EventTypeNumber'] == htcondor.JobEventType.JOB_ABORTED:
            label = nt
        else:
            keys_to_delete.append(job_id)
            
        job_info['label'] = label

    # delete non-labeled data from jobs
    for key in keys_to_delete:
        del jobs[key]

    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description='Filtered geld File.')
    parser.add_argument('geld', type=str, help='Geld relative path.')
    parser.add_argument('out', type=str, help='Name of output file.', 
            default='label_out.json', nargs='?')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # open geld file
    with open(args.geld, 'r') as f:
        jobs = json.load(f)

    # label each job
    labeled_jobs = add_label(jobs)
    with open(args.out, 'w') as out:
        json.dump(labeled_jobs, out)
    

if __name__ == "__main__":
    main()


