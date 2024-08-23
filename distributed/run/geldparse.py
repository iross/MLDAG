#!/usr/bin/env python3

from datetime import datetime, timedelta
from pprint import pprint
import random
import argparse
import csv
import io
import os
import re
import json
import sys
import yaml

import htcondor
import numpy as np
import h5py
import wandb


def add_label(jobs):
    """
    Iterate through jobs and labels it as transient or non-transient. If it cannot label,
    then disregards it.
    """

    nt = 0 # transient
    t = 1 # non-transient
    labeled = []

    # labeling
    for job_info, cycles in jobs:
        event = job_info['events'][-1]

        label = None
        if event['EventTypeNumber'] == htcondor.JobEventType.JOB_TERMINATED:
            label = (t if event['TerminatedNormally'] else nt)
        elif event['EventTypeNumber'] == htcondor.JobEventType.JOB_ABORTED:
            label = nt
        else:
            continue # disregard the job

        labeled.append( (job_info, cycles, label) )

    return labeled


def seconds_from_pre_dhhmmss(duration_string):
    """Extracts duration elements from a string like 'Usr 9 07:50:07' and returns seconds"""

    match = re.search(r'T(\d{2}):(\d{2}):(\d{2})', duration_string)
    if match:
        hours = match.group(1)
        minutes = match.group(2)
        seconds = match.group(3)
        return (int(hours) * 3600) + (int(minutes) * 60) + int(seconds)
    else:
        print(f'Could not parse termination event duration: "{duration_string}"', file=sys.stderr)


def events_within_interval(events, stime, etime):
    """
    Provided a time interval in seconds, return a list of events within that interval inclusive.

    Returns:
        A list of events.
    """

    qualify = []
    for e in events:
        if stime <= seconds_from_pre_dhhmmss(e['EventTime']) <= etime:
            qualify.append(e)

    return qualify


def partition_jobs(jobs):
    """
    Given the geld as a dictionary, iterate through each job and filter accordingly.
    In addition, it labels the first partition of the jobs.

    Returns:
        A tuple (list, int) in which the list is a two partition vector
        The first partition contain filtered jobs where each is a tuple: (Job_Info, Cycles, Label).
        The second partition contains the retenate jobs: (Job_Info).

        The int is the index at which the second partition starts
    """

    filtered = [] # jobs with at-least one cycle
    retenate = [] # jobs with no cycles
    for job_id, job_info in jobs.items():
        held_idx = None
        cycle_detected = False
        cycles_idx = []
        for event_idx, event in enumerate(job_info['events']):
            if event['EventTypeNumber'] == htcondor.JobEventType.JOB_HELD:
                held_idx = event_idx
            elif event['EventTypeNumber'] == htcondor.JobEventType.JOB_RELEASED:
                if held_idx is not None: # indicates a cycle is found
                    cycles_idx.append( (held_idx, event_idx) )
                    cycle_detected = True
                    held_idx = None # reset

        if cycle_detected:
            filtered.append( (job_info, cycles_idx) )
        else:
            retenate.append(job_info)

    labeled_filtered = add_label(filtered)
    spartition_idx = len(labeled_filtered)
    return labeled_filtered + retenate, spartition_idx


def create_time_series(partitioned_jobs, spartition_idx, j, m, timeframe_len):
    """
    Provided a partitioned job vector, create the 3D tensor training data.

    Returns:

    """
    e = 46
    time_series_stack = []
    labels = []


    def construct_matrix(job_info, lastframe_etime):
        """
        Helper function for constructing a job's matrix.

        Parameters:
            job_info Object represntation of the job
            lastframe_etime Timestamp in seconds of the last time frame for the filter job.

        Returns:
            None if job does not exist across the time frames,
            otherwise the one-hot encoding matrix.
        """

        # calculate the time frames given the timestamp of hold event
        timeframes_events = []
        for i in range(m):
            stime = lastframe_etime - (timeframe_len * (i + 1))
            etime = lastframe_etime - (timeframe_len * i)
            qualified_events = events_within_interval(job_info['events'], stime, etime)
            timeframes_events.append(qualified_events)

        # a condensed form that represents events it its first cycle
        event_vector = [None for i in range(len(timeframes_events))]

        # populate the event_vector
        emptiness_log = []
        for i, events in enumerate(timeframes_events):
            selected_event = None
            if len(events) == 0: # empty
                emptiness_log.append(True)
                if i > 0 and len(prev_events := timeframes_events[i-1]) > 0:
                    selected_event = prev_events[-1] # imputation

                    # populates its own timeframe events
                    # for possible imputation from proceeding timeframes
                    timeframes_events[i].append(selected_event)
            else: # not empty
                if i == 0:
                    selected_event = events[0]
                else:
                    selected_event = events[-1]

            # possible for selected_event to be none in the case that job has yet to exist
            event_vector[i] = selected_event['EventTypeNumber'] if selected_event is not None else None
        if len(emptiness_log) == len(timeframes_events):
            return None # indicates job did not exist across the entire timeframes

        # expand the event_vector into m*e matrix
        m_e = [[0 for _ in range(e)] for _ in range(m)] # initialize m*e matrix
                                                        # NULL values are denoted as 1
        for i, event in enumerate(event_vector):
            if event is not None:
                m_e[i][event] = 1

        return m_e

    def job_selection(selected_jobs):
        """
        selects a random job given the partitioned jobs list

        Returns:
            The index of randomly selected job.
        """

        while True:
            job_idx = random.randrange(len(partitioned_jobs))
            if job_idx not in selected_jobs:
                break
        selected_jobs[job_idx] = True
        return job_idx

    # for each filtered job
    # create the job tensor j * m * e
    # new shape is (m, j*e)
    for i, (job_info, cycles_idx, label) in enumerate(partitioned_jobs[:spartition_idx]):

        job_tensor = []

        # filter job (first m*e slice in m*e*j tensor)
        fcycle_time = seconds_from_pre_dhhmmss(job_info['events'][cycles_idx[0][0]]['EventTime']) # timestamp of first cycle's hold event
        m_e = construct_matrix(job_info, fcycle_time)
        job_tensor.append(m_e)

        # uniformly select j-1 other jobs for context window
        selected_jobs = {} # keeps track of which jobs had been selected in random selection process
        m_e_count = 0
        while m_e_count < j - 1:
            job_idx = job_selection(selected_jobs)
            if type(partitioned_jobs[job_idx]) is tuple: # job is a filter job
                next_job_info = partitioned_jobs[job_idx][0]
            else:
                next_job_info = partitioned_jobs[job_idx]
            next_m_e = construct_matrix(next_job_info, fcycle_time)
            if next_m_e is not None: # success
                job_tensor.append(next_m_e)
                m_e_count += 1

        job_tensor = np.array(job_tensor).reshape( (m, j*e) )
        time_series_stack.append(job_tensor)
        labels.append(label)
        if i % 10 == 0: print(f'{i} / {spartition_idx}')

    return np.array(time_series_stack), np.array(labels) # spartition_idx * j * m * e tensor, and labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('geld', type=str, help='Geld file relative path.')
    parser.add_argument('out', type=str, help='Output filename.')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # open geld file
        with open(args.geld, 'r') as f:
            jobs = json.load(f)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except json.JSONDecodeError as ex:
        print(f"Error decoding JSON: {e}")
    except PermissionError:
        print(f"Permission denied for reading the file {file_path}.")
    except Exception as ex:
        print(f"An unexpected error occurred: {e}")


    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_pathname = args.config

    # load in config to get preprocessing hyperparameters
    with open(os.path.join(script_dir, config_pathname), 'r') as file:
        config = yaml.safe_load(file)

    # login to wandb
    os.environ['WANDB_API_KEY'] = config['wandb']['api_key']
    os.environ['WANDB_ENTITY'] = config['wandb']['entity']
    os.environ['WANDB_PROJECT'] = config['wandb']['project']
    os.environ['WANDB_RUN_ID'] = config['wandb']['run_id']
    wandb.login()

    # retrieve preprocessing hyperparameters
    run = wandb.init(resume='must')
    pprint(run.config)

    global_list, spartition_idx = partition_jobs(jobs)
    ts, labels = create_time_series(
            global_list,
            spartition_idx,
            run.config['j'],
            run.config['m'],
            run.config['timeframe_len']
            )
    print(f'size of global list, index of second partition = {len(global_list)}, {spartition_idx}')

    # partition into train, validate, test sets
    partition_dict = {'train': [], 'validate': [], 'test': []}
    for i, v in enumerate(ts):
        partition_dict[random.choice(list(partition_dict.keys()))].append( (v, labels[i]) )

    with h5py.File(args.out, 'w') as h5f:
        # convert each partition into a structured ndarray
        j = run.config['j']
        m = run.config['m']
        e = len(htcondor.JobEventType.names)
        for pname, partition in partition_dict.items():
            dtype = np.dtype([
                ('timeseries', np.float32, (m, j*e)),
                ('label', np.int8),
                ])
            dataset = np.zeros(len(partition), dtype=dtype)
            for i, (v, l) in enumerate(partition):
                dataset[i]['timeseries'] = v
                dataset[i]['label'] = l

            print(f'writing to h5, partition: {pname}')
            h5f.create_dataset(pname, data=dataset)

    print(f"finished: {args.out}")

if __name__ == "__main__":
    main()
