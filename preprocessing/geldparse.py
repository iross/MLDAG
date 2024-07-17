#!/usr/bin/env python3

from datetime import datetime, timedelta
import random
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
    then disregards it. 
    """

    t = 'transient'
    nt = 'non-transient'
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
        if stime <= seconds_from_pre_dhhmmss(['EventTime']) <= etime:
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
    spartition_idx = len(filtered)
    return labeled_filtered + retenate, spartition_idx


def create_time_series(partitioned_jobs, spartition_idx):
    """
    Provided a partitioned job vector, create the 3D tensor training data.
    
    Returns:
        
    """
    
    m = 10
    e = 46
    j = 10
    
    timeframe_len = 10 # duration of a single timeframe in seconds
    num_timeframes = 10 # number of timeframes in a job matrix
    time_series_stack = []
    
    def construct_matrix():
        """
        Helper function for constructing a job's matrix
        
        Returns:
            None if job does not exist across the time frames,
            otherwise the one-hot encoding matrix.
        """
        event_vector = [] # a condensed form that represents events in the lifecycle of a job up to a cycle
        
        # calculate the time frames given the timestamp of hold event
        timeframes_events = []
        fcycle_time = job_info['events'][cycles_idx[1][0]] # timestamp of first cycle's hold event
        for i in range(num_timeframes):
            stime = fcycle_time - (timeframe_len * (i + 1))
            etime = fcycle_time - (timeframe_len * i)
            qualified_events = events_within_interval(job_info['events'], stime, etime)
            timeframes_events.append(qualified_events)
            
        # populate the event_vector
        emptiness_log = []
        for i, events in enumerate(timeframes_events):
            selected_event = None
            if events.empty():
                emptiness_log.append(True)
                if i > 0 and not (prev_events := timeframes_events[i-1]).empty():
                    selected_event = prev_events[-1]
            else: # not empty
                if i == 0:
                    selected_event = events[0]
                else:
                    selected_event = events[-1]
            event_vector[i] = selected_event['EventTypeNumber']
        if len(emptiness_log) == len(timeframes_events):
            return None # indicates job did not exist across the entire timeframes
                    
        # expand the event_vector into m*e matrix
        m_e = [[0 for _ in range(e)] for _ in range(m)] # initialize m*e matrix
        for i, event in enumerate(event_vector):
            m_e[event][i] = 1
            
        return m_e
    
    selected_jobs = {}
    def job_selection():
        """
        selects a random job given the partitioned jobs list
        
        Returns:
            The index of randomly selected job.
        """
    
        job_idx = random.randrange(len(partitioned_jobs))
        while selected_jobs[job_idx] is True:
            job_idx = random.randrange(len(partitioned_jobs))
        selected_jobs[job_idx] = True
        return job_idx
        
    for job_info, cycles_idx in partitioned_jobs[:spartition_idx]:
    
        # filter job (first m*e slice in m*e*j tensor)
        m_e = construct_matrix(job_info, cycles_idex)
        time_series_stack.append(m_e)
        
        # uniformly select j-1 other jobs for context window
        next_m_e = construct_matrix(partitioned_jobs[job_idx][0], partitioned_jobs[job_idx][1])
        while next_m_e is None:
            next_m_e = construct_matrix(partitioned_jobs[job_idx][0], partitioned_jobs[job_idx][1])
        time_series_stack.append(next_m_e)
        
    return time_series_stack # m*e*j tensor
                    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('geld', type=str, help='Geld relative path.')
    parser.add_argument('out', type=str, help='Output filename.')
    return parser.parse_args()


def main():
    args = parse_args()

    # open geld file
    with open(args.geld, 'r') as f:
        jobs = json.load(f)

    # get filtered jobs and write to json
    global_list = partition_jobs(jobs)
    ts = create_time_series(global_list)
    with open(args.out, 'w') as out:
        json.dump(ts, out)
    

if __name__ == "__main__":
    main()
