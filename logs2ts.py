#!/usr/bin/env python3

from datetime import datetime, timedelta
import argparse
import io
import os
import re
import json
import sys
import htcondor 


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


def convert_logs(jobs):
    """Turn logs into time series data. Returns a tuple(data, label)"""

    logs_data = []
    max_events = 65

    # find job with highest number of events for padding the rest
    """max_events = 0
    for job_info in jobs.values():
        max_events = max(len(job_info['events']), max_events)
    """

    # iterate through each job
    filtered_out = 0 # due to exceeding max_events
    for job_id, job_info in jobs.items():
        if len(job_info['events']) <= max_events:
            num_rows = max_events
            num_cols = len(htcondor.JobEventType.names)
            ts_matrix = [[0]*num_cols for i in range(num_rows)] # generate 2D matrix
        
            # iterate through each event of the job
            for r_idx, event in enumerate(job_info['events']):
                c_idx = event['EventTypeNumber']
                ts_matrix[r_idx][c_idx] = 1 # seconds_from_pre_dhhmmss(event['EventTime'])

            logs_data.append((ts_matrix, job_info['label']))
        else:
            filtered_out += 1

    print(f'# that exceeded max_events: {filtered_out}')
    return logs_data


def parse_args():
    parser = argparse.ArgumentParser(description='Convert logs into time-series data.')
    parser.add_argument('geld', type=str, help='Labeled logs json file.')
    parser.add_argument('out', type=str, help='Name of output file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # open geld file
    with open(args.geld, 'r') as f:
        jobs = json.load(f)

    # turn into time-series data
    ts_data = convert_logs(jobs)
    with open(f'{args.out}', 'w') as out:
        json.dump(ts_data, out)
    

if __name__ == "__main__":
    main()


