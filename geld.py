#!/usr/bin/env python3

import argparse
import csv
import io
import os
import re
import json
import sys
import htcondor

event_types = [name for name in htcondor.JobEventType.names]

class JobInfoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for JobInfo"""

    def default(self, obj):
        if isinstance(obj, JobInfo):
            return obj.to_dict()
        elif isinstance(obj, htcondor.JobEvent):
            return dict(obj.items())
        
        try:
            sup = super().default(obj)
        except Exception as e:
            return None

        return sup


class JobInfo:
    """Object representing a single job log"""

    __slots__ = ['job_id', 'events']

    def __init__(self, job_id):
        """Initialize a job object with sensible values."""

        self.job_id = job_id
        self.events = []

    def update(self, event):
        self.events.append(event)

    def to_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}


def process_logs(event_log_list):
    jobs = {}
    for event_log_path in event_log_list:
        try:
            with htcondor.JobEventLog(event_log_path) as job_event_log:
                event_log_basename = os.path.basename(event_log_path)
                for event in job_event_log.events(stop_after=0):
                    job_id = f'{event.cluster}.{event.proc}'
                    if job_id not in jobs:
                        jobs[job_id] = JobInfo(job_id)
                    jobs[job_id].update(event)

        except htcondor.HTCondorIOError as e:
            # The message within the exception instance is not particularly helpful, so it is not printed
            print(f'Failed to read the log file at "{event_log_path}", so it was skipped.', file=sys.stderr)

    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description='Extract individual job data from job event logs and output as CSV.')
    parser.add_argument('logfile', nargs='+', help='job event log file(s)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    jobs = process_logs(args.logfile)

    # write to results to a json file
    with open('geld_out.json', 'w') as out:
        json.dump(jobs, out, cls=JobInfoJSONEncoder)


if __name__ == "__main__":
    main()
