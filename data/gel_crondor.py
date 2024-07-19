#!/usr/bin/env python3
"""
This is a script for copying the HTCondor Global Event Log (GEL) file to a new location, used to
scoop up logs whenever they're rotated

The script submits an HTCondor cron job (Crondor) that periodically checks the configured GEL location,
assumes an old rotated log has the `.old` extension, and copies it to the current working directory
if the last modified time of the `.old` file is newer than the last recorded time.

Note that the Crondor invokes this same script with the argument `crondor` to run the copy logic.
"""

import datetime
import htcondor
import os
import pathlib
import shutil
import sys
import time

'''
Determine the location of the configured Global Event Log (GEL) file.
'''
def get_gel_loc():
    try:
        event_log = htcondor.param["EVENT_LOG"]
        # convert to filepath
        event_log = pathlib.Path(event_log)
        return event_log
    except KeyError:
        print("EVENT_LOG parameter is not set in the HTCondor configuration.")
        return None

'''
Copy the gel_loc_old to the cwd with a timestamp appended to the filename.
This should happen from the context of the crondor's submit dir.
'''
def grab_gel_logs(gel_loc_old, tstamp):
    try:
        base_name = os.path.basename(gel_loc_old)
        file_name, file_extension = os.path.splitext(base_name)
        
        new_file_name = f"{file_name}-{str(tstamp).replace('.', '_')}"
        
        # Define the new file path in the current working directory
        new_file_path = os.path.join(os.getcwd(), new_file_name)
        
        # Copy the file to the new location
        shutil.copy(gel_loc_old, new_file_path)
        print(f"File copied to: {new_file_path}")
        return 0
    except Exception as e:
        print(f"Error copying file: {e}")
        return 1


'''
Submit the Crondor job responsible for determining whether
there's a new GEL file to copy, and copying it if necessary.
'''
def submitCrondor():
    script_path = str(pathlib.Path(sys.argv[0]).absolute())
    print("Script path: ", script_path)

    # check that the logs directory exists, and create it if not
    logs_dir = pathlib.Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir()

    submit_description = htcondor.Submit({
        "executable":              script_path,
        "arguments":               f"crondor",
        "universe":                "local",
        "request_disk":            "6GB",
        "request_cpus":            1,
        "request_memory":          512,
        "log":                     "logs/crondor_$(CLUSTER).log",
        "should_transfer_files":   "YES",
        "when_to_transfer_output": "ON_EXIT",
        "output":                  "logs/crondor_$(CLUSTER).out",
        "error":                   "logs/crondor_$(CLUSTER).err",

        # Cron directives. Set to run every day at 8am, from the timezone as configured at the AP.
        "cron_minute":             "0",
        "cron_hour":               "8",
        "cron_day_of_month":       "*",
        "cron_month":              "*",
        "cron_day_of_week":        "*",
        "on_exit_remove":          "false",

        # Specify `getenv` so that our script uses the appropriate environment
        # when it runs in local universe. This allows the crondor to access
        # modules we've installed in the base env.
        "getenv":                  "true",

        # Finally, set the batch name so we know what this job does.
        "JobBatchName":            f"GEL_log_crondor",
    })

    try:
        schedd = htcondor.Schedd()
        submit_result = schedd.submit(submit_description)
        print("Crondor job was submitted with JobID %d.0" % submit_result.cluster())
        return 0
    except Exception as e:
        print(f"Error submitting crondor job: {e}")
        return 1


'''
This is the function actually run by the crondor job.
'''
def crondorMain():
    # Get the GEL location, as it would be returned by `condor_config_val EVENT_LOG`
    gel_loc = get_gel_loc()
    if gel_loc is None:
        print("Could not determine the GEL location.")
    print("GEL Location: ", gel_loc)

    # Add .old as the extension
    gel_loc_old = gel_loc.with_suffix(".old")
    print("Old GEL Location: ", gel_loc_old)

    # Get last modification time of the old GEL
    last_mod_time = gel_loc_old.stat().st_mtime
    last_mod_time = datetime.datetime.fromtimestamp(last_mod_time)
    last_mod_time = last_mod_time.strftime('%Y-%m-%d_%H-%M-%S')
    print("Last modified time from stat call: ", last_mod_time)

    # Check if we have a current timestamp file. If not, create one and add the last modified time
    timestamp_file = pathlib.Path("gel_timestamp.txt")
    if not timestamp_file.exists():
        with open(timestamp_file, "w") as f:
            f.write(str(last_mod_time) + "\n")
            return grab_gel_logs(gel_loc_old, last_mod_time)

    # Read the timestamp file, compare the last modified time with the timestamp
    with open(timestamp_file, "r") as f:
        timestamp = f.readline().strip()
        print("Timestamp read from file: ", timestamp)        
    
        # If the last modified time is greater than the timestamp, copy the file
        if last_mod_time > timestamp:
            with open(timestamp_file, "w") as f:
                f.write(str(last_mod_time) + "\n")
                return grab_gel_logs(gel_loc_old, last_mod_time)
        else:
            print("No new logs to copy.")
            return 0

def submitterMain():
    return submitCrondor()

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in ["crondor"]:
            return crondorMain()

    return submitterMain()

if __name__ == '__main__':
    sys.exit(main())
