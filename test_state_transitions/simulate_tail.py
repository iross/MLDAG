#!/usr/bin/env python3
"""
Simulate tail mode for testing state transitions.

This script writes log lines incrementally to test files to simulate
the behavior of tail -f on a growing log file.
"""

import time
import sys
from pathlib import Path

def simulate_metl_log_growth():
    """Write metl.log content line by line with delays."""

    # Full log content split into timed chunks
    chunks = [
        # Initial submission and transfer for job 99001
        (0, """000 (99001.000.000) 2025-12-15 10:00:01 Job submitted from host: <test.host>
    DAG Node: test_job_transfer
...
"""),
        (2, """040 (99001.000.000) 2025-12-15 10:05:00 Started transferring input files
...
"""),
        (3, """040 (99001.000.000) 2025-12-15 10:08:00 Finished transferring input files
...
"""),
        (2, """001 (99001.000.000) 2025-12-15 10:08:30 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
"""),
        (4, """005 (99001.000.000) 2025-12-15 10:15:00 Job terminated.
	(1) Normal termination (return value 0)
	1000000  -  Total Bytes Sent By Job
	2000000  -  Total Bytes Received By Job
...
"""),
        # Job 99002 with hold/release cycle
        (1, """000 (99002.000.000) 2025-12-15 10:00:06 Job submitted from host: <test.host>
    DAG Node: test_job_hold
...
"""),
        (2, """040 (99002.000.000) 2025-12-15 10:06:00 Started transferring input files
...
"""),
        (3, """040 (99002.000.000) 2025-12-15 10:09:00 Finished transferring input files
...
"""),
        (2, """001 (99002.000.000) 2025-12-15 10:09:30 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
"""),
        (3, """012 (99002.000.000) 2025-12-15 10:12:00 Job was held.
	Excessive CPU usage
	Code 26 Subcode 102
...
"""),
        (3, """013 (99002.000.000) 2025-12-15 10:15:00 Job was released.
	via condor_release (by user test)
...
"""),
        (2, """001 (99002.000.000) 2025-12-15 10:16:00 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
"""),
        (4, """005 (99002.000.000) 2025-12-15 10:20:00 Job terminated.
	(1) Normal termination (return value 0)
	1500000  -  Total Bytes Sent By Job
	2500000  -  Total Bytes Received By Job
...
"""),
        # Job 99003 - simple job with no transfer events
        (1, """000 (99003.000.000) 2025-12-15 10:00:11 Job submitted from host: <test.host>
    DAG Node: test_job_normal
...
"""),
        (2, """001 (99003.000.000) 2025-12-15 10:02:00 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
"""),
        (5, """005 (99003.000.000) 2025-12-15 10:10:00 Job terminated.
	(1) Normal termination (return value 0)
	1200000  -  Total Bytes Sent By Job
	2200000  -  Total Bytes Received By Job
...
"""),
    ]

    output_file = Path("metl_growing.log")
    output_file.write_text("")  # Clear file

    print("Simulating metl.log growth...")
    print("Watch with: python3 ../dagman_monitor.py test.dag --tail")
    print("\nWriting log chunks:\n")

    for delay, content in chunks:
        time.sleep(delay)

        # Write chunk
        with open(output_file, 'a') as f:
            f.write(content)

        # Show what we wrote
        lines = content.strip().split('\n')
        first_line = lines[0] if lines else ""
        print(f"[t+{delay}s] Wrote: {first_line[:80]}...")
        sys.stdout.flush()

    print("\n✓ Simulation complete!")
    print(f"Final log written to: {output_file}")

if __name__ == "__main__":
    simulate_metl_log_growth()
