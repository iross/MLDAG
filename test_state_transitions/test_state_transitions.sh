#!/bin/bash
# Test state transitions by creating log files incrementally

cd "$(dirname "$0")"

echo "=== State Transition Test Suite ==="
echo ""

# Test 1: Full log parsing (baseline)
echo "Test 1: Full log parsing (baseline)"
echo "-----------------------------------"
python3 ../dagman_monitor.py test.dag
echo ""
echo "✓ Test 1 complete"
echo ""
read -p "Press Enter to continue to Test 2..."

# Test 2: Incremental log parsing (simulates tail mode issues)
echo ""
echo "Test 2: Incremental log parsing"
echo "--------------------------------"
echo "This simulates what happens in --tail mode by parsing the log"
echo "file multiple times as it grows."
echo ""

# Clear the growing log
rm -f metl_growing.log
touch metl_growing.log

# Function to show current status
show_status() {
    echo ""
    echo ">>> After chunk $1:"
    python3 ../dagman_monitor.py test.dag --metl-log metl_growing.log | grep -E "(test_job_|Status:)"
}

# Chunk 1: Submit and transfer job 99001
echo "Chunk 1: Submit + transfer start (99001)"
cat >> metl_growing.log << 'EOF'
000 (99001.000.000) 2025-12-15 10:00:01 Job submitted from host: <test.host>
    DAG Node: test_job_transfer
...
040 (99001.000.000) 2025-12-15 10:05:00 Started transferring input files
...
EOF
show_status 1

read -p "Press Enter for chunk 2..."

# Chunk 2: Finish transfer, start execution
echo ""
echo "Chunk 2: Transfer finish + execution start (99001)"
cat >> metl_growing.log << 'EOF'
040 (99001.000.000) 2025-12-15 10:08:00 Finished transferring input files
...
001 (99001.000.000) 2025-12-15 10:08:30 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
EOF
show_status 2
echo ""
echo "⚠️  CHECK: Is test_job_transfer RUNNING or stuck in TRANSFERRING?"

read -p "Press Enter for chunk 3..."

# Chunk 3: Job 99002 starts with transfer
echo ""
echo "Chunk 3: Submit + transfer (99002)"
cat >> metl_growing.log << 'EOF'
000 (99002.000.000) 2025-12-15 10:00:06 Job submitted from host: <test.host>
    DAG Node: test_job_hold
...
040 (99002.000.000) 2025-12-15 10:06:00 Started transferring input files
...
040 (99002.000.000) 2025-12-15 10:09:00 Finished transferring input files
...
001 (99002.000.000) 2025-12-15 10:09:30 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
EOF
show_status 3

read -p "Press Enter for chunk 4..."

# Chunk 4: Job 99002 gets held
echo ""
echo "Chunk 4: Job hold (99002)"
cat >> metl_growing.log << 'EOF'
012 (99002.000.000) 2025-12-15 10:12:00 Job was held.
	Excessive CPU usage
	Code 26 Subcode 102
...
EOF
show_status 4
echo ""
echo "⚠️  CHECK: Is test_job_hold now HELD?"

read -p "Press Enter for chunk 5..."

# Chunk 5: Job 99002 gets released
echo ""
echo "Chunk 5: Job release (99002)"
cat >> metl_growing.log << 'EOF'
013 (99002.000.000) 2025-12-15 10:15:00 Job was released.
	via condor_release (by user test)
...
EOF
show_status 5
echo ""
echo "⚠️  CHECK: Is test_job_hold now IDLE (after release)?"

read -p "Press Enter for chunk 6..."

# Chunk 6: Re-execution and completion
echo ""
echo "Chunk 6: Re-execution + completion"
cat >> metl_growing.log << 'EOF'
001 (99002.000.000) 2025-12-15 10:16:00 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
005 (99001.000.000) 2025-12-15 10:15:00 Job terminated.
	(1) Normal termination (return value 0)
	1000000  -  Total Bytes Sent By Job
	2000000  -  Total Bytes Received By Job
...
005 (99002.000.000) 2025-12-15 10:20:00 Job terminated.
	(1) Normal termination (return value 0)
	1500000  -  Total Bytes Sent By Job
	2500000  -  Total Bytes Received By Job
...
000 (99003.000.000) 2025-12-15 10:00:11 Job submitted from host: <test.host>
    DAG Node: test_job_normal
...
001 (99003.000.000) 2025-12-15 10:02:00 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
005 (99003.000.000) 2025-12-15 10:10:00 Job terminated.
	(1) Normal termination (return value 0)
	1200000  -  Total Bytes Sent By Job
	2200000  -  Total Bytes Received By Job
...
EOF
show_status 6

echo ""
echo "=== Test Complete ==="
echo ""
echo "Issues to look for:"
echo "1. Did test_job_transfer transition from TRANSFERRING to RUNNING?"
echo "2. Did test_job_hold show HELD status?"
echo "3. Did test_job_hold transition from HELD to IDLE after release?"
