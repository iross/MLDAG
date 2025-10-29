#!/bin/bash
# Quick incremental test to verify state transitions

cd "$(dirname "$0")"

echo "=== Testing State Transition Fixes ==="
echo ""

# Clear growing log
rm -f metl_growing.log
touch metl_growing.log

echo "Step 1: Job submitted and transfer starts"
cat > metl_growing.log << 'EOF'
000 (99001.000.000) 2025-12-15 10:00:01 Job submitted from host: <test.host>
    DAG Node: test_job_transfer
...
040 (99001.000.000) 2025-12-15 10:05:00 Started transferring input files
...
EOF

echo "Running monitor..."
python3 ../dagman_monitor.py test.dag --metl-log metl_growing.log --once 2>&1 | grep -E "test_job_transfer|TRANSFER"
echo ""
echo "Expected: test_job_transfer should show TRANSFERRING"
echo ""
read -p "Press Enter for Step 2..."

echo ""
echo "Step 2: Transfer finishes, execution starts"
cat >> metl_growing.log << 'EOF'
040 (99001.000.000) 2025-12-15 10:08:00 Finished transferring input files
...
001 (99001.000.000) 2025-12-15 10:08:30 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
	AvailableGPUs = { GPUs_GPU_test }
...
EOF

echo "Running monitor..."
python3 ../dagman_monitor.py test.dag --metl-log metl_growing.log --once 2>&1 | grep -E "test_job_transfer|RUNNING"
echo ""
echo "✓ CHECK: test_job_transfer should NOW show RUNNING (not TRANSFERRING)"
echo ""
read -p "Press Enter for Step 3..."

echo ""
echo "Step 3: Job completes"
cat >> metl_growing.log << 'EOF'
005 (99001.000.000) 2025-12-15 10:15:00 Job terminated.
	(1) Normal termination (return value 0)
	1000000  -  Total Bytes Sent By Job
	2000000  -  Total Bytes Received By Job
...
EOF

echo "Running monitor..."
python3 ../dagman_monitor.py test.dag --metl-log metl_growing.log --once 2>&1 | grep -E "test_job_transfer|COMPLETED"
echo ""
echo "Expected: test_job_transfer should show COMPLETED"
echo ""
read -p "Press Enter for Step 4 (test HELD transition)..."

echo ""
echo "Step 4: Test HELD transition"
cat >> metl_growing.log << 'EOF'
000 (99002.000.000) 2025-12-15 10:00:06 Job submitted from host: <test.host>
    DAG Node: test_job_hold
...
001 (99002.000.000) 2025-12-15 10:09:30 Job executing on host: <test-exec.host>
	SlotName: slot1@test-exec.host
...
012 (99002.000.000) 2025-12-15 10:12:00 Job was held.
	Excessive CPU usage
...
EOF

echo "Running monitor..."
python3 ../dagman_monitor.py test.dag --metl-log metl_growing.log --once 2>&1 | grep -E "test_job_hold|HELD"
echo ""
echo "Expected: test_job_hold should show HELD"
echo ""
read -p "Press Enter for Step 5 (test RELEASED transition)..."

echo ""
echo "Step 5: Test RELEASED → IDLE transition"
cat >> metl_growing.log << 'EOF'
013 (99002.000.000) 2025-12-15 10:15:00 Job was released.
	via condor_release (by user test)
...
EOF

echo "Running monitor..."
python3 ../dagman_monitor.py test.dag --metl-log metl_growing.log --once 2>&1 | grep -E "test_job_hold|IDLE"
echo ""
echo "✓ CHECK: test_job_hold should show IDLE (after release)"
echo ""

echo "=== Test Complete ==="
echo ""
echo "Summary of what should work now:"
echo "1. ✓ TRANSFERRING → RUNNING (when job starts executing)"
echo "2. ✓ RUNNING → HELD (when job is held)"
echo "3. ✓ HELD → IDLE (when job is released)"
