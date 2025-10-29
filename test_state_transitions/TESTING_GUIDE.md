# State Transition Testing Guide

## Summary

I've created a complete test suite to investigate state transition issues in `dagman_monitor.py --tail` mode.

## Issue Report

Some state transitions aren't being reflected correctly in --tail mode:
- **TRANSFERRING → RUNNING**: May not update when execution starts after transfer finishes
- **RUNNING → HELD**: This seems to work
- **HELD → IDLE**: May not update after job is released

## Test Files Created

```
test_state_transitions/
├── README.md                    # Detailed documentation
├── TESTING_GUIDE.md            # This file
├── test.dag                    # Simple 3-job DAG
├── test.sub                    # Dummy submit file
├── test.dag.dagman.out         # Cluster-to-job mappings
├── metl.log                    # Complete reference log
├── simulate_tail.py            # Script to simulate log growth
└── test_state_transitions.sh   # Interactive test script
```

## Quick Start

### Option 1: Interactive Test Script (Recommended)

```bash
cd test_state_transitions
./test_state_transitions.sh
```

This script:
1. Shows baseline (full log parsing)
2. Then incrementally adds log lines to simulate tail mode
3. Pauses at each state transition for inspection
4. Highlights what to check at each step

### Option 2: Manual Testing

**Terminal 1** - Start monitor in tail mode:
```bash
cd test_state_transitions
python3 ../dagman_monitor.py test.dag --tail --metl-log metl_growing.log
```

**Terminal 2** - Simulate log growth:
```bash
cd test_state_transitions
python3 simulate_tail.py
```

### Option 3: Baseline Comparison

Test full log parsing (should work correctly):
```bash
cd test_state_transitions
python3 ../dagman_monitor.py test.dag
```

## Test Jobs

| Job Name | Cluster ID | Purpose |
|----------|------------|---------|
| test_job_transfer | 99001 | Tests TRANSFERRING → RUNNING transition |
| test_job_hold | 99002 | Tests RUNNING → HELD → IDLE → RUNNING cycle |
| test_job_normal | 99003 | Control (no transfer events, simple execution) |

## Expected State Transitions

### Job 99001 (test_job_transfer)
```
IDLE (submit)
  ↓
TRANSFERRING (040: Started transferring input files)
  ↓
TRANSFERRING (040: Finished transferring input files)  
  ↓
❌ RUNNING (001: Job executing) ← MAY STAY STUCK IN TRANSFERRING
  ↓
COMPLETED (005: Job terminated)
```

### Job 99002 (test_job_hold)
```
IDLE → TRANSFERRING → RUNNING
  ↓
✓ HELD (012: Job was held) ← SEEMS TO WORK
  ↓
❌ IDLE (013: Job was released) ← MAY NOT UPDATE
  ↓
RUNNING (001: Job executing again)
  ↓
COMPLETED (005: Job terminated)
```

### Job 99003 (test_job_normal)
```
IDLE → RUNNING → COMPLETED
(No transfer events - should work correctly)
```

## What to Look For

1. **After transfer finishes and execution starts**:
   - Does job 99001 show `RUNNING` or is it stuck at `TRANSFERRING`?

2. **When job gets held**:
   - Does job 99002 update to `HELD` status? (This seems to work)

3. **After job is released**:
   - Does job 99002 show `IDLE` or does it keep showing `HELD`?

## Code Areas to Investigate

If bugs are found, check these areas in `dagman_monitor.py`:

1. **Lines 582-594**: Event 040 (transfer) handling
   - Updates `current_status` to 'transferring_input' and 'transferring_output'
   - Check if status gets cleared when transfer finishes

2. **Lines 689-698**: `_update_status_if_newer()`
   - Only updates if timestamp is newer
   - Could this prevent RUNNING from overriding TRANSFERRING?

3. **Lines 909-930**: `apply_metl_data_to_all_jobs()`
   - Maps metl current_status to JobStatus enum
   - Check if 'transferring_input' → TRANSFERRING gets overridden by execution

4. **Lines 533-538**: Event 001 (execution) handling
   - Sets start_time but may not update current_status
   - This could be the bug: execution doesn't override transfer status

## Potential Fixes

Based on the code inspection, potential issues:

### Issue 1: Execution (001) doesn't clear transfer status
```python
# Current code (line 533-538):
elif event_code == '001':  # Job execution
    self.metl_job_timing[cluster_id]['start_time'] = timestamp
    # ❌ No update to current_status here!

# Potential fix:
elif event_code == '001':  # Job execution
    self.metl_job_timing[cluster_id]['start_time'] = timestamp
    self._update_status_if_newer(cluster_id, timestamp, 'running')  # ✓ Add this
```

### Issue 2: Transfer finish doesn't update status
```python
# Line 586-587:
elif 'Finished transferring input files' in line:
    self.metl_job_timing[cluster_id]['transfer_input_end'] = timestamp
    # ❌ No status update - should clear transferring?

# Potential fix:
elif 'Finished transferring input files' in line:
    self.metl_job_timing[cluster_id]['transfer_input_end'] = timestamp
    self._update_status_if_newer(cluster_id, timestamp, 'ready_to_run')  # ✓ Add this
```

## Next Steps

1. Run the tests to confirm the bugs exist
2. Verify the suspected code locations
3. Apply fixes and re-test
4. Ensure fixes don't break other functionality
