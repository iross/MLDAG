# State Transition Testing for dagman_monitor.py

This directory contains test files to verify that state transitions work correctly in `--tail` mode.

## Problem

Some state transitions don't appear to be reflected correctly in --tail mode:
- TRANSFERRING → RUNNING: May not update properly
- RUNNING → HELD: Seems to work
- HELD → IDLE (after release): May not update properly

## Test Setup

### Files

1. **test.dag** - Simple DAG with 3 jobs
2. **test.dag.dagman.out** - DAGMan output with cluster→job mappings
3. **metl.log** - Complete log file (for reference/comparison)
4. **simulate_tail.py** - Script to simulate log growth for tail testing

### Test Jobs

- **test_job_transfer** (99001): Tests TRANSFERRING → RUNNING transition
- **test_job_hold** (99002): Tests RUNNING → HELD → IDLE → RUNNING transitions
- **test_job_normal** (99003): Simple job with no transfer events

## Running Tests

### Test 1: Full log parsing (baseline)

```bash
cd test_state_transitions
python3 ../dagman_monitor.py test.dag
```

Expected behavior:
- All jobs should show COMPLETED status
- Timing information should be correct

### Test 2: Tail mode simulation

Terminal 1 (monitor):
```bash
cd test_state_transitions
python3 ../dagman_monitor.py test.dag --tail --metl-log metl_growing.log
```

Terminal 2 (simulate log growth):
```bash
cd test_state_transitions
python3 simulate_tail.py
```

### What to Watch For

1. **Job 99001 (test_job_transfer)**:
   - Should transition: IDLE → TRANSFERRING → RUNNING → COMPLETED
   - **BUG**: May stay in TRANSFERRING even after execution starts

2. **Job 99002 (test_job_hold)**:
   - Should transition: IDLE → TRANSFERRING → RUNNING → HELD → IDLE → RUNNING → COMPLETED
   - **BUG**: May not update to IDLE after release

3. **Job 99003 (test_job_normal)**:
   - Should transition: IDLE → RUNNING → COMPLETED
   - This should work correctly (no transfer events)

## Expected Timeline

| Time | Job 99001 | Job 99002 | Job 99003 | Event |
|------|-----------|-----------|-----------|-------|
| +0s  | IDLE      | -         | -         | Submit 99001 |
| +2s  | TRANSFERRING | -      | -         | Transfer start 99001 |
| +5s  | TRANSFERRING | -      | -         | Transfer finish 99001 |
| +7s  | **RUNNING** | -        | -         | Execute 99001 (CHECK: Does it update?) |
| +11s | COMPLETED | IDLE      | -         | Terminate 99001, Submit 99002 |
| +13s | COMPLETED | TRANSFERRING | -      | Transfer start 99002 |
| +16s | COMPLETED | TRANSFERRING | -      | Transfer finish 99002 |
| +18s | COMPLETED | RUNNING   | -         | Execute 99002 |
| +21s | COMPLETED | **HELD**  | -         | Hold 99002 (CHECK: Does it update?) |
| +24s | COMPLETED | **IDLE**  | -         | Release 99002 (CHECK: Does it update?) |
| +26s | COMPLETED | RUNNING   | IDLE      | Re-execute 99002, Submit 99003 |
| +28s | COMPLETED | RUNNING   | RUNNING   | Execute 99003 |
| +30s | COMPLETED | COMPLETED | RUNNING   | Terminate 99002 |
| +35s | COMPLETED | COMPLETED | COMPLETED | Terminate 99003 |

## Debugging

To see detailed state tracking, check the code in dagman_monitor.py:

1. `_update_status_if_newer()` - Updates current_status in metl_job_timing
2. `apply_metl_data_to_all_jobs()` - Applies status to JobInfo objects
3. Look for lines 909-930 where current_status is mapped to JobStatus

## Known Issues to Investigate

1. Transfer events (040) update current_status, but does it propagate to display?
2. After "Finished transferring input files", does it clear TRANSFERRING?
3. After execution (001) starts, does it override the TRANSFERRING status?
