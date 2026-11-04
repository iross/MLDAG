---
id: task-6
title: Track the GPU model for each job
status: Done
assignee: []
created_date: '2025-07-07'
updated_date: '2025-07-07'
labels: []
dependencies: []
priority: high
---

## Description

Within the job event log (metl.log in this case), there is information about the GPU (or GPUs) that the job used. 
For each job, track:
1. The number of GPUs
2. The model of GPUs (DeviceName)
3. The memory of the GPU (GlobalMemoryMb)

## Acceptance Criteria
- [x] Each job in the CSV report has a "Number of GPUs", "DeviceName", and "GlobalMemoryMb" column
- [x] The verbose option of the tables should include "Number of GPUs", "DeviceName", and "GlobalMemoryMb"

## Implementation Notes

Successfully implemented GPU tracking for each job. Added gpu_count, gpu_device_name, and gpu_memory_mb fields to JobInfo dataclass. Created _parse_gpu_info_from_metl() method to extract GPU information from metl.log job execution events. Updated CSV export to include 'Number of GPUs', 'DeviceName', and 'GlobalMemoryMb' columns. Added GPU columns to verbose table display with proper formatting and colors. All tests passing. GPU information is automatically extracted and displayed for all jobs that have GPU assignments.
