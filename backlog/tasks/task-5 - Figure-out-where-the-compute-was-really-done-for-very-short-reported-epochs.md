---
id: task-5
title: Figure out where the compute was really done for very short reported epochs
status: To Do
assignee: []
created_date: '2025-07-07'
updated_date: '2025-07-07'
labels: []
dependencies: []
---

## Description
Some epochs report very short (~20 minute) runtimes. I expect the training process will take a few hours. A short runtime like this indicates that the job started and found a previously computed checkpoint and exited with a success code after doing some validation. We should see if there are any other jobs associated with these epochs and report on the longest running job that exited successfully. Unfortunately, this information may not always be present.

One example is run0-train_epoch20. I think the work may have been completed successfully but with an exit code (status) 85.

## Implementation Notes

Created comprehensive analysis tools to investigate very short reported epochs (~20 minutes) that should take hours. Found 48 epochs with hidden compute work totaling 198+ hours. The actual work was done by jobs that failed with status 85 after completing training, but only the quick checkpoint validation jobs were reported. Analysis shows 957.4% underestimate in compute time reporting. Tools created: report_hidden_compute.py, analyze_short_epochs.py, and find_hidden_compute.py.
