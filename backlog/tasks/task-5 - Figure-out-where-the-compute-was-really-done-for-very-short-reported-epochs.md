---
id: task-5
title: Figure out where the compute was really done for very short reported epochs
status: To Do
assignee: []
created_date: '2025-07-07'
labels: []
dependencies: []
---

## Description
Some epochs report very short (~20 minute) runtimes. I expect the training process will take a few hours. A short runtime like this indicates that the job started and found a previously computed checkpoint and exited with a success code after doing some validation. We should see if there are any other jobs associated with these epochs and report on the longest running job that exited successfully. Unfortunately, this information may not always be present.

One example is run0-train_epoch20. I think the work may have been completed successfully but with an exit code (status) 85. 