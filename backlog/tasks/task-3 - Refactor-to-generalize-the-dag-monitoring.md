---
id: task-3
title: Refactor to generalize the dag monitoring
status: To Do
assignee: []
created_date: '2025-07-07'
labels: []
dependencies: []
---

## Description
This code is still heavily specialized. This task is to generalize the dagman_monitor.py script so that it can be utilized in different workflows. The log parsing and monitoring pieces should be generally useful, including information about the dag nodes. However the code should be updated so taht the overall structure of the dag doesn't matter. External things (such as training runs and unique uuids and epochs) should still exist for my particular usecase, but should not be part of the primary framework of the code.

## Other notes
- This work should be done on a new feature git branch.
- The "base mode" should only read from the dag, associated logs, and the job log.