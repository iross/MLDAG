# Machine-Learning-for-OSPool-Failure-Classification

Given a log file "EventLog", the sequence for preprocessing is as follows:

./geld.py EventLog # Parses the global event log categorized by job id into a JSON.

./filter.py geld_out.json # Filters out jobs that does not have hold>release state sequence.

./label.py filter_out.json # Labels the jobs as transient or non-transient.

./logs2ts.py label_out.json # Converts into a time-series data format.

The final processed json file is "logs2ts_out.json".
