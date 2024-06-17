# Machine-Learning-for-OSPool-Failure-Classification

Given a log file "EventLog", the sequence for preprocessing is as follows:
>./geld.py EventLog
>./filter.py geld_out.json
>./label.py filter_out.json
>./logs2ts.py label_out.json

The final processed json file is "logs2ts_out.json".
